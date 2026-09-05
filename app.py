import io
import hashlib
import os
import re
import shutil
from datetime import date
import calendar
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader
import pytesseract
import fitz
import pdfplumber
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from auth import account_username, authenticate, create_account, has_account
from engine import classify, clean, flag_mask, generate_alerts, infer_direction, merchant_name, normalize, recommendations
from storage import (
    init_storage,
    add_transaction,
    deactivate_bill,
    deactivate_account,
    learn_category_rule,
    list_audit_log,
    list_accounts,
    list_bills,
    list_budgets,
    list_category_rules,
    list_keyword_instructions,
    list_category_catalog,
    list_category_examples,
    list_attachments,
    list_goals,
    list_reconciliations,
    list_uploads,
    load_transactions,
    save_bill,
    save_category,
    save_keyword_instruction,
    save_attachment,
    save_account,
    save_ebook,
    save_goal,
    save_reconciliation,
    save_upload,
    split_transaction,
    restore_last_change,
    matching_transaction_ids,
    update_goal,
    update_transaction,
    upsert_budget,
)
from knowledge import ask as ask_mathewzinho, index_pdfs, ollama_status, parse_category_instruction
from backup import apply_pending_restore, create_backup, create_snapshot, restore_backup
from demo import load_demo_data
from insights import decision_priorities, forecast_categories, health_score, subscription_changes, weekly_summary, what_changed


def configure_tesseract():
    """Localiza o Tesseract também quando o Windows não atualizou o PATH."""
    detected = shutil.which("tesseract")
    candidates = [
        detected,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


TESSERACT_PATH = configure_tesseract()


def ocr_pdf(pdf_bytes: bytes) -> str:
    """Lê PDFs escaneados com várias imagens, modos e configurações do OCR."""
    if not TESSERACT_PATH:
        raise pytesseract.pytesseract.TesseractNotFoundError()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for pdf_page in document:
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        original = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        gray = ImageOps.grayscale(original)
        contrast = ImageEnhance.Contrast(gray).enhance(2.0).filter(ImageFilter.SHARPEN)
        binary = contrast.point(lambda pixel: 0 if pixel < 170 else 255)
        denoised = ImageOps.autocontrast(gray).filter(ImageFilter.MedianFilter(size=3))
        readings = []
        for image in (original, gray, contrast, binary, denoised):
            for psm in (3, 6, 11, 12):
                try:
                    readings.append(pytesseract.image_to_string(image, lang="por+eng", config=f"--psm {psm}"))
                except pytesseract.TesseractError:
                    readings.append(pytesseract.image_to_string(image, lang="eng", config=f"--psm {psm}"))
        pages_text.append(max(readings, key=lambda value: len(re.sub(r"\s+", "", value)), default=""))
    document.close()
    return "\n".join(pages_text)


def money_to_float(token: str) -> float:
    token = token.replace("R$", "").replace(" ", "").strip()
    if "," in token:
        return float(token.replace(".", "").replace(",", "."))
    return float(token.replace(",", ""))


def rows_from_lines(lines):
    rows = []
    date_pattern = r"(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)"
    money_pattern = r"(?:R\$\s*)?-?\d{1,3}(?:[. ]\d{3})*,\d{2}|(?:R\$\s*)?-?\d+[.,]\d{2}"
    for line_number, line in enumerate(lines, 1):
        date_match = re.search(date_pattern, line)
        if not date_match:
            continue
        matches = list(re.finditer(money_pattern, line))
        if not matches:
            continue
        # Em extratos com coluna de saldo, o primeiro valor após a data costuma
        # ser o movimento. Mantemos as quantias restantes para auditoria.
        chosen = matches[0]
        values = [m.group(0) for m in matches]
        description = line.replace(date_match.group(1), "")
        for match in reversed(matches):
            description = description.replace(match.group(0), "", 1)
        description = re.sub(r"\s+", " ", description).strip(" -|;")
        try:
            numeric = money_to_float(chosen.group(0))
        except ValueError:
            continue
        normalized_line = clean(line)
        incoming = ("recebido", "recebimento", "credito", "deposito", "salario", "estorno", "resgate", "reembolso", "cashback", "dividendo", "entrada", "recebldo")
        outgoing = ("enviado", "envio", "debito", "saque", "compra", "pagamento", "tarifa", "juros", "parcela", "fatura", "retirada", "saida", "envlado")
        if any(token in normalized_line for token in incoming) and not any(token in normalized_line for token in outgoing):
            direction = "Entrada"
        elif any(token in normalized_line for token in outgoing) and not any(token in normalized_line for token in incoming):
            direction = "Saída"
        elif "-" in chosen.group(0):
            direction = "Saída"
        else:
            direction = "Indeterminada"
        rows.append({"Data": date_match.group(1), "Descrição": description, "Valor": numeric, "Direção": direction, "Valores encontrados": " | ".join(values), "Revisão": "Verificar múltiplos valores" if len(values) > 1 else "OK", "Linha OCR": line_number})
    return pd.DataFrame(rows)


def coordinate_ocr_rows(pdf_bytes: bytes):
    """Reconstrói linhas a partir das coordenadas das palavras do OCR."""
    if not TESSERACT_PATH:
        raise pytesseract.pytesseract.TesseractNotFoundError()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_rows, all_text = [], []
    for pdf_page in document:
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        gray = ImageOps.grayscale(image)
        variants = [image, gray, ImageOps.autocontrast(gray)]
        for variant in variants:
            for psm in (3, 6, 11, 12):
                data = pytesseract.image_to_data(variant, lang="por+eng", config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
                groups = {}
                for i, word in enumerate(data["text"]):
                    word = word.strip()
                    try:
                        confidence = float(data["conf"][i])
                    except (TypeError, ValueError):
                        confidence = 0
                    if not word or confidence < 15:
                        continue
                    key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                    groups.setdefault(key, []).append((data["left"][i], word))
                lines = [" ".join(word for _, word in sorted(words)) for words in groups.values()]
                candidate = rows_from_lines(lines)
                if not candidate.empty:
                    all_rows.append(candidate)
                all_text.extend(lines)
    document.close()
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return result, "\n".join(all_text)


def pdf_to_rows(pdf_bytes: bytes):
    # Primeiro tenta extrair tabelas estruturadas de PDFs digitais.
    table_rows = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    for row in table:
                        if row and any(cell for cell in row):
                            table_rows.append([str(cell or "") for cell in row])
    except Exception:
        table_rows = []

    candidates = []
    if table_rows:
        candidates.append(rows_from_lines([" | ".join(row) for row in table_rows]))

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pypdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    candidates.append(rows_from_lines(pypdf_text.splitlines()))

    fitz_lines = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            fitz_lines.extend(block[4] for block in page.get_text("blocks") if len(block) >= 5)
    candidates.append(rows_from_lines(fitz_lines))

    # OCR só é acionado quando as extrações digitais não produzem uma tabela
    # utilizável; isso evita exigir Tesseract em PDFs que já têm texto nativo.
    if not any(len(candidate) >= 2 for candidate in candidates):
        coordinate_rows, coordinate_text = coordinate_ocr_rows(pdf_bytes)
        candidates.append(coordinate_rows)
    else:
        coordinate_text = ""

    nonempty = [candidate for candidate in candidates if not candidate.empty]
    if not nonempty:
        return pd.DataFrame(), pypdf_text or coordinate_text
    # Prioriza o conjunto com mais lançamentos, mas remove duplicatas entre
    # passagens e preserva a indicação de múltiplos valores para auditoria.
    result = max(nonempty, key=lambda candidate: len(candidate))
    result = result.drop_duplicates(subset=["Data", "Descrição", "Valor"]).reset_index(drop=True)
    return result, pypdf_text or coordinate_text

st.set_page_config(page_title="FINP MAT", page_icon="💰", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root { --green: #2e6bff; --green-soft: #eef3ff; --ink: #16233b; --muted: #708097; --line: #e8edf2; --soft: #f6f8fb; }
.block-container {max-width: 1520px; padding: 1.1rem 2.5rem 3.5rem;}
[data-testid="stSidebar"] {display:none;}
[data-testid="collapsedControl"] {display:none;}
[data-testid="stFileUploader"] {border: 1.5px dashed #b9c5c5; border-radius: 18px; padding: 13px; background: #fbfdfc;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid var(--line); padding: 16px; border-radius: 18px; box-shadow: 0 8px 24px rgba(20, 40, 60, .045);}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--line); border-radius: 18px; background: #fff; box-shadow: 0 8px 24px rgba(20, 40, 60, .035);}
h1, h2, h3 {letter-spacing: -0.035em; color: var(--ink);}
h1 {font-size: 2.1rem !important; margin-bottom: .2rem !important;}
h2 {font-size: 1.35rem !important;}
.brand {display:flex; align-items:center; gap:.7rem; padding:.2rem .35rem; min-height:42px; white-space:nowrap; font-size:1.28rem; font-weight:800; color:#172033; overflow:visible;}
.brand-mark {width:34px; height:34px; border-radius:10px; background:#eafaf4; display:grid; place-items:center;}
.brand-mark svg {width:23px; height:23px;}
.top-navigation {position:sticky; top:.25rem; z-index:100; background:rgba(255,255,255,.96); backdrop-filter:blur(12px); border:1px solid var(--line); border-radius:18px; padding:.55rem .7rem; margin-bottom:1.35rem; box-shadow:0 8px 25px rgba(15,35,55,.07);}
[data-testid="stVerticalBlockBorderWrapper"]:has(.brand) {position:sticky; top:.35rem; z-index:100; background:rgba(255,255,255,.97); backdrop-filter:blur(12px); box-shadow:0 10px 28px rgba(15,35,55,.08);}
[data-testid="stVerticalBlockBorderWrapper"]:has(.brand) [data-testid="stHorizontalBlock"] {align-items:center; gap:.7rem;}
[data-testid="stVerticalBlockBorderWrapper"]:has(.brand) .stButton > button {height:2.6rem; border:1px solid #dce5f4; border-radius:10px; background:#fff; color:#40506b; font-size:.82rem; font-weight:680; padding:.2rem .55rem; box-shadow:none; white-space:nowrap; overflow:visible;}
[data-testid="stVerticalBlockBorderWrapper"]:has(.brand) .stButton > button:hover {border-color:#a7befc; background:#f1f5ff; color:#2453d4;}
[data-testid="stVerticalBlockBorderWrapper"]:has(.brand) .stButton > button[kind="primary"] {background:#2e6bff; border-color:#2e6bff; color:#fff; box-shadow:0 6px 14px rgba(46,107,255,.20);}
[data-testid="stButton"] > button[kind="primary"] {background:#2e6bff; border-color:#2e6bff; color:#fff;}
.category-chip {padding:.65rem .85rem;border:1px solid #e6ecf4;border-radius:14px;background:#fff;min-height:84px;}
.category-chip strong{display:block;color:#172033;font-size:.92rem}.category-chip span{color:#7b879b;font-size:.78rem}
.review-card{border:1px solid #e7ecf3;border-radius:16px;padding:1rem;background:linear-gradient(135deg,#fff,#f8faff);}
.top-navigation .stButton > button {height:2.4rem; min-width:100%; border:1px solid #e6ecf1; border-radius:9px; background:#fff; color:#425169; font-size:.78rem; font-weight:650; padding:.2rem .3rem; box-shadow:none; white-space:nowrap;}
.top-navigation .stButton > button:hover {border-color:#9ce5ca; background:#effbf6; color:#087f58;}
.top-navigation .stButton > button[kind="primary"] {background:#0bbf83; border-color:#0bbf83; color:#fff;}
.metric-card {border:1px solid var(--line); border-radius:18px; background:#fff; padding:1rem 1.05rem; min-height:122px; box-shadow:0 8px 22px rgba(20,40,60,.035);}
.metric-title {font-size:.8rem; font-weight:650; color:#506076; display:flex; justify-content:space-between; align-items:center;}
.metric-value {font-weight:800; font-size:1.42rem; margin:.55rem 0 .25rem; letter-spacing:-.03em;}
.metric-note {font-size:.74rem; color:#8793a5;}
.progress-track {background:#edf2f4; height:7px; border-radius:999px; overflow:hidden; margin:.55rem 0 .25rem;}
.progress-fill {height:100%; border-radius:999px; background:linear-gradient(90deg, #12b981, #45d4a5);}
.small-muted {color:#8390a2; font-size:.8rem;}
.topbar {padding-bottom: .25rem;}
@media (max-width: 900px) {.block-container {padding: .8rem 1rem 2rem;} .top-navigation {position:static; overflow-x:auto;} h1{font-size:1.7rem !important;} [data-testid="stHorizontalBlock"] {gap:.45rem;}}
</style>
""", unsafe_allow_html=True)

if st.session_state.get("finp_theme") == "Escuro":
    st.markdown("""<style>
    .stApp, [data-testid="stAppViewContainer"] {background:#101827;color:#edf2ff;}
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"], .metric-card, .category-chip, .review-card {background:#172235 !important;border-color:#2a3a54 !important;color:#edf2ff !important;}
    h1,h2,h3,.brand {color:#f5f8ff !important;} .small-muted,.metric-note {color:#aebbd0 !important;}
    [data-testid="stDataFrame"], [data-testid="stFileUploader"] {background:#172235;}
    </style>""", unsafe_allow_html=True)


def money_br(value: object) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        number = 0.0
    return f"R$ {number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def logo_html() -> str:
    return '<div class="brand"><span class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="#0bbf83" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M4 18V13M10 18V9M16 18V5M22 18V2"/><path d="m4 9 5-4 5 2 6-5"/></svg></span><span>FINP MAT</span></div>'


CATEGORY_ICONS = {
    "Alimentação": "🍽️", "Moradia": "🏠", "Transporte": "🚗", "Saúde": "❤️",
    "Educação": "📚", "Lazer": "🎮", "Financeiro": "🏦", "Transferências": "↔️",
    "Receitas": "↗️", "Não classificado": "✨",
}


def category_options(data: pd.DataFrame) -> list[str]:
    standard = list(CATEGORY_ICONS)
    catalog = list_category_catalog()
    registered = catalog["name"].tolist() if not catalog.empty else []
    observed = sorted(value for value in data.get("category", pd.Series(dtype=str)).dropna().unique().tolist() if value not in standard)
    return list(dict.fromkeys(standard + registered + observed))


def login_gate() -> None:
    now = time.time()
    if st.session_state.get("authenticated", False) and now - st.session_state.get("last_activity", now) > 15 * 60:
        st.session_state.authenticated = False
        st.warning("Sessão encerrada por inatividade. Entre novamente para proteger seus dados.")
    if st.session_state.get("authenticated", False):
        st.session_state.last_activity = now
        return
    st.markdown(logo_html(), unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.markdown("## Acesso ao FINP MAT")
        st.caption("Seu histórico financeiro fica protegido neste computador.")
        if not has_account():
            st.info("Primeiro acesso: crie uma senha local para proteger seus dados.")
            with st.form("create_account"):
                username = st.text_input("Nome de usuário", value="Matheus")
                password = st.text_input("Crie uma senha", type="password")
                confirmation = st.text_input("Confirme a senha", type="password")
                submitted = st.form_submit_button("Criar acesso", type="primary", use_container_width=True)
            if submitted:
                if len(password) < 6:
                    st.error("Use uma senha com pelo menos 6 caracteres.")
                elif password != confirmation:
                    st.error("As senhas não coincidem.")
                else:
                    create_account(username, password)
                    st.session_state.authenticated = True
                    st.session_state.last_activity = time.time()
                    st.rerun()
        else:
            with st.form("login"):
                username = st.text_input("Usuário", value=account_username())
                password = st.text_input("Senha", type="password")
                submitted = st.form_submit_button("Entrar", type="primary", use_container_width=True)
            if submitted:
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.last_activity = time.time()
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos.")
    st.stop()


def prepare_data(history: pd.DataFrame) -> pd.DataFrame:
    columns = ["id", "date", "description", "merchant", "amount", "direction", "direction_confidence", "category", "subcategory", "tags", "owner", "reimbursement_status", "confidence", "payment_method", "recurring", "installment_hint", "review_status", "user_note", "source_file"]
    if history.empty:
        return pd.DataFrame(columns=columns + ["description_clean", "anomaly"])
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["description"] = data["description"].fillna("")
    data["merchant"] = data["description"].map(merchant_name)
    data["description_clean"] = data["description"].map(clean)
    defaults = {"direction": "Indeterminada", "direction_confidence": 0.0, "confidence": 0.0, "recurring": 0, "installment_hint": 0, "payment_method": "Não identificado", "review_status": "Pendente", "user_note": "", "category": "Não classificado", "subcategory": "", "tags": "", "owner": "Pessoal", "reimbursement_status": "Não", "merchant": ""}
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default
        data[column] = data[column].fillna(default)
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0).abs()
    data["anomaly"] = False
    for _, group in data[data["direction"] == "Saída"].groupby("category"):
        median = group["amount"].median()
        if median > 0:
            data.loc[group.index, "anomaly"] = data.loc[group.index, "amount"] >= median * 2.5
    return data.sort_values("date").reset_index(drop=True)


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "tesseract" in lowered:
        return "OCR indisponível. Abra um novo PowerShell após instalar o Tesseract e tente novamente."
    if "password" in lowered or "senha" in lowered:
        return "Não foi possível abrir o conteúdo com essa senha."
    if "colunas" in lowered or "coluna" in lowered:
        return "Não reconheci a estrutura do arquivo. Use a prévia/auditoria para revisar os lançamentos extraídos."
    if "nenhum lançamento" in lowered:
        return "Não encontrei lançamentos neste arquivo. Pode ser um PDF protegido, escaneado sem nitidez ou fora do formato bancário."
    return "Não foi possível processar este arquivo. Ele foi mantido fora do histórico; tente novamente ou revise o PDF."


def process_imports() -> None:
    bank_profile = st.selectbox("Perfil de leitura", ["Detecção automática", "Nubank", "Inter", "Itaú", "Bradesco", "Caixa", "Santander", "Genérico"], help="Use o perfil do banco quando souber a origem. O modo automático mantém todas as estratégias de leitura ativas.")
    uploaded_files = st.file_uploader("Adicione um ou vários extratos", type=["pdf", "csv", "xlsx"], accept_multiple_files=True, key="statement_files", help="Você pode selecionar vários arquivos. Os arquivos já processados permanecem no histórico.")
    ebook_files = st.file_uploader("Adicione PDFs de educação financeira ao MATHEWZINHO", type=["pdf"], accept_multiple_files=True, key="ebook_files")
    for ebook in ebook_files or []:
        save_ebook(ebook.name, ebook.getvalue())
    if ebook_files:
        st.success(f"{len(ebook_files)} e-book(s) adicionado(s) à base do MATHEWZINHO.")
    st.session_state.setdefault("upload_queue", {})
    st.session_state.setdefault("processed_upload_hashes", set())
    st.session_state.setdefault("status_rows", [])
    for uploaded in uploaded_files or []:
        content = uploaded.getvalue()
        file_hash = hashlib.sha256(content).hexdigest()
        st.session_state.upload_queue[file_hash] = {"name": uploaded.name, "content": content}
    pending = [(file_hash, item) for file_hash, item in st.session_state.upload_queue.items() if file_hash not in st.session_state.processed_upload_hashes]
    if pending:
        st.info(f"{len(pending)} arquivo(s) aguardando processamento.")
    for file_hash, queued in pending:
        filename, content = queued["name"], queued["content"]
        try:
            if not st.session_state.get("snapshot_for_current_import"):
                create_snapshot("antes_importacao")
                st.session_state.snapshot_for_current_import = True
            if filename.lower().endswith(".csv"):
                raw = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
            elif filename.lower().endswith(".xlsx"):
                raw = pd.read_excel(io.BytesIO(content))
            else:
                with st.spinner(f"Analisando {filename} com OCR avançado..."):
                    try:
                        raw, _ = pdf_to_rows(content)
                    except pytesseract.pytesseract.TesseractNotFoundError:
                        raise ValueError("Tesseract não encontrado no Windows.")
                if raw.empty:
                    raise ValueError("Nenhum lançamento reconhecido")
            current_data = classify(normalize(raw), list_category_rules())
            saved, _ = save_upload(filename, content, current_data, bank_profile)
            st.session_state.status_rows.append({"Arquivo": filename, "Resultado": "Novo arquivo salvo" if saved else "Já estava salvo", "Lançamentos": len(current_data)})
            st.session_state.processed_upload_hashes.add(file_hash)
        except Exception as exc:
            st.session_state.status_rows.append({"Arquivo": filename, "Resultado": f"Erro: {friendly_error(exc)}", "Lançamentos": 0})
            st.session_state.processed_upload_hashes.add(file_hash)
    if not pending:
        st.session_state.pop("snapshot_for_current_import", None)


def metric_card(title: str, value: str, note: str, color: str, icon: str) -> str:
    return f'<div class="metric-card"><div class="metric-title"><span>{icon} &nbsp;{title}</span></div><div class="metric-value" style="color:{color}">{value}</div><div class="metric-note">{note}</div></div>'


def style_chart(fig: go.Figure, height: int = 310, currency_axis: str = "y") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=18, r=14, b=12, l=8),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color="#59677a", size=12),
        hoverlabel=dict(bgcolor="#172033", font_color="#ffffff", bordercolor="#172033"),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#edf1f5", tickfont=dict(color="#7e8b9d"))
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f5", zeroline=False, tickfont=dict(color="#7e8b9d"))
    if currency_axis == "x":
        fig.update_xaxes(tickprefix="R$ ", tickformat=",.0f")
    else:
        fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
    return fig


def category_chart(summary: pd.DataFrame) -> go.Figure:
    ordered = summary.sort_values("amount", ascending=True).tail(7)
    total = float(ordered["amount"].sum()) or 1.0
    figure = go.Figure(go.Bar(
        x=ordered["amount"],
        y=ordered["category"],
        orientation="h",
        marker=dict(color="#18bd88", line=dict(width=0)),
        text=[f"{value / total:.0%}" for value in ordered["amount"]],
        textposition="outside",
        textfont=dict(color="#66758a", size=11),
        hovertemplate="<b>%{y}</b><br>R$ %{x:,.2f}<extra></extra>",
    ))
    figure.update_layout(showlegend=False)
    figure.update_yaxes(showgrid=False, tickfont=dict(color="#334155"))
    return style_chart(figure, 300, currency_axis="x")


def cashflow_chart(data: pd.DataFrame) -> go.Figure:
    monthly = data.assign(month=data["date"].dt.to_period("M").astype(str)).pivot_table(index="month", columns="direction", values="amount", aggfunc="sum", fill_value=0).reset_index()
    for column in ("Entrada", "Saída"):
        if column not in monthly.columns:
            monthly[column] = 0.0
    monthly["Saldo"] = monthly["Entrada"] - monthly["Saída"]
    figure = go.Figure()
    figure.add_trace(go.Bar(name="Despesas", x=monthly["month"], y=monthly["Saída"], marker_color="#fa6b73", opacity=.84, hovertemplate="<b>Despesas</b><br>R$ %{y:,.2f}<extra></extra>"))
    figure.add_trace(go.Scatter(name="Entradas", x=monthly["month"], y=monthly["Entrada"], mode="lines+markers", line=dict(color="#11b981", width=3), marker=dict(size=7, color="#11b981"), hovertemplate="<b>Entradas</b><br>R$ %{y:,.2f}<extra></extra>"))
    figure.add_trace(go.Scatter(name="Saldo", x=monthly["month"], y=monthly["Saldo"], mode="lines", line=dict(color="#3b82f6", width=2, dash="dot"), hovertemplate="<b>Saldo</b><br>R$ %{y:,.2f}<extra></extra>"))
    return style_chart(figure, 310)


def render_overview(all_data: pd.DataFrame, period_data: pd.DataFrame, selected_month: str, goals: pd.DataFrame) -> None:
    label = "todos os períodos" if selected_month == "Todos os períodos" else selected_month
    st.markdown(f"# Olá, {account_username()}! 👋")
    st.caption(f"Aqui está o resumo das suas finanças em {label}.")
    income = period_data.loc[period_data["direction"] == "Entrada", "amount"].sum()
    expense = period_data.loc[period_data["direction"] == "Saída", "amount"].sum()
    balance = income - expense
    economy = (balance / income * 100) if income else 0
    recurring = period_data.loc[(period_data["direction"] == "Saída") & flag_mask(period_data, "recurring"), "amount"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Saldo do período", money_br(balance), "Entradas menos despesas", "#12875f", "▣"), unsafe_allow_html=True)
    c2.markdown(metric_card("Entradas", money_br(income), f"{int((period_data['direction'] == 'Entrada').sum())} lançamentos", "#1677e8", "↓"), unsafe_allow_html=True)
    c3.markdown(metric_card("Despesas", money_br(expense), f"{int((period_data['direction'] == 'Saída').sum())} lançamentos", "#d92d3e", "↗"), unsafe_allow_html=True)
    c4.markdown(metric_card("Economia", f"{economy:.0f}%", f"{money_br(recurring)} recorrentes", "#7d4be8", "%"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1.12])
    expenses = period_data[period_data["direction"] == "Saída"].copy()
    with left:
        with st.container(border=True):
            st.subheader("Despesas por categoria")
            summary = expenses.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            if summary.empty:
                st.info("Importe um extrato para visualizar a distribuição dos gastos.")
            else:
                st.plotly_chart(category_chart(summary), use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            st.subheader("Fluxo de caixa")
            if all_data.empty:
                st.info("A evolução aparecerá depois da importação.")
            else:
                st.plotly_chart(cashflow_chart(all_data), use_container_width=True, config={"displayModeBar": False})
    bottom_left, bottom_mid, bottom_right = st.columns([1.1, 1.1, .95])
    with bottom_left:
        with st.container(border=True):
            st.subheader("Orçamentos")
            budgets = list_budgets("global" if selected_month == "Todos os períodos" else selected_month)
            spent = expenses.groupby("category")["amount"].sum().to_dict()
            if budgets.empty:
                st.caption("Crie seus primeiros limites na seção Orçamentos.")
            else:
                for _, row in budgets.head(5).iterrows():
                    limit = float(row["limit_amount"] or 0)
                    used = float(spent.get(row["category"], 0))
                    percentage = min(100, used / limit * 100) if limit else 0
                    indicator = "⚠" if percentage >= 90 else "●"
                    st.markdown(f"**{indicator} {row['category']}** <span class='small-muted'>{money_br(used)} de {money_br(limit)} · {percentage:.0f}%</span>", unsafe_allow_html=True)
                    st.progress(percentage / 100 if limit else 0)
    with bottom_mid:
        with st.container(border=True):
            st.subheader("Últimos lançamentos")
            recent = period_data.sort_values("date", ascending=False).head(5)
            if recent.empty:
                st.caption("Nenhum lançamento disponível.")
            else:
                for _, row in recent.iterrows():
                    color = "#069b68" if row["direction"] == "Entrada" else "#e6384a"
                    sign = "+" if row["direction"] == "Entrada" else "-"
                    st.markdown(f"**{row['merchant'] or row['description'][:28]}**  \n<span class='small-muted'>{row['category']} · {row['date'].strftime('%d/%m/%Y')}</span><span style='float:right;color:{color}'>{sign}{money_br(row['amount'])}</span>", unsafe_allow_html=True)
                    st.divider()
    with bottom_right:
        with st.container(border=True):
            st.subheader("Metas")
            active = goals[goals["status"] != "Concluída"] if not goals.empty else goals
            if active.empty:
                st.write("Defina uma meta para acompanhar seu progresso.")
                if st.button("Criar meta", key="overview_goal"):
                    st.session_state.page = "Metas"
                    st.rerun()
            else:
                goal = active.iloc[0]
                target = float(goal["target_amount"] or 0)
                current = float(goal["current_amount"] or 0)
                progress = min(100, current / target * 100) if target else 0
                st.markdown(f"**{goal['name']}**")
                st.markdown(f"{money_br(current)} de {money_br(target)}")
                st.progress(progress / 100 if target else 0)
                st.caption(f"{progress:.0f}% concluída")
    st.subheader("Orientações do MATHEWZINHO")
    items = recommendations(period_data)
    budgets_for_alerts = list_budgets("global" if selected_month == "Todos os períodos" else selected_month)
    alerts = generate_alerts(period_data, budgets_for_alerts)
    for alert in alerts[:4]:
        renderer = st.error if alert["level"] == "alto" else st.warning
        renderer(f"**{alert['title']}** — {alert['text']}")
    if not items:
        if not alerts:
            st.success("Ainda não há alertas relevantes. Continue alimentando o histórico para receber análises melhores.")
    for item in items[:3]:
        st.info(f"**{item['title']}** — {item['text']}")
    st.subheader("O que mudou?")
    changes = what_changed(all_data)
    if not changes:
        st.caption("Importe pelo menos dois meses para comparar mudanças por categoria.")
    else:
        for change in changes[:3]:
            direction = "aumentou" if change["difference"] > 0 else "diminuiu"
            st.write(f"• **{change['category']}** {direction} {money_br(abs(change['difference']))} em relação ao período anterior.")
    health, health_notes = health_score(period_data, budgets_for_alerts)
    st.caption(f"Índice de organização financeira: **{health}/100** · {health_notes[0]}")


def render_imports() -> None:
    st.markdown("# Extratos")
    st.caption("Importe vários arquivos, acompanhe o processamento e mantenha o histórico organizado por período.")
    if st.session_state.get("status_rows"):
        st.dataframe(pd.DataFrame(st.session_state.status_rows), use_container_width=True, hide_index=True)
    uploads = list_uploads()
    st.subheader("Arquivos armazenados")
    if uploads.empty:
        st.info("Nenhum arquivo armazenado ainda.")
    else:
        st.dataframe(uploads, use_container_width=True, hide_index=True, column_config={"quality_score": st.column_config.ProgressColumn("Qualidade da leitura", min_value=0, max_value=1, format="%.0f%%"), "total_rows": "Linhas lidas", "review_rows": "A revisar"})
        low_quality = uploads[uploads["quality_score"] < .65]
        if not low_quality.empty:
            st.warning("Há extratos com qualidade baixa. Abra Lançamentos ou Auditoria para revisar antes de usar os gráficos como decisão final.")
    with st.expander("Importar texto de notificação bancária"):
        st.caption("Cole uma notificação de Pix, compra ou débito recebida no celular. O lançamento fica em revisão para sua confirmação.")
        with st.form("notification_import"):
            notification = st.text_area("Texto da notificação", placeholder="Ex.: Pix enviado de R$ 45,90 para João")
            notification_date = st.date_input("Data da notificação", value=date.today(), key="notification_date")
            if st.form_submit_button("Criar lançamento para revisar") and notification.strip():
                tokens = re.findall(r"-?\d{1,3}(?:\.\d{3})*,\d{2}|-?\d+[.,]\d{2}", notification)
                value = money_to_float(tokens[-1]) if tokens else 0.0
                direction, _ = infer_direction(notification, value)
                if value <= 0:
                    st.error("Não encontrei um valor. Informe um texto com valor em reais.")
                else:
                    add_transaction(notification_date.isoformat(), notification, value, direction, "Não classificado", "Notificação bancária", source_file="notificação")
                    st.success("Lançamento criado. Revise a categoria na Central de categorização visual.")
                    st.rerun()


def render_transactions(data: pd.DataFrame) -> None:
    st.markdown("# Lançamentos")
    st.caption("Consulte, pesquise e corrija os dados antes de tomar decisões.")
    pending_review = data[(data["review_status"] != "Concluído") | (data["direction"] == "Indeterminada")].copy() if not data.empty else pd.DataFrame()
    if not pending_review.empty:
        safe = pending_review[(pending_review["direction"] != "Indeterminada") & (pending_review["direction_confidence"] >= .85) & (pending_review["confidence"] >= .55)]
        with st.container(border=True):
            a, b, c = st.columns([1.2, 1.2, 3])
            a.metric("Pendências", len(pending_review))
            b.metric("Prontas para confirmar", len(safe))
            with c:
                st.caption("Confirme apenas leituras com direção e categoria confiáveis. As demais continuam na Central de categorização visual.")
                if not safe.empty and st.button("Confirmar leituras confiáveis", type="primary"):
                    create_snapshot("antes_confirmacao")
                    for row in safe.itertuples():
                        update_transaction(int(row.id), float(row.amount), row.direction, row.category, "Concluído", "Confirmado pela fila de revisão", True, row.subcategory, row.tags, row.owner, row.reimbursement_status)
                    st.success(f"{len(safe)} lançamento(s) confirmados e usados para ensinar o sistema.")
                    st.rerun()
    with st.expander("+ Novo lançamento manual"):
        with st.form("new_transaction"):
            a, b, c = st.columns(3)
            with a:
                launch_date = st.date_input("Data", value=date.today())
                description = st.text_input("Descrição")
            with b:
                amount_value = st.number_input("Valor", min_value=0.0, step=0.01, format="%.2f")
                direction = st.selectbox("Direção", ["Saída", "Entrada", "Indeterminada"])
            with c:
                category = st.text_input("Categoria", value="Não classificado")
                accounts = list_accounts()
                account_names = accounts["name"].tolist() if not accounts.empty else []
                payment_method = st.selectbox("Conta / método", ["Dinheiro", "Pix", "Não identificado"] + account_names)
            if st.form_submit_button("Adicionar lançamento", type="primary"):
                if not description.strip() or amount_value <= 0:
                    st.error("Informe descrição e valor maior que zero.")
                else:
                    add_transaction(launch_date.isoformat(), description, amount_value, direction, category, payment_method)
                    st.success("Lançamento adicionado.")
                    st.rerun()
    if data.empty:
        st.info("Nenhum lançamento disponível.")
        return
    q = st.text_input("Pesquisar por descrição ou estabelecimento", placeholder="Ex.: supermercado, Pix, Uber")
    filters = st.columns(4)
    direction_filter = filters[0].multiselect("Direção", ["Entrada", "Saída", "Indeterminada"], default=["Entrada", "Saída", "Indeterminada"])
    category_filter = filters[1].multiselect("Categoria", sorted(data["category"].dropna().unique().tolist()))
    status_filter = filters[2].multiselect("Status", ["Pendente", "Concluído"], default=["Pendente", "Concluído"])
    owner_filter = filters[3].multiselect("Responsável", sorted(data["owner"].dropna().unique().tolist()), default=sorted(data["owner"].dropna().unique().tolist()))
    filtered = data[data["direction"].isin(direction_filter)].copy()
    filtered = filtered[filtered["review_status"].isin(status_filter)]
    if owner_filter:
        filtered = filtered[filtered["owner"].isin(owner_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if q:
        needle = clean(q)
        filtered = filtered[filtered["description_clean"].str.contains(needle, na=False)]
    size = st.selectbox("Itens por página", [10, 25, 50, 100], index=1)
    pages = max(1, (len(filtered) + size - 1) // size)
    number = st.number_input("Página", min_value=1, max_value=pages, value=1, step=1)
    start = (number - 1) * size
    st.caption(f"{len(filtered)} lançamento(s) encontrados · página {number} de {pages}")
    columns = ["id", "date", "description", "amount", "direction", "category", "subcategory", "tags", "owner", "reimbursement_status", "payment_method", "transaction_kind", "review_status", "user_note"]
    edited = st.data_editor(filtered.iloc[start:start + size][[column for column in columns if column in filtered.columns]], use_container_width=True, hide_index=True, num_rows="fixed", column_config={"direction": st.column_config.SelectboxColumn("Direção", options=["Entrada", "Saída", "Indeterminada"]), "review_status": st.column_config.SelectboxColumn("Status", options=["Pendente", "Concluído"]), "amount": st.column_config.NumberColumn("Valor", min_value=0, format="R$ %.2f")})
    remember = st.checkbox("Aprender as categorias corrigidas nesta página", help="A próxima importação com descrição semelhante receberá a mesma categoria.")
    if st.button("Salvar alterações", type="primary"):
        for _, row in edited.iterrows():
            if pd.notna(row.get("id")):
                update_transaction(int(row["id"]), float(row["amount"]), row["direction"], row["category"], row.get("review_status", "Pendente"), row.get("user_note", ""), remember, row.get("subcategory", ""), row.get("tags", ""), row.get("owner", "Pessoal"), row.get("reimbursement_status", "Não"))
        st.success("Lançamentos atualizados no histórico.")
        st.rerun()


def render_category_studio(data: pd.DataFrame) -> None:
    st.subheader("Central de categorização visual")
    st.caption("Escolha um lançamento e atribua uma categoria em um clique. A regra pode ser lembrada para futuras importações.")
    if data.empty:
        return
    pending = data[(data["category"] == "Não classificado") | (data["review_status"] != "Concluído")].copy()
    candidates = pd.concat([pending, data]).drop_duplicates("id").head(40)
    choices = candidates["id"].tolist()
    selected_id = st.selectbox("Lançamento para revisar", choices, format_func=lambda item: f"#{item} · {candidates.loc[candidates.id == item, 'description'].iloc[0][:55]}")
    item = candidates.loc[candidates.id == selected_id].iloc[0]
    st.markdown(f"<div class='review-card'><strong>{item['merchant'] or item['description']}</strong><br><span class='small-muted'>{item['date'].strftime('%d/%m/%Y')} · {item['direction']} · {money_br(item['amount'])}</span><br><span class='small-muted'>Categoria atual: {item['category']}</span></div>", unsafe_allow_html=True)
    undo_col, confidence_col = st.columns([1.2, 4])
    with undo_col:
        if st.button("↶ Desfazer última alteração", key=f"undo_{selected_id}"):
            if restore_last_change(int(selected_id)):
                st.success("Última alteração restaurada.")
                st.rerun()
            st.caption("Não há uma alteração reversível para este lançamento.")
    with confidence_col:
        confidence = float(item.get("confidence", 0) or 0)
        st.progress(max(0, min(1, confidence)))
        st.caption(f"Confiança de leitura: {confidence:.0%}. Valores baixos merecem conferência antes de confirmar.")
    options = category_options(data)
    for start in range(0, len(options), 5):
        row = options[start:start + 5]
        columns = st.columns(5)
        for column, category in zip(columns, row):
            with column:
                icon = CATEGORY_ICONS.get(category, "•")
                if st.button(f"{icon} {category}", key=f"quick_category_{selected_id}_{category}", use_container_width=True):
                    update_transaction(int(selected_id), float(item["amount"]), item["direction"], category, "Concluído", "Categorizado pela central visual", True)
                    st.success(f"Categoria alterada para {category}. A preferência foi memorizada.")
                    st.rerun()
    with st.expander("Categoria personalizada ou correção em lote"):
        left, right = st.columns(2)
        custom = left.text_input("Nova categoria", placeholder="Ex.: Pets, Vestuário, Impostos")
        chosen = right.selectbox("Ou selecione uma categoria", options, key="category_studio_custom")
        details = st.columns(3)
        subcategory = details[0].text_input("Subcategoria", placeholder="Ex.: Mercado, Delivery")
        tags = details[1].text_input("Etiquetas", placeholder="Ex.: UFF, casa, viagem")
        owner = details[2].selectbox("Responsável", ["Pessoal", "Compartilhado", "Reembolsável"], index=["Pessoal", "Compartilhado", "Reembolsável"].index(item.get("owner", "Pessoal")) if item.get("owner", "Pessoal") in ["Pessoal", "Compartilhado", "Reembolsável"] else 0)
        reimbursement = st.selectbox("Situação de reembolso", ["Não", "A receber", "Recebido"], index=["Não", "A receber", "Recebido"].index(item.get("reimbursement_status", "Não")) if item.get("reimbursement_status", "Não") in ["Não", "A receber", "Recebido"] else 0)
        if st.button("Aplicar ao lançamento selecionado", key="apply_custom_category"):
            category = custom.strip() or chosen
            save_category(category, CATEGORY_ICONS.get(category, "•"))
            update_transaction(int(selected_id), float(item["amount"]), item["direction"], category, "Concluído", "Categoria ajustada visualmente", True, subcategory, tags, owner, reimbursement)
            st.rerun()
        st.divider()
        st.markdown("#### Corrigir vários estabelecimentos de uma só vez")
        st.caption("Os nomes são normalizados automaticamente: horário, ‘enviado para’ e outros ruídos do extrato não criam estabelecimentos diferentes.")
        full_history = prepare_data(load_transactions())
        merchant_values = sorted(value for value in full_history["merchant"].dropna().unique().tolist() if value and value != "Não identificado")
        default_merchants = [item["merchant"]] if item.get("merchant") in merchant_values else []
        selected_merchants = st.multiselect(
            "Estabelecimentos para corrigir",
            merchant_values,
            default=default_merchants,
            placeholder="Selecione um ou vários estabelecimentos",
        )
        bulk_category = st.selectbox("Categoria para todos os selecionados", options, key="bulk_category")
        subset = full_history[full_history["merchant"].isin(selected_merchants)].copy()
        if selected_merchants:
            preview = subset.groupby("merchant", as_index=False).agg(lançamentos=("id", "size"), total=("amount", "sum")).sort_values("total", ascending=False)
            st.caption(f"Prévia: {len(subset)} lançamento(s), em todo o histórico, serão atualizados.")
            st.dataframe(preview, use_container_width=True, hide_index=True, column_config={"total": st.column_config.NumberColumn("Total", format="R$ %.2f")})
        apply_bulk = st.checkbox("Confirmo a alteração de todos os lançamentos dos estabelecimentos selecionados.", disabled=not selected_merchants)
        if selected_merchants and apply_bulk and st.button("Aplicar correção em lote", type="primary"):
            create_snapshot("antes_correcao_lote_categorias")
            for row in subset.itertuples():
                update_transaction(int(row.id), float(row.amount), row.direction, bulk_category, "Concluído", "Categoria ajustada em lote", True, row.subcategory, row.tags, row.owner, row.reimbursement_status)
            st.success(f"{len(subset)} lançamento(s) de {len(selected_merchants)} estabelecimento(s) foram atualizados e a preferência foi memorizada.")
            st.rerun()
    with st.expander("Dividir este lançamento entre categorias"):
        st.caption("O lançamento original deixa de entrar nos totais e é substituído pelas partes criadas.")
        a, b, c, d = st.columns(4)
        first_category = a.selectbox("Primeira categoria", options, key=f"split_first_category_{selected_id}")
        first_value = b.number_input("Valor da primeira parte", min_value=0.0, max_value=float(item["amount"]), step=.01, key=f"split_first_value_{selected_id}")
        second_category = c.selectbox("Segunda categoria", options, key=f"split_second_category_{selected_id}")
        second_value = d.number_input("Valor da segunda parte", min_value=0.0, max_value=float(item["amount"]), step=.01, key=f"split_second_value_{selected_id}")
        if st.button("Dividir lançamento", key=f"split_action_{selected_id}", type="primary"):
            try:
                split_transaction(int(selected_id), [(first_category, first_value), (second_category, second_value)])
                st.success("Lançamento dividido e registrado no histórico.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with st.expander("Comprovante ou anexo"):
        attachment = st.file_uploader("Anexar comprovante, boleto ou recibo", type=["pdf", "png", "jpg", "jpeg"], key=f"attachment_{selected_id}")
        if attachment and st.button("Salvar anexo", key=f"save_attachment_{selected_id}"):
            save_attachment(int(selected_id), attachment.name, attachment.getvalue())
            st.success("Anexo salvo localmente.")
        attachments = list_attachments(int(selected_id))
        if not attachments.empty:
            st.dataframe(attachments, use_container_width=True, hide_index=True)


def render_categories(data: pd.DataFrame) -> None:
    st.markdown("# Categorias")
    expenses = data[data["direction"] == "Saída"]
    if expenses.empty:
        st.info("Importe lançamentos de saída para analisar categorias.")
        return
    summary = expenses.groupby("category", as_index=False).agg(valor=("amount", "sum"), ocorrencias=("amount", "size")).sort_values("valor", ascending=False)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Distribuição por categoria")
            chart_data = summary.rename(columns={"valor": "amount"})
            st.plotly_chart(category_chart(chart_data), use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            st.subheader("Estabelecimentos")
            merchants = expenses.groupby("merchant", as_index=False).agg(valor=("amount", "sum"), ocorrencias=("merchant", "size")).sort_values("valor", ascending=False).head(15)
            st.dataframe(merchants, use_container_width=True, hide_index=True)
    st.subheader("Qualidade da leitura")
    q1, q2, q3 = st.columns(3)
    q1.metric("Lançamentos", len(data))
    q2.metric("Baixa confiança", int((data["confidence"] < .55).sum()))
    q3.metric("Possíveis anomalias", int(data["anomaly"].sum()))
    anomalies = data.loc[flag_mask(data, "anomaly"), ["date", "description", "amount", "category"]]
    if anomalies.empty:
        st.caption("Não há anomalias relevantes no período selecionado.")
    else:
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
    st.divider()
    render_category_studio(data)
    with st.expander("Gerenciar categorias e memória de aprendizado"):
        a, b, c = st.columns(3)
        new_name = a.text_input("Nome da categoria", placeholder="Ex.: Pets")
        new_icon = b.text_input("Ícone", value="•", max_chars=4)
        new_color = c.color_picker("Cor", value="#2e6bff")
        if st.button("Criar ou atualizar categoria") and new_name.strip():
            save_category(new_name, new_icon, new_color)
            st.rerun()
        catalog = list_category_catalog()
        examples = list_category_examples()
        if not catalog.empty:
            st.dataframe(catalog, use_container_width=True, hide_index=True)
        if not examples.empty:
            st.caption("Exemplos aprendidos: quanto mais você confirma, maior a prioridade da sugestão.")
            st.dataframe(examples.head(25), use_container_width=True, hide_index=True)


def render_budgets(data: pd.DataFrame, selected_month: str) -> None:
    st.markdown("# Orçamentos")
    st.caption("Defina limites por categoria e acompanhe o consumo de cada período.")
    budget_month = selected_month if selected_month != "Todos os períodos" else "global"
    with st.form("budget_form"):
        c1, c2 = st.columns(2)
        category = c1.text_input("Categoria")
        limit_value = c2.number_input("Limite mensal", min_value=0.0, step=50.0, format="%.2f")
        if st.form_submit_button("Salvar orçamento", type="primary"):
            if category.strip() and limit_value > 0:
                upsert_budget(category, limit_value, budget_month)
                st.success("Orçamento salvo.")
                st.rerun()
    budgets = list_budgets(budget_month)
    expenses = data[data["direction"] == "Saída"].groupby("category")["amount"].sum().to_dict()
    if budgets.empty:
        st.info("Nenhum orçamento cadastrado. Comece adicionando uma categoria acima.")
        return
    rows = []
    for _, row in budgets.drop_duplicates("category").iterrows():
        used = float(expenses.get(row["category"], 0))
        limit = float(row["limit_amount"] or 0)
        rows.append({"Categoria": row["category"], "Limite": limit, "Utilizado": used, "Percentual": min(100, used / limit * 100) if limit else 0})
    st.dataframe(pd.DataFrame(rows).style.format({"Limite": money_br, "Utilizado": money_br, "Percentual": "{:.0f}%"}), use_container_width=True, hide_index=True)


def render_goals() -> None:
    st.markdown("# Metas")
    st.caption("Transforme seus objetivos em acompanhamento visível e persistente.")
    with st.form("goal_form"):
        a, b, c = st.columns(3)
        name = a.text_input("Nome da meta", placeholder="Reserva de emergência")
        target = b.number_input("Valor-alvo", min_value=0.0, step=100.0, format="%.2f")
        current = c.number_input("Valor já acumulado", min_value=0.0, step=100.0, format="%.2f")
        deadline = st.date_input("Prazo", value=date.today())
        if st.form_submit_button("Criar meta", type="primary"):
            if name.strip() and target > 0:
                save_goal(name, target, current, deadline.isoformat())
                st.success("Meta criada.")
                st.rerun()
    goals = list_goals()
    if goals.empty:
        st.info("Nenhuma meta cadastrada.")
        return
    st.subheader("Suas metas")
    edited = st.data_editor(goals[["id", "name", "target_amount", "current_amount", "deadline", "status"]], use_container_width=True, hide_index=True, column_config={"status": st.column_config.SelectboxColumn("Status", options=["Em andamento", "Concluída", "Pausada"]), "target_amount": st.column_config.NumberColumn("Alvo", format="R$ %.2f"), "current_amount": st.column_config.NumberColumn("Acumulado", format="R$ %.2f")})
    if st.button("Atualizar metas", type="primary"):
        for _, row in edited.iterrows():
            update_goal(int(row["id"]), float(row["current_amount"]), row["status"])
        st.success("Metas atualizadas.")
        st.rerun()


def render_accounts(data: pd.DataFrame) -> None:
    st.markdown("# Contas e cartões")
    st.caption("Centralize saldos, cartões, faturas e patrimônio líquido em uma única tela.")
    with st.expander("+ Cadastrar ou atualizar conta", expanded=False):
        with st.form("account_form"):
            a, b, c = st.columns(3)
            name = a.text_input("Nome", placeholder="Nubank principal")
            account_type = b.selectbox("Tipo", ["Conta corrente", "Carteira", "Cartão de crédito", "Investimento", "Dívida"])
            institution = c.text_input("Instituição", placeholder="Nubank, Inter, Itaú...")
            d, e, f, g, h = st.columns(5)
            balance = d.number_input("Saldo / patrimônio", value=0.0, step=50.0)
            limit = e.number_input("Limite do cartão", value=0.0, step=100.0)
            closing = f.number_input("Fechamento", min_value=1, max_value=31, value=10)
            due = g.number_input("Vencimento", min_value=1, max_value=31, value=17)
            interest = h.number_input("Juros ao mês (%)", min_value=0.0, max_value=30.0, value=0.0, step=.1)
            if st.form_submit_button("Salvar conta", type="primary") and name.strip():
                save_account(name, account_type, institution, balance, limit, closing if account_type == "Cartão de crédito" else None, due if account_type == "Cartão de crédito" else None, interest if account_type == "Dívida" else 0)
                st.success("Conta salva.")
                st.rerun()
    accounts = list_accounts()
    if accounts.empty:
        st.info("Cadastre suas contas, cartões, investimentos ou dívidas para acompanhar o patrimônio.")
        return
    assets = float(accounts.loc[accounts["account_type"].isin(["Conta corrente", "Carteira", "Investimento"]), "current_balance"].sum())
    debts = abs(float(accounts.loc[accounts["account_type"] == "Dívida", "current_balance"].sum()))
    cards = accounts[accounts["account_type"] == "Cartão de crédito"]
    card_names = cards["name"].tolist()
    card_spend = float(data.loc[(data["direction"] == "Saída") & (data["payment_method"].isin(card_names) | data["payment_method"].str.contains("cart", case=False, na=False)), "amount"].sum())
    a, b, c = st.columns(3)
    a.metric("Patrimônio líquido informado", money_br(assets - debts))
    b.metric("Dívidas informadas", money_br(debts))
    c.metric("Fatura identificada no período", money_br(card_spend), help="Soma de saídas cuja forma de pagamento contém 'cartão'.")
    for row in accounts.itertuples():
        left, center, right = st.columns([2.7, 2.2, 1])
        left.markdown(f"<div class='category-chip'><strong>{'💳' if row.account_type == 'Cartão de crédito' else '🏦'} {row.name}</strong><span>{row.institution or row.account_type}</span></div>", unsafe_allow_html=True)
        if row.account_type == "Cartão de crédito":
            statement = float(data.loc[(data["direction"] == "Saída") & ((data["payment_method"] == row.name) | ((data["payment_method"].str.contains("cart", case=False, na=False)) & len(card_names) == 1)), "amount"].sum())
            available = max(0.0, float(row.credit_limit) - statement)
            center.metric("Fatura / limite", f"{money_br(statement)} · disponível {money_br(available)}", help=f"Fechamento dia {row.closing_day or '-'} · vencimento dia {row.due_day or '-'}")
        else:
            center.metric("Saldo informado", money_br(row.current_balance))
        if right.button("Desativar", key=f"remove_account_{row.id}"):
            deactivate_account(row.id)
            st.rerun()
    debts = accounts[accounts["account_type"] == "Dívida"].copy()
    if not debts.empty:
        st.subheader("Plano de quitação de dívidas")
        budget_for_debt = st.number_input("Valor mensal disponível para quitar dívidas", min_value=0.0, step=50.0, format="%.2f")
        if budget_for_debt > 0:
            snowball = debts.sort_values("current_balance")[['name', 'current_balance', 'interest_rate']].copy()
            avalanche = debts.sort_values("interest_rate", ascending=False)[['name', 'current_balance', 'interest_rate']].copy()
            a, b = st.columns(2)
            a.markdown("**Bola de neve — menor saldo primeiro**")
            a.dataframe(snowball, use_container_width=True, hide_index=True)
            b.markdown("**Avalanche — maior juro primeiro**")
            b.dataframe(avalanche, use_container_width=True, hide_index=True)
            st.caption("As ordens são educativas: escolha a estratégia e confirme condições reais, juros e vencimentos com o credor.")


def render_reports(data: pd.DataFrame) -> None:
    st.markdown("# Relatórios")
    st.caption("Exporte seu histórico para continuar a análise em Excel ou compartilhar um resumo.")
    export_data = data.drop(columns=["description_clean", "anomaly"], errors="ignore")
    st.download_button("Baixar histórico em CSV", export_data.to_csv(index=False).encode("utf-8-sig"), "finp_mat_historico.csv", "text/csv")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_data.to_excel(writer, index=False, sheet_name="Histórico")
    st.download_button("Baixar histórico em Excel", buffer.getvalue(), "finp_mat_historico.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if not data.empty:
        st.subheader("Resumo por categoria")
        st.dataframe(data.groupby(["direction", "category"], as_index=False)["amount"].sum(), use_container_width=True, hide_index=True)
        st.subheader("Hábitos de consumo")
        expenses = data[data["direction"] == "Saída"].copy()
        if not expenses.empty:
            expenses["dia_semana"] = expenses["date"].dt.day_name().map({"Monday": "Seg", "Tuesday": "Ter", "Wednesday": "Qua", "Thursday": "Qui", "Friday": "Sex", "Saturday": "Sáb", "Sunday": "Dom"})
            habit = expenses.groupby("dia_semana", as_index=False)["amount"].sum()
            order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
            habit["dia_semana"] = pd.Categorical(habit["dia_semana"], order, ordered=True)
            figure = go.Figure(go.Bar(x=habit.sort_values("dia_semana")["dia_semana"], y=habit.sort_values("dia_semana")["amount"], marker_color="#2e6bff", hovertemplate="%{x}<br>R$ %{y:,.2f}<extra></extra>"))
            st.plotly_chart(style_chart(figure, 260), use_container_width=True, config={"displayModeBar": False})
        if data["date"].dt.year.nunique() >= 2:
            st.subheader("Comparação anual")
            annual = data[data["direction"] == "Saída"].assign(ano=data["date"].dt.year, mes=data["date"].dt.month).groupby(["ano", "mes"], as_index=False)["amount"].sum()
            st.dataframe(annual.pivot(index="mes", columns="ano", values="amount").fillna(0).style.format(money_br), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Previsão para o próximo mês")
            forecast = forecast_categories(data)
            if forecast.empty:
                st.caption("Ainda não há dados suficientes para projeção.")
            else:
                st.dataframe(forecast, use_container_width=True, hide_index=True, column_config={"forecast": st.column_config.NumberColumn("Média projetada", format="R$ %.2f"), "months": "Meses observados"})
        with right:
            st.subheader("Assinaturas com possível aumento")
            changes = subscription_changes(data)
            if changes.empty:
                st.caption("Nenhuma elevação recorrente relevante identificada.")
            else:
                st.dataframe(changes, use_container_width=True, hide_index=True, column_config={"previous": st.column_config.NumberColumn("Anterior", format="R$ %.2f"), "current": st.column_config.NumberColumn("Atual", format="R$ %.2f"), "change": st.column_config.NumberColumn("Aumento", format="R$ %.2f")})
        st.subheader("Decisões com maior impacto")
        priorities = decision_priorities(data)
        if not priorities.empty:
            st.dataframe(priorities, use_container_width=True, hide_index=True, column_config={"impacto_estimado": st.column_config.NumberColumn("Economia mensal estimada", format="R$ %.2f")})
        if not expenses.empty:
            st.subheader("Mapa do fluxo de dinheiro")
            flow = expenses.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(8)
            nodes = ["Saídas"] + flow["category"].tolist()
            figure = go.Figure(go.Sankey(
                node=dict(label=nodes, pad=18, thickness=20, color=["#2e6bff"] + ["#a7befc"] * len(flow)),
                link=dict(source=[0] * len(flow), target=list(range(1, len(flow) + 1)), value=flow["amount"].tolist(), color=["rgba(46,107,255,.28)"] * len(flow)),
            ))
            figure.update_layout(height=300, margin=dict(l=10, r=10, t=8, b=8), paper_bgcolor="#ffffff", font=dict(color="#40506b"))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        st.subheader("Organização para declaração")
        tax_data = data[[column for column in ["date", "description", "amount", "direction", "category", "subcategory", "tags", "reimbursement_status", "source_file"] if column in data.columns]].copy()
        st.caption("Este arquivo organiza comprovantes e lançamentos; a classificação tributária e a declaração devem ser confirmadas com fonte oficial ou profissional habilitado.")
        st.download_button("Baixar dados para conferência de IR", tax_data.to_csv(index=False).encode("utf-8-sig"), "finp_mat_dados_ir.csv", "text/csv")


def render_planning(data: pd.DataFrame) -> None:
    st.markdown("# Planejamento")
    st.caption("Antecipe compromissos recorrentes, parcelas e pontos que exigem decisão.")
    recurring = data[(data["direction"] == "Saída") & flag_mask(data, "recurring")]
    installments = data[flag_mask(data, "installment_hint")]
    review_needed = data[(data["direction"] == "Indeterminada") | (data["review_status"] != "Concluído")]
    expected_fixed = recurring["amount"].sum()
    a, b, c = st.columns(3)
    a.metric("Recorrências", len(recurring), help="Lançamentos semelhantes encontrados em mais de um mês.")
    b.metric("Parcelas prováveis", len(installments), help="Descrições que indicam parcela ou fração do total.")
    c.metric("Próximo mês estimado", money_br(expected_fixed), help="Soma simples das saídas recorrentes detectadas.")
    left, right = st.columns([1.15, .85])
    with left:
        with st.container(border=True):
            st.subheader("Assinaturas e recorrências")
            if recurring.empty:
                st.caption("Nenhuma recorrência confirmada ainda. Mais meses importados melhoram esta leitura.")
            else:
                subscriptions = recurring.groupby(["merchant", "category"], as_index=False).agg(
                    valor_mensal=("amount", "median"),
                    ocorrencias=("amount", "size"),
                    ultima_data=("date", "max"),
                ).sort_values("valor_mensal", ascending=False)
                st.dataframe(subscriptions, use_container_width=True, hide_index=True, column_config={"valor_mensal": st.column_config.NumberColumn("Estimativa mensal", format="R$ %.2f")})
    with right:
        with st.container(border=True):
            st.subheader("Saúde dos dados")
            coverage = 100 - (len(review_needed) / len(data) * 100) if len(data) else 100
            st.metric("Confiabilidade", f"{coverage:.0f}%")
            st.progress(max(0, min(1, coverage / 100)))
            st.caption(f"{len(review_needed)} lançamento(s) ainda precisam de revisão.")
            if not installments.empty:
                st.markdown("**Parcelas encontradas**")
                st.caption("Revise se os valores representam compromissos ainda ativos.")
    st.subheader("Sugestões priorizadas")
    for item in recommendations(data):
        st.info(f"**{item['title']}** — {item['text']}")
    if not data.empty and not review_needed.empty:
        st.warning("Há lançamentos sem direção confirmada. Revise-os em Lançamentos para evitar distorções no saldo e nos gráficos.")
    st.subheader("Calendário de contas fixas")
    with st.expander("+ Adicionar conta ou vencimento"):
        with st.form("bill_form"):
            a, b, c, d = st.columns(4)
            name = a.text_input("Conta")
            category = b.text_input("Categoria", value="Moradia")
            bill_amount = c.number_input("Valor", min_value=0.0, step=10.0)
            due_day = d.number_input("Dia", min_value=1, max_value=31, value=10)
            method = st.selectbox("Forma de pagamento", ["Conta", "Cartão", "Boleto", "Pix"])
            if st.form_submit_button("Salvar vencimento", type="primary") and name.strip() and bill_amount > 0:
                save_bill(name, category, bill_amount, due_day, method)
                st.success("Conta adicionada ao calendário.")
                st.rerun()
    bills = list_bills()
    if bills.empty:
        st.caption("Cadastre contas fixas para visualizar o fluxo previsto do próximo mês.")
    else:
        planned = float(bills["amount"].sum())
        income_base = float(data.loc[data["direction"] == "Entrada", "amount"].sum())
        st.metric("Saídas fixas previstas", money_br(planned), help="Soma das contas ativas cadastradas.")
        st.dataframe(bills[["id", "due_day", "name", "category", "amount", "payment_method"]], use_container_width=True, hide_index=True, column_config={"amount": st.column_config.NumberColumn("Valor", format="R$ %.2f")})
        to_remove = st.selectbox("Desativar conta", ["Nenhuma"] + [f"{row.id} · dia {row.due_day} · {row.name}" for row in bills.itertuples()])
        if st.button("Desativar selecionada") and to_remove != "Nenhuma":
            deactivate_bill(int(to_remove.split(" · ")[0]))
            st.rerun()
    st.subheader("Simulador de decisão")
    sim_a, sim_b, sim_c = st.columns(3)
    monthly_cut = sim_a.number_input("Quanto pretende reduzir por mês?", min_value=0.0, step=25.0, format="%.2f")
    horizon = sim_b.selectbox("Horizonte", [6, 12, 24, 36], index=1)
    return_rate = sim_c.number_input("Rendimento mensal estimado (%)", min_value=0.0, max_value=5.0, value=0.0, step=.1)
    if monthly_cut > 0:
        rate = return_rate / 100
        future = monthly_cut * horizon if rate == 0 else monthly_cut * (((1 + rate) ** horizon - 1) / rate)
        st.info(f"Reduzindo {money_br(monthly_cut)} por {horizon} meses, você pode acumular aproximadamente **{money_br(future)}**. É uma simulação, não uma promessa de rentabilidade.")
    if not bills.empty:
        st.subheader("Linha do tempo e projeção diária")
        current_balance = float(data.loc[data["direction"] == "Entrada", "amount"].sum() - data.loc[data["direction"] == "Saída", "amount"].sum())
        year, month = date.today().year, date.today().month
        timeline = []
        for row in bills.itertuples():
            due = date(year, month, min(int(row.due_day), calendar.monthrange(year, month)[1]))
            if due < date.today():
                next_month = month % 12 + 1
                next_year = year + (month == 12)
                due = date(next_year, next_month, min(int(row.due_day), calendar.monthrange(next_year, next_month)[1]))
            timeline.append({"data": due, "conta": row.name, "categoria": row.category, "valor": float(row.amount)})
        schedule = pd.DataFrame(timeline).sort_values("data")
        schedule["saldo_projetado"] = current_balance - schedule["valor"].cumsum()
        st.dataframe(schedule, use_container_width=True, hide_index=True, column_config={"data": st.column_config.DateColumn("Vencimento", format="DD/MM/YYYY"), "valor": st.column_config.NumberColumn("Valor", format="R$ %.2f"), "saldo_projetado": st.column_config.NumberColumn("Saldo após conta", format="R$ %.2f")})
        if (schedule["saldo_projetado"] < 0).any():
            st.warning("A projeção indica saldo negativo após alguns vencimentos. Considere ajustar o calendário ou reservar valores antes dessas datas.")
        ics = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//FINP MAT//PT-BR\n" + "".join(f"BEGIN:VEVENT\nDTSTART;VALUE=DATE:{row.data.strftime('%Y%m%d')}\nSUMMARY:FINP MAT · {row.conta} · R$ {row.valor:.2f}\nEND:VEVENT\n" for row in schedule.itertuples()) + "END:VCALENDAR\n"
        st.download_button("Exportar vencimentos para calendário (.ics)", ics.encode("utf-8"), "finp_mat_vencimentos.ics", "text/calendar")


def render_audit(data: pd.DataFrame) -> None:
    st.markdown("# Auditoria da leitura")
    st.caption("Confira o que foi extraído, qual regra definiu entrada/saída e se o extrato fecha com os saldos informados.")
    if data.empty:
        st.info("Importe um extrato para iniciar a auditoria.")
        return
    files = list_uploads()
    source_options = {f"{row.filename} · {row.transaction_month}": row.source_hash for row in files.itertuples()}
    selected = st.selectbox("Arquivo para conferir", list(source_options) or ["Sem arquivo"])
    source_hash = source_options.get(selected, "")
    source_data = data[data.get("source_hash", "") == source_hash].copy() if source_hash else data.copy()
    st.subheader("Evidências de extração")
    audit_columns = ["id", "date", "description", "amount", "direction", "direction_confidence", "transaction_kind", "direction_reason", "raw_values", "source_line", "review_status"]
    st.dataframe(source_data[[col for col in audit_columns if col in source_data.columns]], use_container_width=True, hide_index=True, column_config={"amount": st.column_config.NumberColumn("Valor escolhido", format="R$ %.2f"), "direction_confidence": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1, format="%.0f%%")})
    st.subheader("Reconciliação de saldo")
    existing = list_reconciliations()
    prior = existing[existing["source_hash"] == source_hash] if source_hash else pd.DataFrame()
    opening = float(prior.iloc[0]["opening_balance"]) if not prior.empty else 0.0
    closing = float(prior.iloc[0]["closing_balance"]) if not prior.empty else 0.0
    a, b, c = st.columns(3)
    opening = a.number_input("Saldo inicial informado", value=opening, format="%.2f")
    closing = b.number_input("Saldo final informado", value=closing, format="%.2f")
    incomes = float(source_data.loc[source_data["direction"] == "Entrada", "amount"].sum())
    expenses = float(source_data.loc[source_data["direction"] == "Saída", "amount"].sum())
    calculated = opening + incomes - expenses
    difference = closing - calculated
    c.metric("Diferença", money_br(difference), delta="Fechado" if abs(difference) < .02 else "Revisar")
    st.caption(f"Cálculo: {money_br(opening)} + {money_br(incomes)} − {money_br(expenses)} = {money_br(calculated)}")
    note = st.text_input("Observação da conferência")
    if st.button("Salvar reconciliação", type="primary") and source_hash:
        save_reconciliation(source_hash, opening, closing, note)
        st.success("Conferência salva.")
    st.subheader("Histórico de alterações")
    chosen = st.selectbox("Lançamento para histórico", source_data["id"].tolist(), format_func=lambda item: f"#{item} · {source_data.loc[source_data.id == item, 'description'].iloc[0]}")
    log = list_audit_log(int(chosen))
    if log.empty:
        st.caption("Nenhuma alteração registrada ainda.")
    else:
        st.dataframe(log[["changed_at", "action", "note", "before_json", "after_json"]], use_container_width=True, hide_index=True)


def render_agent(data: pd.DataFrame) -> None:
    st.markdown("# MATHEWZINHO")
    st.caption("Agente local de IA com os seus e-books e o histórico financeiro consolidado.")
    indexed = index_pdfs()
    available, model = ollama_status()
    a, b = st.columns(2)
    a.metric("Trechos indexados", len(indexed))
    b.metric("Modelo local", model)
    if available:
        st.success("Ollama conectado. O MATHEWZINHO está usando geração local.")
    else:
        st.warning("Ollama não está disponível. A busca nos e-books continua funcionando, mas a resposta gerada por IA será ativada quando o serviço local estiver instalado e iniciado.")
    week = weekly_summary(data)
    st.info(f"**Resumo semanal:** {money_br(week['current'])} em saídas nos últimos 7 dias. Variação de {money_br(abs(week['change']))} em relação à semana anterior.")
    with st.expander("Ensinar o MATHEWZINHO a corrigir categorias", expanded=True):
        st.caption("Escreva uma regra como: “tudo que tiver LMS House é Moradia”, “Uber e 99 vão para Transporte” ou “iFood, Rappi e restaurante são Alimentação”.")
        instruction = st.text_area("Instrução de categoria", key="agent_category_instruction", placeholder="Ex.: Tudo que tiver LMS House é Moradia")
        full_history = prepare_data(load_transactions())
        parsed = parse_category_instruction(instruction, category_options(full_history))
        if instruction.strip() and not parsed:
            st.warning("Não consegui identificar categoria e palavras-chave. Use o formato “palavra-chave é Categoria” ou escolha uma categoria existente.")
        if parsed:
            matched = sorted(set(item for keyword in parsed["keywords"] for item in matching_transaction_ids(keyword)))
            st.success(f"Entendi: **{', '.join(parsed['keywords'])} → {parsed['category']}**. {len(matched)} lançamento(s) existente(s) serão atualizados e as próximas importações seguirão essa regra.")
            confirm_instruction = st.checkbox("Confirmo a criação e aplicação desta instrução", key="confirm_agent_instruction")
            if confirm_instruction and st.button("Aplicar instrução do MATHEWZINHO", type="primary", key="apply_agent_instruction"):
                create_snapshot("antes_instrucao_agente")
                for keyword in parsed["keywords"]:
                    save_keyword_instruction(keyword, parsed["category"])
                for transaction_id in matched:
                    row = full_history.loc[full_history["id"] == transaction_id].iloc[0]
                    update_transaction(int(transaction_id), float(row.amount), row.direction, parsed["category"], "Concluído", f"Regra do MATHEWZINHO: {parsed['instruction']}", False, row.subcategory, row.tags, row.owner, row.reimbursement_status)
                st.success(f"Instrução aplicada a {len(matched)} lançamento(s) e salva para o futuro.")
                st.rerun()
        rules = list_keyword_instructions()
        if not rules.empty:
            st.caption("Instruções ativas")
            st.dataframe(rules[["pattern", "category", "updated_at"]], use_container_width=True, hide_index=True)
    with st.expander("Plano de organização em 30 dias"):
        priorities = decision_priorities(data)
        st.markdown("1. Revise os lançamentos pendentes e corrija as categorias na Central visual.\n2. Cadastre contas fixas, cartões e limites.\n3. Defina um orçamento para as categorias de maior peso.\n4. Escolha uma meta e programe uma reserva mensal.")
        if not priorities.empty:
            st.markdown("**Ações de maior impacto no seu histórico:**")
            for row in priorities.itertuples():
                st.write(f"• {row.ação}: potencial aproximado de {money_br(row.impacto_estimado)} por mês.")
    question = st.text_area("Como posso ajudar?", placeholder="Por que meus gastos aumentaram? Como organizar uma reserva de emergência?", height=90)
    if st.button("Perguntar ao MATHEWZINHO", type="primary") and question.strip():
        with st.spinner("MATHEWZINHO analisando seus dados..."):
            response = ask_mathewzinho(question, transactions=data)
        st.caption(f"Modo de resposta: {response.get('provider', 'local')}")
        if response.get("tools"):
            st.caption("Cálculos usados: " + ", ".join(response["tools"]))
        st.markdown(response["answer"])
        if response["sources"]:
            st.caption("Fontes recuperadas")
            for source in response["sources"]:
                st.write(f"• {source['file']} — página {source['page']} (relevância {source['score']})")


def render_settings() -> None:
    st.markdown("# Configurações")
    st.caption("Ajustes locais do FINP MAT e do MATHEWZINHO.")
    st.subheader("Aparência")
    theme = st.radio("Tema", ["Claro", "Escuro"], horizontal=True, index=1 if st.session_state.get("finp_theme") == "Escuro" else 0)
    if theme != st.session_state.get("finp_theme", "Claro"):
        st.session_state.finp_theme = theme
        st.rerun()
    st.subheader("Conta")
    st.write(f"Usuário ativo: **{account_username()}**")
    if st.button("Sair da conta"):
        st.session_state.authenticated = False
        st.rerun()
    st.subheader("Ollama")
    available, model = ollama_status()
    st.write(f"Modelo configurado: `{model}`")
    st.write("Status: " + ("conectado" if available else "não conectado"))
    st.caption("Para trocar o modelo no Windows: $env:OLLAMA_MODEL = \"nome-do-modelo\"")
    st.subheader("Privacidade")
    st.info("Os lançamentos, e-books e senha ficam na pasta local do projeto. O MATHEWZINHO usa o Ollama local quando ele está disponível.")
    st.subheader("Backup protegido")
    password = st.text_input("Senha do backup", type="password", help="Use uma senha nova com pelo menos 8 caracteres. Ela será necessária para restaurar o arquivo.")
    if st.button("Preparar backup criptografado"):
        try:
            st.session_state.finp_backup = create_backup(password)
            st.success("Backup preparado. Baixe-o abaixo e guarde a senha em local seguro.")
        except ValueError as exc:
            st.error(str(exc))
    if st.session_state.get("finp_backup"):
        st.download_button("Baixar backup FINP MAT", st.session_state.finp_backup, "finp_mat_backup.enc", "application/octet-stream")
    restore_file = st.file_uploader("Restaurar backup protegido", type=["enc"], key="restore_backup")
    confirm = st.checkbox("Entendo que restaurar substituirá os dados locais atuais.")
    if restore_file and confirm and st.button("Restaurar agora", type="primary"):
        try:
            restore_backup(restore_file.getvalue(), password)
            st.success("Backup preparado com segurança. Agora feche o FINP MAT (Ctrl+C no terminal) e abra novamente pelo arquivo abrir_finpmat.bat. A restauração será aplicada antes de o banco ser aberto.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    st.subheader("Modo demonstração")
    if st.button("Carregar dados de demonstração"):
        total = load_demo_data()
        st.success(f"{total} lançamentos fictícios foram adicionados. Eles aparecem como 'demonstração'.")
        st.rerun()
    st.subheader("Categorias aprendidas")
    rules = list_category_rules()
    if rules.empty:
        st.caption("Quando você salvar uma correção com aprendizado ativado, a regra aparecerá aqui.")
    else:
        st.dataframe(rules[["pattern", "category", "updated_at"]], use_container_width=True, hide_index=True)


try:
    restore_applied_at_start = apply_pending_restore()
    restore_start_error = ""
except RuntimeError as exc:
    restore_applied_at_start = False
    restore_start_error = str(exc)

init_storage()
login_gate()
if restore_start_error:
    st.error(restore_start_error)
if restore_applied_at_start:
    st.success("Backup restaurado com sucesso. Seus dados anteriores estão disponíveis novamente.")
st.session_state.setdefault("page", "Visão Geral")
history = load_transactions()
all_data = prepare_data(history)
available_months = sorted(all_data["date"].dt.strftime("%Y-%m").dropna().unique().tolist(), reverse=True) if not all_data.empty else []
main_navigation = [("Visão Geral", "⌂"), ("Extratos", "⇩"), ("Lançamentos", "↕"), ("Planejamento", "⌁"), ("MATHEWZINHO", "✦")]
secondary_navigation = ["Categorias", "Contas", "Orçamentos", "Metas", "Relatórios", "Auditoria", "Configurações"]
with st.container(border=True):
    header = st.columns([1.55, 1.02, 1.02, 1.15, 1.18, 1.34, 1.38, 1.45])
    brand_col = header[0]
    with brand_col:
        st.markdown(logo_html(), unsafe_allow_html=True)
    for column, (label, icon) in zip(header[1:6], main_navigation):
        with column:
            if st.button(f"{icon} {label}", key=f"nav_{label}", type="primary" if st.session_state.page == label else "secondary", use_container_width=True):
                st.session_state.page = label
                st.rerun()
    with header[6]:
        more = st.selectbox("Mais", ["Mais"] + secondary_navigation, key="nav_more", label_visibility="collapsed")
        if more != "Mais" and st.session_state.page != more:
            st.session_state.page = more
            st.rerun()
    with header[7]:
        if st.button("＋ Lançamento", key="header_new", type="primary", use_container_width=True):
            st.session_state.page = "Lançamentos"
            st.rerun()
    controls = st.columns([2.1, 7.9])
    with controls[0]:
        selected_month = st.selectbox("Período", ["Todos os períodos"] + available_months, key="period_selector")
    with controls[1]:
        include_pending = st.toggle("Incluir lançamentos pendentes nos gráficos", value=False, help="Desligado por padrão para evitar que leituras ainda não revisadas distorçam seus totais.")
page = st.session_state.page
if page == "Extratos":
    process_imports()
period_data = all_data if selected_month == "Todos os períodos" else all_data[all_data["date"].dt.strftime("%Y-%m") == selected_month].copy()
analysis_data = period_data if include_pending else period_data[period_data["review_status"] == "Concluído"].copy()
goals = list_goals()

page = st.session_state.page
top_left, = st.columns([1])
with top_left:
    st.markdown("<div class='topbar'><span class='small-muted'>FINANÇAS PESSOAIS · VISÃO CONSOLIDADA</span></div>", unsafe_allow_html=True)

if page == "Visão Geral":
    render_overview(all_data if include_pending else all_data[all_data["review_status"] == "Concluído"], analysis_data, selected_month, goals)
elif page == "Extratos":
    render_imports()
elif page == "Lançamentos":
    render_transactions(period_data)
elif page == "Categorias":
    # A central de correção deve mostrar também lançamentos pendentes, para que possam ser revisados.
    render_categories(period_data)
elif page == "Orçamentos":
    render_budgets(analysis_data, selected_month)
elif page == "Metas":
    render_goals()
elif page == "Contas":
    render_accounts(analysis_data)
elif page == "Relatórios":
    render_reports(analysis_data)
elif page == "Planejamento":
    render_planning(analysis_data)
elif page == "Auditoria":
    render_audit(period_data)
elif page == "MATHEWZINHO":
    render_agent(analysis_data)
elif page == "Configurações":
    render_settings()

st.caption("FINP MAT · MATHEWZINHO · organização financeira local e explicável")
