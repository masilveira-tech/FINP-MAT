from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


ALIASES = {
    "date": ["data", "date", "dt", "data lançamento", "data transação"],
    "description": ["descrição", "descricao", "histórico", "historico", "lançamento", "lancamento", "memo"],
    "amount": ["valor", "amount", "quantia", "valor transação"],
    "payment_method": ["tipo", "type", "método", "metodo", "forma de pagamento"],
}

RULES = {
    "Alimentação": ["ifood", "rappi", "restaurante", "lanch", "pizza", "mercado", "supermercado", "padaria", "carrefour", "assai", "atacadao"],
    "Moradia": ["aluguel", "condominio", "energia", "enel", "luz", "gas", "agua", "internet", "claro", "vivo", "tim"],
    "Transporte": ["uber", "99", "combust", "posto", "shell", "ipva", "estacion", "pedagio", "onibus", "metro"],
    "Saúde": ["farmacia", "drogaria", "hospital", "consulta", "clinica", "unimed"],
    "Educação": ["curso", "faculdade", "universidade", "livro", "udemy", "escola", "mensalidade"],
    "Lazer": ["netflix", "spotify", "cinema", "teatro", "steam", "academia", "jogo"],
    "Financeiro": ["tarifa", "anuidade", "juros", "iof", "multa", "corretora", "ted", "doc"],
    "Transferências": ["pix", "transferencia", "transferência"],
}


def clean(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", value).strip()


def merchant_name(description: object) -> str:
    """Remove ruído bancário/OCR para agrupar corretamente o mesmo estabelecimento."""
    raw = str(description or "")
    raw = re.sub(r"(?i)^\s*(?:as|às)?\s*\d{1,2}:\d{2}(?::\d{2})?\s*", "", raw)
    raw = re.sub(r"(?i)\b(?:pix\s+)?(?:enviado|envio|transferencia\s+enviada|pagamento)\s+(?:para|p/)?\s*", "", raw)
    raw = re.sub(r"(?i)\b(?:pix\s+)?(?:recebido|recebimento|transferencia\s+recebida)\s+(?:de|do|da)?\s*", "", raw)
    raw = re.sub(r"(?i)\b(?:compra|debito|credito|cartao|r\$)\b", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" -:|;")
    return raw.title() if raw else "Não identificado"


def flag_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """Retorna uma máscara booleana segura para campos 0/1, verdadeiro/falso ou vazios.

    O SQLite pode devolver indicadores como inteiros. Usar uma série de 0 e 1
    diretamente dentro de ``frame[...]`` faz o pandas interpretá-la como nomes
    de colunas, e não como filtro de linhas.
    """
    values = frame.get(column)
    if not isinstance(values, pd.Series):
        return pd.Series(False, index=frame.index, dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.lower()
    return numeric.fillna(0).ne(0) | text.isin({"true", "sim", "yes", "y", "x"})


def find_columns(frame: pd.DataFrame) -> dict[str, str]:
    columns = {clean(c): c for c in frame.columns}
    result = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if clean(alias) in columns:
                result[target] = columns[clean(alias)]
                break
    return result


def amount(value: object) -> float:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).replace("R$", "").replace(" ", "").strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return 0.0
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def infer_direction_details(description: object, raw_value: object = 0, payment_method: object = "") -> tuple[str, float, str, str]:
    """Retorna direção, confiança, tipo e evidência. Prioriza sinais explícitos do banco."""
    text = clean(f"{description} {payment_method}")
    raw_text = clean(raw_value)
    signed = amount(raw_value)
    try:
        if "-" in str(raw_value):
            signed = -abs(signed)
    except TypeError:
        pass
    pix = "pix" in text
    incoming = ("recebido", "recebimento", "credito", "creditado", "deposito", "depositado", "salario", "transferencia recebida", "ted recebida", "doc recebida", "estorno", "resgate", "reembolso", "cashback", "dividendo", "recebldo", "devolucao recebida")
    outgoing = ("enviado", "envio", "debito", "debitado", "saque", "compra", "pagamento", "tarifa", "boleto pago", "transferencia enviada", "ted enviada", "doc enviada", "parcela", "fatura", "retirada", "envlado")
    scheduled = ("agendado", "agendada", "programado", "a agendar")
    returned = ("devolvido", "devolucao", "estorno", "cancelado", "reversao", "revertido")
    if any(word in text for word in scheduled):
        direction = "Saída" if any(word in text for word in outgoing) else "Indeterminada"
        return direction, 0.82, "Pix agendado" if pix else "Pagamento agendado", "texto indica agendamento"
    if any(word in text for word in returned):
        return "Entrada", 0.95, "Pix devolvido" if pix else "Estorno ou devolução", "texto indica devolução/estorno"
    has_in, has_out = any(word in text for word in incoming), any(word in text for word in outgoing)
    if has_in and not has_out:
        return "Entrada", 0.98, "Pix recebido" if pix else "Recebimento", "texto indica crédito/recebimento"
    if has_out and not has_in:
        return "Saída", 0.98, "Pix enviado" if pix else "Pagamento", "texto indica débito/pagamento"
    if signed < 0:
        return "Saída", 0.95, "Transação com sinal negativo", "sinal do valor"
    if re.search(r"(?:^|\s)d(?:\s|$)|\bdeb\b", raw_text):
        return "Saída", 0.78, "Débito", "marcador D/débito"
    if re.search(r"(?:^|\s)c(?:\s|$)|\bcred\b", raw_text):
        return "Entrada", 0.78, "Crédito", "marcador C/crédito"
    return "Indeterminada", 0.25, "Pix não identificado" if pix else "Não identificado", "nenhuma evidência confiável"


def infer_direction(description: object, raw_value: object = 0, payment_method: object = "") -> tuple[str, float]:
    direction, confidence, _, _ = infer_direction_details(description, raw_value, payment_method)
    return direction, confidence


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    cols = find_columns(frame)
    missing = [key for key in ("date", "description", "amount") if key not in cols]
    if missing:
        raise ValueError("Não encontrei as colunas de " + ", ".join(missing) + ". Revise o arquivo ou corrija a prévia.")
    out = pd.DataFrame({
        "date": pd.to_datetime(frame[cols["date"]], dayfirst=True, errors="coerce"),
        "description": frame[cols["description"]].fillna("").astype(str),
        "amount": frame[cols["amount"]].map(amount).abs(),
        "payment_method": frame[cols["payment_method"]].astype(str) if "payment_method" in cols else "Não identificado",
        "raw_amount": frame[cols["amount"]],
        "raw_direction": frame["Direção"].fillna("").astype(str) if "Direção" in frame.columns else "",
        "raw_values": frame["Valores encontrados"].fillna("").astype(str) if "Valores encontrados" in frame.columns else "",
        "source_line": frame["Linha OCR"].fillna("").astype(str) if "Linha OCR" in frame.columns else "",
    }).dropna(subset=["date"])
    out["description_clean"] = out["description"].map(clean)
    details = [infer_direction_details(f"{row.description} {row.raw_direction}", row.raw_amount, row.payment_method) for row in out.itertuples()]
    out[["direction", "direction_confidence", "transaction_kind", "direction_reason"]] = pd.DataFrame(details, index=out.index)
    return out.drop(columns=["raw_amount", "raw_direction"]).sort_values("date").reset_index(drop=True)


def _rule_map(custom_rules: pd.DataFrame | dict | None) -> list[tuple[str, str]]:
    if custom_rules is None:
        return []
    if isinstance(custom_rules, dict):
        return [(clean(pattern), category) for pattern, category in custom_rules.items()]
    if not custom_rules.empty and {"pattern", "category"}.issubset(custom_rules.columns):
        return [(clean(row.pattern), row.category) for row in custom_rules.itertuples()]
    return []


def learned_category(text: str, learned: list[tuple[str, str]]) -> tuple[str | None, float]:
    """Combina coincidência exata, tokens e semelhança para tolerar variações do OCR."""
    best_category, best_score = None, 0.0
    text_tokens = set(clean(text).split())
    for pattern, category in learned:
        if not pattern:
            continue
        pattern_tokens = set(pattern.split())
        if pattern in text:
            score = .99
        else:
            overlap = len(text_tokens & pattern_tokens) / max(1, len(pattern_tokens))
            sequence = SequenceMatcher(None, pattern, clean(text)).ratio()
            score = max(overlap * .94, sequence * .84)
        if score > best_score:
            best_category, best_score = category, score
    return (best_category, best_score) if best_score >= .78 else (None, 0.0)


def classify(frame: pd.DataFrame, custom_rules: pd.DataFrame | dict | None = None) -> pd.DataFrame:
    out = frame.copy()
    learned = _rule_map(custom_rules)
    categories, confidences = [], []
    for text in out["description_clean"]:
        remembered, learned_confidence = learned_category(text, learned)
        if remembered:
            categories.append(remembered); confidences.append(learned_confidence); continue
        scores = {cat: sum(clean(word) in text for word in words) for cat, words in RULES.items()}
        category = max(scores, key=scores.get) if max(scores.values(), default=0) else "Não classificado"
        matches = scores.get(category, 0)
        categories.append(category); confidences.append(min(0.95, 0.55 + matches * 0.12) if matches else 0.0)
    out["category"], out["confidence"] = categories, confidences
    out["merchant"] = out["description"].map(merchant_name)
    installment = out["description"].str.extract(r"(?i)(?:parcela\s*)?(\d{1,2})\s*(?:/|de)\s*(\d{1,2})")
    out["installment_number"] = pd.to_numeric(installment[0], errors="coerce")
    out["installment_total"] = pd.to_numeric(installment[1], errors="coerce")
    out["installment_remaining"] = (out["installment_total"] - out["installment_number"]).clip(lower=0)
    out["installment_hint"] = out["installment_total"].notna() | out["description"].str.contains("parcela", case=False, na=False)
    months = out.assign(month=out["date"].dt.to_period("M")).groupby("description_clean")["month"].nunique()
    out["recurring"] = out["description_clean"].map(months).fillna(0).ge(2)
    out["due_day"] = out["date"].dt.day
    return out


def generate_alerts(frame: pd.DataFrame, budgets: pd.DataFrame | None = None) -> list[dict[str, str]]:
    if frame.empty:
        return []
    alerts: list[dict[str, str]] = []
    expenses = frame[frame["direction"] == "Saída"].copy()
    income = frame[frame["direction"] == "Entrada"].copy()
    if not expenses.empty:
        threshold = expenses["amount"].median() * 3
        for row in expenses[expenses["amount"] > max(threshold, 300)].nlargest(3, "amount").itertuples():
            alerts.append({"level": "atenção", "title": "Gasto fora do padrão", "text": f"{row.description}: R$ {row.amount:,.2f} está acima do padrão do histórico."})
        if budgets is not None and not budgets.empty:
            spent = expenses.groupby("category")["amount"].sum()
            for budget in budgets.itertuples():
                value, limit = float(spent.get(budget.category, 0)), float(budget.limit_amount)
                if limit and value / limit >= .8:
                    alerts.append({"level": "alto" if value > limit else "atenção", "title": f"Orçamento de {budget.category}", "text": f"Uso de {value / limit:.0%}: R$ {value:,.2f} de R$ {limit:,.2f}."})
        recurring = expenses[flag_mask(expenses, "recurring")]
        if not recurring.empty:
            duplicated = recurring.groupby("merchant").size()
            for merchant, count in duplicated[duplicated >= 2].items():
                alerts.append({"level": "atenção", "title": "Possível assinatura duplicada", "text": f"{merchant} aparece {count} vezes entre despesas recorrentes."})
    if len(income) and "date" in income:
        months = income.assign(month=income.date.dt.to_period("M")).groupby("month")["amount"].sum().sort_index()
        if len(months) >= 2 and months.iloc[-1] < months.iloc[-2] * .8:
            alerts.append({"level": "atenção", "title": "Queda de entradas", "text": "As entradas do período mais recente caíram mais de 20% em relação ao anterior."})
    net = income["amount"].sum() - expenses["amount"].sum()
    if net < 0:
        alerts.append({"level": "alto", "title": "Saldo projetado negativo", "text": f"No recorte atual, as saídas superam as entradas em R$ {abs(net):,.2f}."})
    return alerts[:8]


def recommendations(frame: pd.DataFrame) -> list[dict[str, str]]:
    expenses = frame[frame["direction"] == "Saída"]
    total = expenses["amount"].sum()
    if not total:
        return []
    result = []
    for category, group in expenses.groupby("category"):
        share = group["amount"].sum() / total
        if share >= .30:
            result.append({"title": f"Peso elevado em {category}", "text": f"Essa categoria representa {share:.1%} das saídas. Revise os maiores lançamentos e separe gastos essenciais dos ajustáveis."})
    recurring = expenses.loc[flag_mask(expenses, "recurring"), "amount"].sum()
    if recurring:
        result.append({"title": "Revise despesas recorrentes", "text": f"Há aproximadamente R$ {recurring:,.2f} em lançamentos possivelmente recorrentes."})
    return result
