"""Base documental local do agente MATHEWZINHO."""

from __future__ import annotations

import re
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
EBOOKS_DIR = ROOT / "ebooks"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


def ollama_config() -> tuple[str, str]:
    return (
        os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/"),
        os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    )


def ollama_status() -> tuple[bool, str]:
    """Verifica o serviço local e se o modelo configurado está instalado."""
    base_url, model = ollama_config()
    try:
        with urlopen(f"{base_url}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        installed = {item.get("name", "") for item in payload.get("models", [])}
        model_base = model.split(":", 1)[0]
        available = model in installed or any(name.split(":", 1)[0] == model_base for name in installed)
        return available, model
    except (OSError, HTTPError, URLError, ValueError, json.JSONDecodeError):
        return False, model


def index_pdfs() -> pd.DataFrame:
    EBOOKS_DIR.mkdir(exist_ok=True)
    chunks = []
    for path in sorted(EBOOKS_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
                if not text:
                    continue
                words = text.split()
                for start in range(0, len(words), 180):
                    chunk = " ".join(words[start:start + 220]).strip()
                    if len(chunk) >= 80:
                        chunks.append({"source": path.name, "page": page_number, "text": chunk})
        except Exception:
            continue
    return pd.DataFrame(chunks, columns=["source", "page", "text"])


def _financial_context(transactions: pd.DataFrame | None) -> str:
    if transactions is None or transactions.empty:
        return "Ainda não há lançamentos financeiros consolidados."
    frame = transactions.copy()
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    frame["direction"] = frame["direction"].fillna("Indeterminada")
    expenses = frame[frame["direction"] == "Saída"]
    income = frame[frame["direction"] == "Entrada"]
    lines = [
        f"Lançamentos analisados: {len(frame)}",
        f"Total de entradas identificadas: R$ {income['amount'].sum():.2f}",
        f"Total de saídas identificadas: R$ {expenses['amount'].sum():.2f}",
        f"Lançamentos com direção indeterminada: {int((frame['direction'] == 'Indeterminada').sum())}",
    ]
    if not expenses.empty and "category" in expenses:
        top_categories = expenses.groupby("category")["amount"].sum().sort_values(ascending=False).head(8)
        lines.append("Maiores categorias: " + "; ".join(f"{cat}: R$ {value:.2f}" for cat, value in top_categories.items()))
    if not expenses.empty and "merchant" in expenses:
        top_merchants = expenses.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(5)
        lines.append("Maiores estabelecimentos: " + "; ".join(f"{merchant}: R$ {value:.2f}" for merchant, value in top_merchants.items()))
    recent = frame.sort_values("date", ascending=False).head(12) if "date" in frame else frame.head(12)
    lines.append("Lançamentos recentes: " + " | ".join(
        f"{row.get('date', '')}: {row.get('description', '')} — {row.get('direction', '')} R$ {row.get('amount', 0):.2f}"
        for _, row in recent.iterrows()
    ))
    return "\n".join(lines)


def financial_tools(question: str, transactions: pd.DataFrame | None) -> tuple[str, list[str]]:
    """Ferramentas determinísticas: o agente recebe números calculados, não inventados."""
    if transactions is None or transactions.empty:
        return "Não há dados financeiros para calcular.", []
    frame = transactions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    text = question.lower()
    result, used = [], []
    expenses = frame[frame["direction"] == "Saída"]
    income = frame[frame["direction"] == "Entrada"]
    if any(word in text for word in ("gastei", "despesa", "categoria", "onde")):
        groups = expenses.groupby("category")["amount"].sum().sort_values(ascending=False).head(5)
        result.append("Gastos por categoria: " + "; ".join(f"{name}: R$ {value:.2f}" for name, value in groups.items()))
        used.append("gastos por categoria")
    if any(word in text for word in ("aumentou", "mudou", "compar", "mês")) and frame["date"].notna().any():
        monthly = expenses.assign(month=expenses.date.dt.to_period("M")).groupby(["month", "category"])["amount"].sum().unstack(fill_value=0).sort_index()
        if len(monthly) >= 2:
            diff = (monthly.iloc[-1] - monthly.iloc[-2]).sort_values(ascending=False).head(4)
            result.append("Variação do último período: " + "; ".join(f"{name}: R$ {value:+.2f}" for name, value in diff.items()))
            used.append("comparação mensal")
    if any(word in text for word in ("saldo", "orçamento", "posso guardar", "econom")):
        total_income, total_expenses = income.amount.sum(), expenses.amount.sum()
        result.append(f"Entradas: R$ {total_income:.2f}; saídas: R$ {total_expenses:.2f}; saldo calculado: R$ {total_income-total_expenses:.2f}.")
        used.append("saldo consolidado")
    pending = int((frame["review_status"] != "Concluído").sum()) if "review_status" in frame.columns else 0
    if pending:
        result.append(f"Atenção: {pending} lançamento(s) ainda não foram confirmados.")
        used.append("qualidade do histórico")
    return "\n".join(result) or _financial_context(frame), used


def parse_category_instruction(instruction: str, categories: list[str]) -> dict[str, str] | None:
    """Entende comandos locais de categorização sem depender de modelo externo.

    Exemplos: 'tudo que tiver uber é Transporte' ou 'ifood e rappi vão para Alimentação'.
    """
    original = re.sub(r"\s+", " ", instruction or "").strip()
    if not original:
        return None
    lowered = original.lower()
    category = next((item for item in sorted(categories, key=len, reverse=True) if item.lower() in lowered), None)
    if not category:
        return None
    before = re.split(r"(?i)\b(?:é|são|vai|vão|fica|ficam|categoria|para|como)\b", original, maxsplit=1)[0]
    before = re.sub(r"(?i)\b(?:tudo|todos|todas|lançamentos|itens|que|tiver|contiver|aparecer|com|o|a|os|as)\b", " ", before)
    keywords = [item.strip(" .,:;\"'") for item in re.split(r"(?i)\s*(?:,|/|\be\b|\bou\b)\s*", before) if len(item.strip()) >= 2]
    if not keywords:
        return None
    return {"category": category, "keywords": keywords, "instruction": original}


def _generate_with_ollama(prompt: str) -> str | None:
    base_url, model = ollama_config()
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }).encode("utf-8")
    request = Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = str(result.get("response", "")).strip()
        return answer or None
    except (OSError, HTTPError, URLError, ValueError, json.JSONDecodeError):
        return None


def ask(question: str, top_k: int = 4, transactions: pd.DataFrame | None = None) -> dict:
    """Usa recuperação documental e, quando disponível, geração local pelo Ollama."""
    index = index_pdfs()
    if index.empty:
        relevant = index
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None, sublinear_tf=True)
        matrix = vectorizer.fit_transform(index["text"])
        query = vectorizer.transform([question])
        scores = cosine_similarity(query, matrix).ravel()
        best = index.assign(score=scores).sort_values("score", ascending=False).head(top_k)
        relevant = best[best["score"] > 0.03]

    sources = [
        {"file": row["source"], "page": int(row["page"]), "score": round(float(row["score"]), 3)}
        for _, row in relevant.iterrows()
    ]
    document_context = "\n\n".join(
        f"Fonte: {row['source']}, página {int(row['page'])}\nTrecho: {row['text']}"
        for _, row in relevant.iterrows()
    ) or "Nenhum trecho relevante foi recuperado dos e-books."
    tool_context, tools_used = financial_tools(question, transactions)
    prompt = f"""
Você é MATHEWZINHO, o agente local de organização financeira do FINP MAT.
Responda em português brasileiro, com clareza e sem inventar dados.
Use os lançamentos apenas para cálculos e observações; use os e-books para
fundamentos de educação financeira. Diferencie fato, interpretação e sugestão.
Não prometa rentabilidade nem dê recomendação personalizada de investimento.
Avise quando houver lançamentos indeterminados ou dados insuficientes.
Quando usar os e-books, cite arquivo e página no texto.

PERGUNTA:
{question}

CONTEXTO FINANCEIRO:
{tool_context}

TRECHOS DOS E-BOOKS:
{document_context}
""".strip()
    available, model = ollama_status()
    if available:
        generated = _generate_with_ollama(prompt)
        if generated:
            return {"answer": generated, "sources": sources, "provider": f"Ollama ({model})", "tools": tools_used}
    if relevant.empty:
        answer = "Não encontrei um trecho suficientemente relacionado nos e-books disponíveis. Adicione materiais ou formule a dúvida com outros termos."
    else:
        answer = "Encontrei estes fundamentos na sua base documental. Ative o Ollama para que o MATHEWZINHO sintetize e relacione os trechos aos seus lançamentos:\n\n"
        answer += "\n\n".join(f"• {row['text']}" for _, row in relevant.iterrows())
    return {"answer": answer, "sources": sources, "provider": "Busca documental local", "tools": tools_used}
