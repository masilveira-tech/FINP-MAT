"""Análises proativas e explicáveis usadas pelo painel e pelo MATHEWZINHO."""
from __future__ import annotations

import pandas as pd

from engine import flag_mask


def _expenses(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.get("direction", pd.Series(dtype=str)) == "Saída"].copy() if not frame.empty else frame.copy()


def what_changed(frame: pd.DataFrame) -> list[dict[str, object]]:
    expenses = _expenses(frame)
    if expenses.empty or "date" not in expenses:
        return []
    monthly = expenses.assign(month=expenses.date.dt.to_period("M")).groupby(["month", "category"], as_index=False)["amount"].sum()
    periods = sorted(monthly.month.unique())
    if len(periods) < 2:
        return []
    now, before = periods[-1], periods[-2]
    current = monthly[monthly.month == now].set_index("category").amount
    previous = monthly[monthly.month == before].set_index("category").amount
    rows = []
    for category in set(current.index) | set(previous.index):
        difference = float(current.get(category, 0) - previous.get(category, 0))
        if abs(difference) >= 1:
            rows.append({"category": category, "difference": difference, "current": float(current.get(category, 0)), "previous": float(previous.get(category, 0))})
    return sorted(rows, key=lambda row: abs(row["difference"]), reverse=True)[:5]


def weekly_summary(frame: pd.DataFrame) -> dict[str, float]:
    expenses = _expenses(frame)
    if expenses.empty:
        return {"current": 0.0, "previous": 0.0, "change": 0.0}
    latest = expenses.date.max().normalize()
    current = float(expenses[expenses.date >= latest - pd.Timedelta(days=6)].amount.sum())
    previous = float(expenses[(expenses.date >= latest - pd.Timedelta(days=13)) & (expenses.date < latest - pd.Timedelta(days=6))].amount.sum())
    return {"current": current, "previous": previous, "change": current - previous}


def forecast_categories(frame: pd.DataFrame) -> pd.DataFrame:
    expenses = _expenses(frame)
    if expenses.empty:
        return pd.DataFrame(columns=["category", "forecast", "months"])
    base = expenses.assign(month=expenses.date.dt.to_period("M")).groupby(["category", "month"], as_index=False).amount.sum()
    result = base.groupby("category", as_index=False).agg(forecast=("amount", "mean"), months=("month", "nunique"))
    return result.sort_values("forecast", ascending=False)


def subscription_changes(frame: pd.DataFrame) -> pd.DataFrame:
    expenses = _expenses(frame)
    if expenses.empty:
        return pd.DataFrame(columns=["merchant", "previous", "current", "change"])
    recurring = expenses[flag_mask(expenses, "recurring")].copy()
    if recurring.empty:
        return pd.DataFrame(columns=["merchant", "previous", "current", "change"])
    monthly = recurring.assign(month=recurring.date.dt.to_period("M")).groupby(["merchant", "month"], as_index=False).amount.median()
    rows = []
    for merchant, group in monthly.groupby("merchant"):
        group = group.sort_values("month")
        if len(group) >= 2:
            previous, current = float(group.iloc[-2].amount), float(group.iloc[-1].amount)
            if current > previous + .01:
                rows.append({"merchant": merchant, "previous": previous, "current": current, "change": current - previous})
    return pd.DataFrame(rows).sort_values("change", ascending=False) if rows else pd.DataFrame(columns=["merchant", "previous", "current", "change"])


def health_score(frame: pd.DataFrame, budgets: pd.DataFrame | None = None) -> tuple[int, list[str]]:
    if frame.empty:
        return 0, ["Importe ou registre lançamentos para iniciar o diagnóstico."]
    score, reasons = 100, []
    uncertain = (frame.direction == "Indeterminada").sum() + (frame.review_status != "Concluído").sum()
    if uncertain:
        score -= min(25, uncertain * 3); reasons.append("Há lançamentos pendentes ou sem direção confirmada.")
    income = float(frame.loc[frame.direction == "Entrada", "amount"].sum())
    expenses = float(frame.loc[frame.direction == "Saída", "amount"].sum())
    if expenses > income and income:
        score -= 25; reasons.append("As despesas do período superam as entradas.")
    if budgets is not None and not budgets.empty:
        used = frame[frame.direction == "Saída"].groupby("category").amount.sum()
        exceeded = sum(used.get(row.category, 0) > row.limit_amount for row in budgets.itertuples() if row.limit_amount)
        if exceeded:
            score -= min(20, exceeded * 8); reasons.append("Há orçamento(s) acima do limite definido.")
    if not reasons:
        reasons.append("Os indicadores atuais não apontam um risco relevante no recorte analisado.")
    return max(0, int(score)), reasons


def decision_priorities(frame: pd.DataFrame) -> pd.DataFrame:
    forecast = forecast_categories(frame)
    if forecast.empty:
        return pd.DataFrame(columns=["ação", "impacto_estimado"])
    top = forecast.head(3).copy()
    top["ação"] = top["category"].map(lambda value: f"Reduzir 10% em {value}")
    top["impacto_estimado"] = top["forecast"] * .10
    return top[["ação", "impacto_estimado"]]
