"""Dados sintéticos para conhecer o FINP MAT sem expor dados pessoais."""
from __future__ import annotations

from datetime import date

from storage import add_transaction


def load_demo_data() -> int:
    items = [
        ("2026-06-05", "Salário demonstrativo", 5200, "Entrada", "Receitas"),
        ("2026-06-06", "Aluguel demonstrativo", 1300, "Saída", "Moradia"),
        ("2026-06-10", "Supermercado demonstrativo", 430, "Saída", "Alimentação"),
        ("2026-06-14", "Pix enviado demonstrativo", 95, "Saída", "Transferências"),
        ("2026-07-05", "Salário demonstrativo", 5200, "Entrada", "Receitas"),
        ("2026-07-06", "Aluguel demonstrativo", 1300, "Saída", "Moradia"),
        ("2026-07-12", "Supermercado demonstrativo", 510, "Saída", "Alimentação"),
        ("2026-07-20", "Streaming demonstrativo", 39.9, "Saída", "Lazer"),
        ("2026-08-05", "Salário demonstrativo", 5200, "Entrada", "Receitas"),
        ("2026-08-06", "Aluguel demonstrativo", 1300, "Saída", "Moradia"),
        ("2026-08-11", "Supermercado demonstrativo", 680, "Saída", "Alimentação"),
        ("2026-08-15", "Pix recebido demonstrativo", 210, "Entrada", "Transferências"),
        ("2026-08-23", "Academia demonstrativo", 89.9, "Saída", "Saúde"),
    ]
    for item in items:
        add_transaction(*item, source_file="demonstração")
    return len(items)
