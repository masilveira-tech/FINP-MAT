"""Persistência local, auditável e resistente a duplicidades do FINP MAT."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "extratos"
ATTACHMENTS_DIR = ROOT / "data" / "anexos"
DB_PATH = ROOT / "data" / "finp_mat.db"


def _clean(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def transaction_fingerprint(date_value: object, description: object, amount: object, direction: object = "") -> str:
    """Chave estável: evita importar a mesma movimentação por arquivos distintos."""
    numeric = pd.to_numeric(amount, errors="coerce")
    numeric = 0.0 if pd.isna(numeric) else abs(float(numeric))
    when = pd.to_datetime(date_value, errors="coerce")
    date_key = str(when.date()) if not pd.isna(when) else ""
    payload = "|".join((date_key, _clean(description), f"{numeric:.2f}", _clean(direction)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row[1] for row in connection.execute("PRAGMA table_info(transactions)").fetchall()}


def _audit(connection: sqlite3.Connection, transaction_id: int | None, action: str, before: dict | None, after: dict | None, note: str = "") -> None:
    connection.execute("INSERT INTO audit_log(transaction_id, action, before_json, after_json, note) VALUES (?, ?, ?, ?, ?)", (transaction_id, action, json.dumps(before or {}, ensure_ascii=False, default=str), json.dumps(after or {}, ensure_ascii=False, default=str), note))


def init_storage() -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    FILES_DIR.mkdir(exist_ok=True)
    ATTACHMENTS_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, description TEXT, merchant TEXT,
            amount REAL NOT NULL, direction TEXT, direction_confidence REAL DEFAULT 0, category TEXT,
            confidence REAL DEFAULT 0, payment_method TEXT DEFAULT 'Não identificado', recurring INTEGER DEFAULT 0,
            installment_hint INTEGER DEFAULT 0, source_file TEXT, source_hash TEXT,
            review_status TEXT DEFAULT 'Pendente', user_note TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        definitions = (("direction_confidence", "REAL DEFAULT 0"), ("confidence", "REAL DEFAULT 0"), ("payment_method", "TEXT DEFAULT 'Não identificado'"), ("recurring", "INTEGER DEFAULT 0"), ("installment_hint", "INTEGER DEFAULT 0"), ("review_status", "TEXT DEFAULT 'Pendente'"), ("user_note", "TEXT DEFAULT ''"), ("transaction_kind", "TEXT DEFAULT 'Não identificado'"), ("direction_reason", "TEXT DEFAULT ''"), ("raw_values", "TEXT DEFAULT ''"), ("source_line", "TEXT DEFAULT ''"), ("fingerprint", "TEXT DEFAULT ''"), ("installment_number", "INTEGER"), ("installment_total", "INTEGER"), ("installment_remaining", "INTEGER"), ("due_day", "INTEGER"), ("subcategory", "TEXT DEFAULT ''"), ("tags", "TEXT DEFAULT ''"), ("owner", "TEXT DEFAULT 'Pessoal'"), ("reimbursement_status", "TEXT DEFAULT 'Não'"))
        existing = _columns(connection)
        for column, definition in definitions:
            if column not in existing:
                connection.execute(f"ALTER TABLE transactions ADD COLUMN {column} {definition}")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_fingerprint ON transactions(fingerprint) WHERE fingerprint IS NOT NULL AND fingerprint <> ''")
        connection.execute("""CREATE TABLE IF NOT EXISTS uploaded_files (source_hash TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL, transaction_month TEXT, uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        upload_columns = {row[1] for row in connection.execute("PRAGMA table_info(uploaded_files)").fetchall()}
        for column, definition in (("total_rows", "INTEGER DEFAULT 0"), ("review_rows", "INTEGER DEFAULT 0"), ("quality_score", "REAL DEFAULT 0"), ("bank_profile", "TEXT DEFAULT 'Genérico'")):
            if column not in upload_columns:
                connection.execute(f"ALTER TABLE uploaded_files ADD COLUMN {column} {definition}")
        connection.execute("""CREATE TABLE IF NOT EXISTS budgets (id INTEGER PRIMARY KEY AUTOINCREMENT, month TEXT NOT NULL DEFAULT 'global', category TEXT NOT NULL, limit_amount REAL NOT NULL DEFAULT 0, updated_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(month, category))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS goals (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, target_amount REAL NOT NULL DEFAULT 0, current_amount REAL NOT NULL DEFAULT 0, deadline TEXT, status TEXT NOT NULL DEFAULT 'Em andamento', created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS category_rules (id INTEGER PRIMARY KEY AUTOINCREMENT, pattern TEXT UNIQUE NOT NULL, category TEXT NOT NULL, source TEXT DEFAULT 'correção manual', created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id INTEGER, action TEXT NOT NULL, before_json TEXT, after_json TEXT, note TEXT DEFAULT '', changed_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS reconciliations (source_hash TEXT PRIMARY KEY, opening_balance REAL, closing_balance REAL, notes TEXT DEFAULT '', updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS bill_schedule (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL, due_day INTEGER NOT NULL, payment_method TEXT DEFAULT 'Conta', active INTEGER DEFAULT 1, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS financial_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, account_type TEXT NOT NULL,
            institution TEXT DEFAULT '', current_balance REAL DEFAULT 0, credit_limit REAL DEFAULT 0,
            closing_day INTEGER, due_day INTEGER, active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        account_columns = {row[1] for row in connection.execute("PRAGMA table_info(financial_accounts)").fetchall()}
        if "interest_rate" not in account_columns:
            connection.execute("ALTER TABLE financial_accounts ADD COLUMN interest_rate REAL DEFAULT 0")
        connection.execute("""CREATE TABLE IF NOT EXISTS category_catalog (
            name TEXT PRIMARY KEY, icon TEXT DEFAULT '•', color TEXT DEFAULT '#2e6bff', active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS category_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pattern TEXT NOT NULL, category TEXT NOT NULL, subcategory TEXT DEFAULT '',
            hits INTEGER DEFAULT 1, updated_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(pattern, category, subcategory))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS transaction_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id INTEGER NOT NULL, filename TEXT NOT NULL,
            path TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        for name, icon, color in (("Alimentação", "🍽️", "#f59e0b"), ("Moradia", "🏠", "#7c3aed"), ("Transporte", "🚗", "#2e6bff"), ("Saúde", "❤️", "#ef476f"), ("Educação", "📚", "#14b8a6"), ("Lazer", "🎮", "#a855f7"), ("Financeiro", "🏦", "#475569"), ("Transferências", "↔️", "#64748b"), ("Receitas", "↗️", "#16a34a")):
            connection.execute("INSERT OR IGNORE INTO category_catalog(name,icon,color) VALUES(?,?,?)", (name, icon, color))


def safe_name(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(filename).name)


def _row_payload(row: pd.Series, filename: str = "", file_hash: str = "") -> tuple:
    date_value = pd.to_datetime(row.get("date"), errors="coerce")
    amount_value = pd.to_numeric(row.get("amount"), errors="coerce")
    if pd.isna(date_value) or pd.isna(amount_value):
        raise ValueError("Lançamento sem data ou valor válido")
    direction = str(row.get("direction", "Indeterminada"))
    def optional_int(key: str):
        value = pd.to_numeric(row.get(key), errors="coerce")
        return None if pd.isna(value) else int(value)
    return (date_value.strftime("%Y-%m-%d"), str(row.get("description", "")), str(row.get("merchant", "")), abs(float(amount_value)), direction, float(row.get("direction_confidence", 0) or 0), str(row.get("category", "Não classificado")), str(row.get("subcategory", "")), str(row.get("tags", "")), str(row.get("owner", "Pessoal")), str(row.get("reimbursement_status", "Não")), float(row.get("confidence", 0) or 0), str(row.get("payment_method", "Não identificado")), int(bool(row.get("recurring", False))), int(bool(row.get("installment_hint", False))), filename, file_hash, str(row.get("review_status", "Pendente")), str(row.get("user_note", "")), str(row.get("transaction_kind", "Não identificado")), str(row.get("direction_reason", "")), str(row.get("raw_values", "")), str(row.get("source_line", "")), transaction_fingerprint(date_value, row.get("description", ""), amount_value, direction), optional_int("installment_number"), optional_int("installment_total"), optional_int("installment_remaining"), optional_int("due_day"))


def _insert_transaction(connection: sqlite3.Connection, payload: tuple) -> int | None:
    cursor = connection.execute("""INSERT OR IGNORE INTO transactions(date,description,merchant,amount,direction,direction_confidence,category,subcategory,tags,owner,reimbursement_status,confidence,payment_method,recurring,installment_hint,source_file,source_hash,review_status,user_note,transaction_kind,direction_reason,raw_values,source_line,fingerprint,installment_number,installment_total,installment_remaining,due_day) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", payload)
    return int(cursor.lastrowid) if cursor.rowcount else None


def save_upload(filename: str, content: bytes, transactions: pd.DataFrame, bank_profile: str = "Genérico") -> tuple[bool, str]:
    init_storage()
    file_hash = hashlib.sha256(content).hexdigest()
    months = transactions["date"].dropna().dt.strftime("%Y-%m").unique().tolist() if not transactions.empty else []
    month = months[0] if len(months) == 1 else "multiplos_periodos"
    target_dir = FILES_DIR / month
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{file_hash[:12]}_{safe_name(filename)}"
    with sqlite3.connect(DB_PATH) as connection:
        if connection.execute("SELECT 1 FROM uploaded_files WHERE source_hash = ?", (file_hash,)).fetchone():
            return False, str(target)
        target.write_bytes(content)
        confidence = pd.to_numeric(transactions.get("direction_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0) if not transactions.empty else pd.Series(dtype=float)
        review_rows = int((transactions["review_status"] != "Concluído").sum()) if (not transactions.empty and "review_status" in transactions.columns) else int(len(transactions))
        quality = float(confidence.mean()) if len(confidence) else 0.0
        connection.execute("INSERT INTO uploaded_files(source_hash, filename, path, transaction_month, total_rows, review_rows, quality_score, bank_profile) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (file_hash, filename, str(target), month, int(len(transactions)), review_rows, quality, bank_profile))
        for _, row in transactions.iterrows():
            try:
                payload = _row_payload(row, filename, file_hash)
                inserted = _insert_transaction(connection, payload)
                if inserted:
                    _audit(connection, inserted, "importação", None, {"arquivo": filename, "regra": payload[20]})
            except ValueError:
                continue
    return True, str(target)


def load_transactions() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC, id DESC", connection, parse_dates=["date"])


def list_uploads() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT filename, source_hash, transaction_month, bank_profile, total_rows, review_rows, quality_score, uploaded_at FROM uploaded_files ORDER BY uploaded_at DESC", connection)


def list_category_rules() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT id, pattern, category, source, updated_at FROM category_rules ORDER BY updated_at DESC", connection)


def learn_category_rule(pattern: str, category: str) -> None:
    pattern = _clean(pattern)
    if len(pattern) < 3 or not category.strip() or category == "Não classificado":
        return
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""INSERT INTO category_rules(pattern, category) VALUES (?, ?) ON CONFLICT(pattern) DO UPDATE SET category=excluded.category, updated_at=CURRENT_TIMESTAMP""", (pattern, category.strip()))


def save_keyword_instruction(keyword: str, category: str, subcategory: str = "") -> None:
    """Memória explícita do usuário: palavra-chave → categoria, aplicada em importações futuras."""
    pattern = _clean(keyword)
    if len(pattern) < 2 or not category.strip():
        raise ValueError("Informe uma palavra-chave e uma categoria.")
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""INSERT INTO category_rules(pattern,category,source) VALUES(?,?,?)
            ON CONFLICT(pattern) DO UPDATE SET category=excluded.category,source=excluded.source,updated_at=CURRENT_TIMESTAMP""", (pattern, category.strip(), "instrução do usuário"))
        connection.execute("""INSERT INTO category_examples(pattern,category,subcategory) VALUES(?,?,?)
            ON CONFLICT(pattern,category,subcategory) DO UPDATE SET hits=hits+1,updated_at=CURRENT_TIMESTAMP""", (pattern, category.strip(), subcategory.strip()))
        connection.execute("INSERT OR IGNORE INTO category_catalog(name) VALUES(?)", (category.strip(),))


def matching_transaction_ids(keyword: str) -> list[int]:
    normalized = _clean(keyword)
    if not normalized:
        return []
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute("SELECT id, description, merchant FROM transactions").fetchall()
    return [int(row[0]) for row in rows if normalized in _clean(row[1]) or normalized in _clean(row[2])]


def list_keyword_instructions() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT id,pattern,category,source,updated_at FROM category_rules WHERE source='instrução do usuário' ORDER BY updated_at DESC", connection)


def update_transaction(transaction_id: int, amount: float, direction: str, category: str, review_status: str, user_note: str = "", learn_category: bool = False, subcategory: str | None = None, tags: str | None = None, owner: str | None = None, reimbursement_status: str | None = None) -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("SELECT * FROM transactions WHERE id=?", (int(transaction_id),))
        row = cursor.fetchone()
        if not row:
            return
        before = dict(zip([item[0] for item in cursor.description], row))
        fingerprint = transaction_fingerprint(before["date"], before["description"], amount, direction)
        subcategory = before.get("subcategory", "") if subcategory is None else subcategory.strip()
        tags = before.get("tags", "") if tags is None else tags.strip()
        owner = before.get("owner", "Pessoal") if owner is None else owner.strip()
        reimbursement_status = before.get("reimbursement_status", "Não") if reimbursement_status is None else reimbursement_status
        connection.execute("UPDATE transactions SET amount=?,direction=?,category=?,subcategory=?,tags=?,owner=?,reimbursement_status=?,review_status=?,user_note=?,fingerprint=? WHERE id=?", (abs(float(amount)), direction, category, subcategory, tags, owner, reimbursement_status, review_status, user_note, fingerprint, int(transaction_id)))
        after = {**before, "amount": abs(float(amount)), "direction": direction, "category": category, "subcategory": subcategory, "tags": tags, "owner": owner, "reimbursement_status": reimbursement_status, "review_status": review_status, "user_note": user_note}
        _audit(connection, int(transaction_id), "correção manual", before, after, user_note)
        if learn_category:
            pattern = _clean(before.get("merchant") or before.get("description", ""))
            if len(pattern) >= 3 and category != "Não classificado":
                connection.execute("""INSERT INTO category_rules(pattern, category) VALUES (?, ?) ON CONFLICT(pattern) DO UPDATE SET category=excluded.category, updated_at=CURRENT_TIMESTAMP""", (pattern, category.strip()))
                connection.execute("""INSERT INTO category_examples(pattern,category,subcategory) VALUES(?,?,?) ON CONFLICT(pattern,category,subcategory) DO UPDATE SET hits=hits+1,updated_at=CURRENT_TIMESTAMP""", (pattern, category.strip(), subcategory))
                connection.execute("INSERT OR IGNORE INTO category_catalog(name) VALUES(?)", (category.strip(),))


def split_transaction(transaction_id: int, allocations: list[tuple[str, float]]) -> None:
    """Substitui um lançamento por partes categorizadas, preservando a trilha de auditoria."""
    valid = [(category.strip(), abs(float(value))) for category, value in allocations if category.strip() and float(value) > 0]
    if len(valid) < 2:
        raise ValueError("Informe pelo menos duas partes com valor maior que zero.")
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("SELECT * FROM transactions WHERE id=?", (int(transaction_id),))
        row = cursor.fetchone()
        if not row:
            raise ValueError("Lançamento não encontrado.")
        parent = dict(zip([item[0] for item in cursor.description], row))
        total = round(sum(value for _, value in valid), 2)
        if abs(total - abs(float(parent["amount"]))) > .01:
            raise ValueError("A soma das partes precisa ser igual ao valor original.")
        connection.execute("UPDATE transactions SET direction='Ignorado',category='Dividido',review_status='Concluído',user_note=? WHERE id=?", ("Substituído por lançamento dividido", int(transaction_id)))
        _audit(connection, int(transaction_id), "lançamento dividido", parent, {"partes": valid})
        for category, value in valid:
            child = pd.Series({"date": parent["date"], "description": f"{parent['description']} · parte {category}", "merchant": parent.get("merchant", ""), "amount": value, "direction": parent["direction"], "direction_confidence": 1.0, "category": category, "confidence": 1.0, "payment_method": parent.get("payment_method", "Não identificado"), "review_status": "Concluído", "transaction_kind": "Parte de lançamento dividido", "direction_reason": "Dividido manualmente", "user_note": f"Origem #{transaction_id}"})
            child_id = _insert_transaction(connection, _row_payload(child, parent.get("source_file", ""), parent.get("source_hash", "")))
            if child_id:
                _audit(connection, child_id, "criado por divisão", None, {"origem": transaction_id, "categoria": category, "valor": value})


def restore_last_change(transaction_id: int) -> bool:
    """Restaura os campos financeiros a partir do último histórico com estado anterior."""
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        log = connection.execute("SELECT before_json FROM audit_log WHERE transaction_id=? AND before_json NOT IN ('', '{}', NULL) ORDER BY id DESC LIMIT 1", (int(transaction_id),)).fetchone()
        if not log:
            return False
        before = json.loads(log[0])
        fields = ("amount", "direction", "category", "review_status", "user_note", "fingerprint")
        if not all(field in before for field in fields):
            return False
        current = connection.execute("SELECT * FROM transactions WHERE id=?", (int(transaction_id),)).fetchone()
        connection.execute("UPDATE transactions SET amount=?,direction=?,category=?,review_status=?,user_note=?,fingerprint=? WHERE id=?", tuple(before[field] for field in fields) + (int(transaction_id),))
        _audit(connection, int(transaction_id), "restauração", None, before, "Última alteração desfeita")
        return True


def list_category_catalog() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT name,icon,color,active FROM category_catalog WHERE active=1 ORDER BY name", connection)


def save_category(name: str, icon: str = "•", color: str = "#2e6bff") -> None:
    if not name.strip():
        return
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("INSERT INTO category_catalog(name,icon,color) VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET icon=excluded.icon,color=excluded.color,active=1", (name.strip(), icon or "•", color or "#2e6bff"))


def list_category_examples() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT pattern,category,subcategory,hits,updated_at FROM category_examples ORDER BY hits DESC,updated_at DESC", connection)


def save_attachment(transaction_id: int, filename: str, content: bytes) -> None:
    init_storage()
    digest = hashlib.sha256(content).hexdigest()[:12]
    target = ATTACHMENTS_DIR / f"{int(transaction_id)}_{digest}_{safe_name(filename)}"
    target.write_bytes(content)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("INSERT INTO transaction_attachments(transaction_id,filename,path) VALUES(?,?,?)", (int(transaction_id), safe_name(filename), str(target)))


def list_attachments(transaction_id: int) -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT id,filename,created_at FROM transaction_attachments WHERE transaction_id=? ORDER BY id DESC", connection, params=(int(transaction_id),))


def add_transaction(date_value: str, description: str, amount: float, direction: str, category: str, payment_method: str = "Não identificado", due_day: int | None = None, source_file: str = "manual") -> None:
    init_storage()
    row = pd.Series({"date": date_value, "description": description.strip(), "merchant": description.strip().title(), "amount": amount, "direction": direction, "direction_confidence": 1.0, "category": category, "confidence": 1.0, "payment_method": payment_method, "review_status": "Concluído", "transaction_kind": "Lançamento manual", "direction_reason": "Informado manualmente", "due_day": due_day})
    with sqlite3.connect(DB_PATH) as connection:
        inserted = _insert_transaction(connection, _row_payload(row, source_file, ""))
        if inserted:
            _audit(connection, inserted, "lançamento manual", None, {"descrição": description, "valor": amount})


def list_audit_log(transaction_id: int | None = None) -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        query, params = ("SELECT * FROM audit_log WHERE transaction_id=? ORDER BY changed_at DESC", (int(transaction_id),)) if transaction_id else ("SELECT * FROM audit_log ORDER BY changed_at DESC LIMIT 300", ())
        return pd.read_sql_query(query, connection, params=params)


def save_reconciliation(source_hash: str, opening_balance: float, closing_balance: float, notes: str = "") -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""INSERT INTO reconciliations(source_hash,opening_balance,closing_balance,notes) VALUES(?,?,?,?) ON CONFLICT(source_hash) DO UPDATE SET opening_balance=excluded.opening_balance,closing_balance=excluded.closing_balance,notes=excluded.notes,updated_at=CURRENT_TIMESTAMP""", (source_hash, float(opening_balance), float(closing_balance), notes))


def list_reconciliations() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT * FROM reconciliations", connection)


def list_bills(active_only: bool = True) -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        query = "SELECT * FROM bill_schedule" + (" WHERE active=1" if active_only else "") + " ORDER BY due_day, name"
        return pd.read_sql_query(query, connection)


def save_bill(name: str, category: str, amount: float, due_day: int, payment_method: str = "Conta") -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("INSERT INTO bill_schedule(name,category,amount,due_day,payment_method) VALUES(?,?,?,?,?)", (name.strip(), category.strip(), abs(float(amount)), max(1, min(31, int(due_day))), payment_method))


def deactivate_bill(bill_id: int) -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("UPDATE bill_schedule SET active=0 WHERE id=?", (int(bill_id),))


def list_accounts(active_only: bool = True) -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        query = "SELECT * FROM financial_accounts" + (" WHERE active=1" if active_only else "") + " ORDER BY account_type, name"
        return pd.read_sql_query(query, connection)


def save_account(name: str, account_type: str, institution: str, current_balance: float = 0, credit_limit: float = 0, closing_day: int | None = None, due_day: int | None = None, interest_rate: float = 0) -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""INSERT INTO financial_accounts(name,account_type,institution,current_balance,credit_limit,closing_day,due_day,interest_rate)
            VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET account_type=excluded.account_type,institution=excluded.institution,current_balance=excluded.current_balance,credit_limit=excluded.credit_limit,closing_day=excluded.closing_day,due_day=excluded.due_day,interest_rate=excluded.interest_rate,updated_at=CURRENT_TIMESTAMP""", (name.strip(), account_type, institution.strip(), float(current_balance), max(0.0, float(credit_limit)), closing_day, due_day, max(0.0, float(interest_rate))))


def deactivate_account(account_id: int) -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("UPDATE financial_accounts SET active=0 WHERE id=?", (int(account_id),))


def save_ebook(filename: str, content: bytes) -> Path:
    target_dir = ROOT / "ebooks"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / safe_name(filename)
    target.write_bytes(content)
    return target


def list_budgets(month: str = "global") -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT id,month,category,limit_amount,updated_at FROM budgets WHERE month IN (?, 'global') ORDER BY category", connection, params=(month,))


def upsert_budget(category: str, limit_amount: float, month: str = "global") -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""INSERT INTO budgets(month,category,limit_amount) VALUES(?,?,?) ON CONFLICT(month,category) DO UPDATE SET limit_amount=excluded.limit_amount,updated_at=CURRENT_TIMESTAMP""", (month, category.strip(), max(0.0, float(limit_amount))))


def list_goals() -> pd.DataFrame:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT id,name,target_amount,current_amount,deadline,status,created_at,updated_at FROM goals ORDER BY status,deadline", connection)


def save_goal(name: str, target_amount: float, current_amount: float, deadline: str = "") -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("INSERT INTO goals(name,target_amount,current_amount,deadline,status) VALUES(?,?,?,?,?)", (name.strip(), max(0.0, float(target_amount)), max(0.0, float(current_amount)), deadline or None, "Concluída" if current_amount >= target_amount else "Em andamento"))


def update_goal(goal_id: int, current_amount: float, status: str) -> None:
    init_storage()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("UPDATE goals SET current_amount=?,status=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (max(0.0, float(current_amount)), status, int(goal_id)))
