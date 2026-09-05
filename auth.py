"""Autenticação local simples para proteger o histórico financeiro."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUTH_PATH = ROOT / "data" / "user_auth.json"


def _digest(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 180_000).hex()


def has_account() -> bool:
    return AUTH_PATH.exists()


def create_account(username: str, password: str) -> None:
    AUTH_PATH.parent.mkdir(exist_ok=True)
    salt = secrets.token_hex(16)
    payload = {"username": username.strip() or "Matheus", "salt": salt, "password_hash": _digest(password, salt)}
    AUTH_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def authenticate(username: str, password: str) -> bool:
    try:
        payload = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
        expected = _digest(password, payload["salt"])
        return hmac.compare_digest(username.strip().lower(), payload["username"].strip().lower()) and hmac.compare_digest(expected, payload["password_hash"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def account_username() -> str:
    try:
        return str(json.loads(AUTH_PATH.read_text(encoding="utf-8")).get("username", "Matheus"))
    except (OSError, TypeError, json.JSONDecodeError):
        return "Matheus"
