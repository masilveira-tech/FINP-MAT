"""Backup local cifrado por senha; não envia dados a nenhum serviço externo."""
from __future__ import annotations

import base64
import io
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from storage import ROOT

MAGIC = b"FINPBACK1"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
PENDING_RESTORE_DIRNAME = ".finp_restore_pending"


def _key(password: str, salt: bytes) -> bytes:
    material = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000).derive(password.encode("utf-8"))
    return base64.urlsafe_b64encode(material)


def create_backup(password: str) -> bytes:
    if len(password) < 8:
        raise ValueError("Use uma senha de backup com pelo menos 8 caracteres.")
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in ("data", "extratos", "ebooks"):
            folder = ROOT / relative
            if folder.exists():
                for item in folder.rglob("*"):
                    if item.is_file():
                        archive.write(item, item.relative_to(ROOT))
    salt = os.urandom(16)
    return MAGIC + salt + Fernet(_key(password, salt)).encrypt(raw.getvalue())


def restore_backup(payload: bytes, password: str) -> None:
    """Valida e prepara uma restauração para o próximo início do aplicativo.

    No Windows não é possível substituir ``finp_mat.db`` enquanto o Streamlit
    está executando. Por isso, esta etapa nunca apaga o banco em uso: ela só
    deixa os arquivos prontos em uma pasta temporária fora de ``data``.
    """
    if not payload.startswith(MAGIC) or len(payload) < len(MAGIC) + 17:
        raise ValueError("Arquivo de backup FINP MAT inválido.")
    salt, encrypted = payload[len(MAGIC):len(MAGIC) + 16], payload[len(MAGIC) + 16:]
    try:
        decrypted = Fernet(_key(password, salt)).decrypt(encrypted)
    except Exception as exc:
        raise ValueError("Não foi possível abrir o backup. Confira a senha.") from exc
    with zipfile.ZipFile(io.BytesIO(decrypted)) as archive:
        targets = [Path(name) for name in archive.namelist()]
        if any(path.is_absolute() or ".." in path.parts or path.parts[0] not in {"data", "extratos", "ebooks"} for path in targets):
            raise ValueError("Backup contém caminhos inválidos.")
        staging = ROOT / PENDING_RESTORE_DIRNAME
        if staging.exists():
            shutil.rmtree(staging)
        archive.extractall(staging)


def apply_pending_restore() -> bool:
    """Aplica uma restauração pendente antes do SQLite ser aberto.

    Retorna ``True`` quando uma restauração foi aplicada. A troca usa renomear
    diretórios, mantendo a cópia anterior até que a nova seja posicionada.
    """
    staging = ROOT / PENDING_RESTORE_DIRNAME
    if not staging.exists():
        return False
    targets = ("data", "extratos", "ebooks")
    previous_paths: list[Path] = []
    try:
        for name in targets:
            source = staging / name
            if not source.exists():
                continue
            target = ROOT / name
            previous = ROOT / f".{name}_antes_da_restauracao"
            if previous.exists():
                shutil.rmtree(previous)
            if target.exists():
                target.replace(previous)
                previous_paths.append(previous)
            source.replace(target)
    except OSError as exc:
        # Tenta recuperar a cópia anterior caso uma troca tenha falhado.
        for previous in previous_paths:
            target = ROOT / previous.name.removeprefix(".").removesuffix("_antes_da_restauracao")
            if previous.exists() and not target.exists():
                previous.replace(target)
        raise RuntimeError("Não foi possível aplicar a restauração. Feche todas as janelas do FINP MAT e tente abrir novamente.") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    for previous in previous_paths:
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)
    return True


def create_snapshot(reason: str = "rotina", keep: int = 14) -> Path:
    """Cria um ponto de restauração local antes de importações e correções amplas."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = SNAPSHOTS_DIR / f"finp_{reason}_{stamp}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in ("data/finp_mat.db", "data/user_auth.json", "extratos", "ebooks"):
            source = ROOT / relative
            if source.is_file():
                archive.write(source, source.relative_to(ROOT))
            elif source.exists():
                for item in source.rglob("*"):
                    if item.is_file() and "snapshots" not in item.parts:
                        archive.write(item, item.relative_to(ROOT))
    snapshots = sorted(SNAPSHOTS_DIR.glob("finp_*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old in snapshots[keep:]:
        old.unlink(missing_ok=True)
    return target
