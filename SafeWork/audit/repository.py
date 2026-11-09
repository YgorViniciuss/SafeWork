"""Audit Trail: persistência dos registros de violação em SQLite.

Cada registro guarda o hash do anterior (cadeia de hashes, como num livro-razão).
Alterar ou apagar uma linha quebra a cadeia, e `verify()` aponta onde. Junto com
o SHA-256 da imagem, isso dá rastreabilidade para perícia e auditorias
(ISO 45001 / NR-6 / NR-12).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

from safework.audit.records import DATETIME_FORMAT, GENESIS_HASH, ViolationRecord
from safework.core.config import resolve_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid           TEXT    NOT NULL UNIQUE,
    event_type          TEXT    NOT NULL,
    occurred_at         TEXT    NOT NULL,
    warning_started_at  TEXT,
    camera_id           TEXT    NOT NULL,
    camera_name         TEXT    NOT NULL,
    camera_area         TEXT    NOT NULL,
    camera_source       TEXT    NOT NULL,
    zone_id             TEXT,
    zone_name           TEXT,
    track_id            INTEGER NOT NULL,
    missing_ppe         TEXT    NOT NULL,
    infraction_type     TEXT    NOT NULL,
    elapsed_s           REAL    NOT NULL,
    frame_index         INTEGER NOT NULL,
    evidence_path       TEXT,
    evidence_sha256     TEXT,
    previous_hash       TEXT    NOT NULL,
    record_hash         TEXT    NOT NULL UNIQUE,
    email_status        TEXT    NOT NULL,
    email_sent_at       TEXT,
    email_error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_violations_occurred ON violations (occurred_at);
CREATE INDEX IF NOT EXISTS idx_violations_camera ON violations (camera_id, occurred_at);
"""

_COLUMNS = [item.name for item in fields(ViolationRecord) if item.name != "id"]


class AuditRepository:
    """Grava e consulta a trilha de auditoria. Seguro para várias threads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")  # leitores não bloqueiam a escrita
            connection.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Uma conexão por operação: segura entre a thread do vídeo e a do e-mail.

        Faz commit ao sair sem erro, rollback em exceção e sempre fecha o arquivo.
        """
        connection = sqlite3.connect(self._path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    # ------------------------------------------------------------------ escrita
    def append(self, record: ViolationRecord) -> ViolationRecord:
        """Encadeia o registro ao último e o grava. Devolve-o com id e hashes."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_hash FROM violations ORDER BY id DESC LIMIT 1"
            ).fetchone()
            chained = record.chained(row["record_hash"] if row else GENESIS_HASH)
            values = [getattr(chained, name) for name in _COLUMNS]
            cursor = connection.execute(
                f"INSERT INTO violations ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                values,
            )
            record_id = cursor.lastrowid
        return replace(chained, id=record_id)

    def update_email_status(self, record_id: int, status: str, error: str | None = None) -> None:
        """Atualiza o envio. Esses campos ficam fora do hash de propósito."""
        sent_at = datetime.now().strftime(DATETIME_FORMAT) if status == "enviado" else None
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE violations SET email_status = ?, email_sent_at = ?, email_error = ? "
                "WHERE id = ?",
                (status, sent_at, error, record_id),
            )

    # ------------------------------------------------------------------ leitura
    def get(self, record_id: int) -> ViolationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM violations WHERE id = ?", (record_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def recent(self, limit: int = 50) -> list[ViolationRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM violations").fetchone()[0]

    def verify(self, check_files: bool = True) -> list[str]:
        """Confere a cadeia de hashes e, opcionalmente, o SHA-256 de cada imagem.

        Devolve a lista de problemas encontrados (vazia = trilha íntegra).
        """
        problems: list[str] = []
        expected_previous = GENESIS_HASH
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM violations ORDER BY id").fetchall()

        for row in rows:
            record = self._to_record(row)
            if record.previous_hash != expected_previous:
                problems.append(f"Registro {record.id}: elo quebrado (registro anterior removido "
                                "ou alterado).")
            if record.compute_hash() != record.record_hash:
                problems.append(f"Registro {record.id}: conteúdo alterado após a gravação.")
            expected_previous = record.record_hash

            if check_files and record.evidence_path:
                file_path = resolve_path(record.evidence_path)
                if not file_path.exists():
                    problems.append(f"Registro {record.id}: evidência ausente ({record.evidence_path}).")
                elif hashlib.sha256(file_path.read_bytes()).hexdigest() != record.evidence_sha256:
                    problems.append(f"Registro {record.id}: imagem de evidência modificada.")
        return problems

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ViolationRecord:
        return ViolationRecord(**{key: row[key] for key in row.keys()})
