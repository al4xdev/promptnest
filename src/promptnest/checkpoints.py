"""Durable stage checkpoints for resumable PromptNest runs."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import suppress
from pathlib import Path
from typing import Protocol, runtime_checkable

from promptnest.exceptions import ConfigurationError


@runtime_checkable
class CheckpointStore(Protocol):
    """Asynchronous checkpoint storage contract."""

    async def prepare(self, run_id: str, run_revision: str) -> None: ...

    async def load(
        self,
        run_id: str,
        job_id: str,
        stage: str,
        fragment_index: int | None = None,
    ) -> str | None: ...

    async def save(
        self,
        run_id: str,
        job_id: str,
        stage: str,
        payload: str,
        *,
        fragment_index: int | None = None,
        provider: str | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


class SQLiteCheckpointStore:
    """SQLite WAL checkpoint store with serialized asynchronous access."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def _ensure_connection(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_revision TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                fragment_index INTEGER NOT NULL DEFAULT -1,
                provider TEXT,
                payload TEXT NOT NULL,
                checksum TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, job_id, stage, fragment_index),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            """,
        )
        connection.commit()
        with suppress(OSError):
            os.chmod(self.path, 0o600)
        self._connection = connection
        return connection

    async def prepare(self, run_id: str, run_revision: str) -> None:
        if not run_id or not run_revision:
            raise ConfigurationError("run_id and run_revision must be non-empty")
        async with self._lock:
            connection = await self._ensure_connection()
            cursor = connection.execute(
                "SELECT run_revision FROM runs WHERE run_id = ?",
                (run_id,),
            )
            row = cursor.fetchone()
            if row is not None and row[0] != run_revision:
                raise ConfigurationError(
                    f"checkpoint run {run_id!r} has revision {row[0]!r}, not {run_revision!r}"
                )
            connection.execute(
                "INSERT OR IGNORE INTO runs(run_id, run_revision) VALUES (?, ?)",
                (run_id, run_revision),
            )
            connection.commit()

    async def load(
        self,
        run_id: str,
        job_id: str,
        stage: str,
        fragment_index: int | None = None,
    ) -> str | None:
        async with self._lock:
            connection = await self._ensure_connection()
            cursor = connection.execute(
                """
                SELECT payload, checksum
                FROM checkpoints
                WHERE run_id = ? AND job_id = ? AND stage = ? AND fragment_index = ?
                """,
                (run_id, job_id, stage, fragment_index if fragment_index is not None else -1),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        payload, expected_checksum = row
        actual_checksum = _checksum(payload)
        if actual_checksum != expected_checksum:
            raise ConfigurationError(f"checkpoint checksum mismatch for {job_id!r}/{stage}")
        return str(payload)

    async def save(
        self,
        run_id: str,
        job_id: str,
        stage: str,
        payload: str,
        *,
        fragment_index: int | None = None,
        provider: str | None = None,
    ) -> None:
        json.loads(payload)
        async with self._lock:
            connection = await self._ensure_connection()
            connection.execute(
                """
                INSERT INTO checkpoints(
                    run_id, job_id, stage, fragment_index, provider, payload, checksum
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, job_id, stage, fragment_index)
                DO UPDATE SET
                    provider = excluded.provider,
                    payload = excluded.payload,
                    checksum = excluded.checksum,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    run_id,
                    job_id,
                    stage,
                    fragment_index if fragment_index is not None else -1,
                    provider,
                    payload,
                    _checksum(payload),
                ),
            )
            connection.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None


def canonical_job_id(key: object) -> str:
    """Serialize primitive keys deterministically for checkpoints."""
    try:
        return json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "checkpoint keys must be JSON serializable; provide a primitive key"
        ) from exc


def _checksum(payload: str) -> str:
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
