"""Vote storage: PostgreSQL when DATABASE_URL is set, else local SQLite for development."""

from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

OPTION_1 = "Option 1"
OPTION_2 = "Option 2"
DEFAULT_OPTIONS = (OPTION_1, OPTION_2)


class VoteStore(ABC):
    @abstractmethod
    def ensure_schema(self) -> None:
        pass

    @abstractmethod
    def get_count(self, option_name: str) -> int:
        pass

    @abstractmethod
    def increment(self, option_name: str) -> None:
        pass


class SqliteVoteStore(VoteStore):
    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS votes (
                    option_name TEXT PRIMARY KEY,
                    vote_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            for opt in DEFAULT_OPTIONS:
                conn.execute(
                    "INSERT OR IGNORE INTO votes (option_name, vote_count) VALUES (?, 0)",
                    (opt,),
                )

    def get_count(self, option_name: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT vote_count FROM votes WHERE option_name = ?", (option_name,)
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def increment(self, option_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO votes (option_name, vote_count) VALUES (?, 1)
                ON CONFLICT(option_name) DO UPDATE SET
                    vote_count = vote_count + 1
                """,
                (option_name,),
            )


class PostgresVoteStore(VoteStore):
    def __init__(self, dsn: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn

    def ensure_schema(self) -> None:
        with self._psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS votes (
                    option_name TEXT PRIMARY KEY,
                    vote_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            for opt in DEFAULT_OPTIONS:
                conn.execute(
                    "INSERT INTO votes (option_name, vote_count) VALUES (%s, 0) "
                    "ON CONFLICT (option_name) DO NOTHING",
                    (opt,),
                )
            conn.commit()

    def get_count(self, option_name: str) -> int:
        with self._psycopg.connect(self._dsn) as conn:
            cur = conn.execute(
                "SELECT vote_count FROM votes WHERE option_name = %s", (option_name,)
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def increment(self, option_name: str) -> None:
        with self._psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                INSERT INTO votes (option_name, vote_count) VALUES (%s, 1)
                ON CONFLICT (option_name) DO UPDATE SET
                    vote_count = votes.vote_count + 1
                """,
                (option_name,),
            )
            conn.commit()


def resolve_database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])
    except Exception:
        pass
    return None


def get_vote_store(repo_root: Path | None = None) -> VoteStore:
    """Prefer Postgres when DATABASE_URL is available (env or Streamlit secrets)."""
    url = resolve_database_url()
    if url:
        return PostgresVoteStore(url)
    root = repo_root or Path(__file__).resolve().parents[1]
    return SqliteVoteStore(root / "votes.db")
