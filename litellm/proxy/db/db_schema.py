"""Helpers for reading the configured PostgreSQL schema."""

import os


def get_database_schema() -> str:
    """Return the PostgreSQL schema name from DATABASE_SCHEMA env var.

    Falls back to ``'public'`` when the variable is unset or empty.
    """
    return os.getenv("DATABASE_SCHEMA") or "public"
