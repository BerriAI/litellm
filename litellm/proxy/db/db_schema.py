"""Helpers for reading the configured PostgreSQL schema."""

import os
import re

# Only allow characters that are safe in an unquoted SQL identifier.
_SAFE_SCHEMA_RE = re.compile(r"^[A-Za-z0-9_]+$")


def get_database_schema() -> str:
    """Return the PostgreSQL schema name from DATABASE_SCHEMA env var.

    Falls back to ``'public'`` when the variable is unset, empty, or whitespace-only.

    Raises ``ValueError`` if the value contains characters outside
    ``[A-Za-z0-9_]`` to prevent SQL-injection via the env var.
    """
    schema = os.getenv("DATABASE_SCHEMA", "").strip() or "public"
    if not _SAFE_SCHEMA_RE.match(schema):
        raise ValueError(
            f"Invalid DATABASE_SCHEMA {schema!r}: only alphanumeric characters and "
            "underscores are allowed."
        )
    return schema
