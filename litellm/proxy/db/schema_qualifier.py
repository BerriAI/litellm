"""Schema qualifier for unqualified raw SQL run via Prisma's ``query_raw``.

When LiteLLM's Postgres tables live in a non-default schema (e.g. operators
deploy with ``DATABASE_SCHEMA=litellm`` or ``DATABASE_URL=…?schema=litellm``),
Prisma's model API honours the schema automatically. Its ``query_raw`` /
``execute_raw`` paths do not: the raw SQL strings throughout this codebase
reference table names unqualified (``FROM "LiteLLM_VerificationToken" v``),
so the lookup resolves against ``search_path`` at query time.

Under transaction-pooled connections (PgBouncer transaction mode, Neon's
``-pooler.*`` hostnames, anything matching prisma/prisma#7975) the pool
resets ``search_path`` to the database default on every checkout, ignoring
the per-session ``SET search_path`` Prisma emits at session start. The
unqualified table reference then resolves to ``public`` (or wherever the
default lands), the row isn't found, and virtual-key auth on
``/chat/completions`` fails with ``relation "LiteLLM_VerificationToken"
does not exist``.

``qualify(sql)`` rewrites the raw SQL to prepend the configured schema to
every ``"LiteLLM_*"`` identifier, so the query is unambiguous regardless of
``search_path``. When no schema is configured (the default for installations
that leave tables in ``public``) the function is a no-op — existing
deployments see no behaviour change.

See BerriAI/litellm#29093 for the underlying bug report and reproducer.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional
from urllib.parse import parse_qs, urlparse

# Match every double-quoted identifier starting with "LiteLLM_". This is the
# convention every Prisma-managed table follows (see schema.prisma), so an
# allow-list isn't required — any future "LiteLLM_..." table picks up the
# qualifier automatically.
_TABLE_REF_RE = re.compile(r'"(LiteLLM_\w+)"')


@lru_cache(maxsize=1)
def get_schema() -> Optional[str]:
    """Return the configured Postgres schema for LiteLLM tables, or ``None``
    when tables live in the database default (typically ``public``).

    Resolution order:
      1. ``DATABASE_SCHEMA`` env var (the canonical setting; the same name
         the Helm chart and ``DatabaseURLSettings`` already use).
      2. ``?schema=…`` query parameter on ``DATABASE_URL``.

    Returns ``None`` when either nothing is configured or the configured
    schema is exactly ``public`` (the database default — no qualifier needed
    and stripping it keeps the rewrite a no-op on stock deployments).
    """
    schema = os.environ.get("DATABASE_SCHEMA", "").strip()
    if not schema:
        url = os.environ.get("DATABASE_URL", "")
        if "?" in url:
            try:
                qs = parse_qs(urlparse(url).query)
                schema = (qs.get("schema") or [""])[0].strip()
            except Exception:
                schema = ""
    if not schema or schema == "public":
        return None
    return schema


def reset_cache() -> None:
    """Drop the cached schema resolution. Tests that mutate ``os.environ``
    between cases should call this; production code never needs to."""
    get_schema.cache_clear()


def qualify(sql: str) -> str:
    """Rewrite ``sql`` so every ``"LiteLLM_*"`` identifier is prefixed with
    the configured schema (e.g. ``"litellm"."LiteLLM_VerificationToken"``).

    When no schema is configured the input is returned unchanged — there is
    no perf cost beyond one cached env lookup and a regex no-match scan.

    Idempotent: applying the rewrite twice produces the same output as
    once. (Already-qualified identifiers like ``"litellm"."LiteLLM_..."``
    don't match the regex because the regex anchors on the opening quote
    of the identifier.)
    """
    schema = get_schema()
    if not schema:
        return sql
    prefix = f'"{schema}".'
    return _TABLE_REF_RE.sub(lambda m: f'{prefix}"{m.group(1)}"', sql)
