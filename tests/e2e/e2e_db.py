"""Shared e2e database helpers usable across suites without sys.path hacks.

Lives alongside e2e_config.py / lifecycle.py so any suite (or the top-level
conftest cleanup) can import it normally with `from e2e_db import ...`.
"""

from __future__ import annotations

import os


def reset_spend_logs() -> None:
    """Truncate LiteLLM_SpendLogs for a clean slate. No proxy endpoint deletes
    spend logs (/global/spend/reset keeps them), so go to the DB directly. Uses
    DATABASE_URL (default: the local docker postgres on its mapped host port; note
    the in-container `@db` host isn't resolvable from the host, so default to
    localhost).
    """
    import psycopg

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm",
    )
    with psycopg.connect(url) as conn:
        _ = conn.execute('TRUNCATE TABLE "LiteLLM_SpendLogs"')
