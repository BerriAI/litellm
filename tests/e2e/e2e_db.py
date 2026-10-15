"""Shared, destructive DB helpers for the e2e harness.

Kept at the top level next to e2e_config and lifecycle so every suite imports it
by name (`from e2e_db import ...`); no suite reaches into another's directory by
mutating sys.path.

reset_spend_logs truncates LiteLLM_SpendLogs and cannot be undone, so the
session-finish cleanup routes through run_spend_log_cleanup, which fires the
truncate only on an explicit operator opt-in. "An e2e test ran" is necessary but
never sufficient: a DATABASE_URL pointing at a shared or staging instance must
not be wiped by a routine local run that merely exercised a test.
"""

import os
from collections.abc import Callable

RESET_OPT_IN_ENV = "E2E_RESET_SPEND_LOGS"


def run_spend_log_cleanup(
    *, opt_in: str | None, e2e_test_ran: bool, truncate: Callable[[], None]
) -> bool:
    """Invoke `truncate` iff the destructive spend-log reset is both opted into
    and warranted, returning whether the truncate was attempted.

    The truncate fires only when the opt-in value is exactly "1" AND an e2e test
    body actually ran. Any other opt-in value (unset, "0", "true", "") leaves the
    DB untouched, so the destructive path is never armed by the env var's mere
    presence or by a test run on its own. Best-effort: a truncate failure is
    swallowed so cleanup never fails the session, so the returned bool reports
    that the reset was attempted, not that the DB call succeeded.
    """
    if opt_in != "1" or not e2e_test_ran:
        return False
    try:
        truncate()
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        print(f"spend-log cleanup best-effort failed: {exc}")
    return True


def reset_spend_logs() -> None:
    """Truncate LiteLLM_SpendLogs for a clean slate. No proxy endpoint deletes
    spend logs (/global/spend/reset keeps them), so go to the DB directly. Uses
    DATABASE_URL (default: the local docker postgres on its mapped host port; the
    in-container `@db` host isn't resolvable from the host, so default to
    localhost).
    """
    import psycopg

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm",
    )
    with psycopg.connect(url) as conn:
        _ = conn.execute('TRUNCATE TABLE "LiteLLM_SpendLogs"')
