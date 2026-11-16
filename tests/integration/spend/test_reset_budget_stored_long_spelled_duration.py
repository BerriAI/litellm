import os
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy

pytestmark: Final = pytest.mark.shares_worker_state

FAST_RESET_TICK: Final = MappingProxyType(
    {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "2"}
)


def _store_due_budget(key: str, budget_duration: str, spend: float) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            'UPDATE "LiteLLM_VerificationToken" SET spend = %s, budget_duration = %s, '
            "budget_reset_at = now() - interval '1 day' WHERE token = %s",
            (spend, budget_duration, sha256(key.encode()).hexdigest()),
        )


def _key_budget(key: str) -> dict[str, object]:
    rows: Final = read_rows(
        'SELECT spend, budget_reset_at > now() AS reset_is_ahead FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )
    assert len(rows) == 1, rows
    return dict(rows[0])


@pytest.mark.parametrize("budget_duration", ("99999999999days", "999999999weeks"))
def test_a_key_stored_with_a_huge_long_spelled_budget_duration_still_resets(
    gateway: Gateway, tmp_path: Path, budget_duration: str
) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(max_budget=10.0, budget_duration="1d")
        _store_due_budget(key, budget_duration, spend=0.5)
        with owned_proxy(gateway, tmp_path, FAST_RESET_TICK):
            swept: Final = eventually(
                lambda: _key_budget(key),
                lambda budget: budget["spend"] == 0.0,
                seconds=30,
                return_last_on_timeout=True,
            )
        assert swept == {"spend": 0.0, "reset_is_ahead": True}, (
            f"key stored with budget_duration={budget_duration!r} was not reset: {swept}"
        )
