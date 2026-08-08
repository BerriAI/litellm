from datetime import datetime, timezone

from litellm.models.team import BudgetLimitEntry
from litellm.proxy.common_utils.budget_limits import (
    resolve_budget_limit_windows,
    serialize_budget_limit_windows,
)

EXISTING_RESET_AT = datetime(2027, 1, 1, tzinfo=timezone.utc)


def test_resolve_preserves_reset_at_for_unchanged_duration():
    existing = [
        BudgetLimitEntry(budget_duration="365d", max_budget=100.0, reset_at=EXISTING_RESET_AT),
    ]
    incoming = [BudgetLimitEntry(budget_duration="365d", max_budget=250.0)]

    resolved = resolve_budget_limit_windows(incoming=incoming, existing=existing)

    assert len(resolved) == 1
    assert resolved[0].max_budget == 250.0
    assert resolved[0].reset_at == EXISTING_RESET_AT


def test_resolve_initializes_fresh_reset_at_for_new_duration():
    existing = [
        BudgetLimitEntry(budget_duration="365d", max_budget=100.0, reset_at=EXISTING_RESET_AT),
    ]
    incoming = [
        BudgetLimitEntry(budget_duration="365d", max_budget=100.0),
        BudgetLimitEntry(budget_duration="30d", max_budget=20.0),
    ]

    resolved = {w.budget_duration: w for w in resolve_budget_limit_windows(incoming=incoming, existing=existing)}

    assert resolved["365d"].reset_at == EXISTING_RESET_AT
    assert resolved["30d"].reset_at is not None
    assert resolved["30d"].reset_at != EXISTING_RESET_AT


def test_resolve_on_create_initializes_all_windows():
    incoming = [BudgetLimitEntry(budget_duration="30d", max_budget=20.0)]

    resolved = resolve_budget_limit_windows(incoming=incoming)

    assert resolved[0].reset_at is not None


def test_resolve_ignores_client_supplied_reset_at_on_new_window():
    bogus = datetime(2000, 1, 1, tzinfo=timezone.utc)
    incoming = [BudgetLimitEntry(budget_duration="30d", max_budget=20.0, reset_at=bogus)]

    resolved = resolve_budget_limit_windows(incoming=incoming)

    assert resolved[0].reset_at != bogus


def test_resolve_accepts_dict_windows():
    existing = [{"budget_duration": "365d", "max_budget": 100.0, "reset_at": EXISTING_RESET_AT.isoformat()}]
    incoming = [{"budget_duration": "365d", "max_budget": 250.0}]

    resolved = resolve_budget_limit_windows(incoming=incoming, existing=existing)

    assert resolved[0].max_budget == 250.0
    assert resolved[0].reset_at == EXISTING_RESET_AT


def test_resolve_accepts_existing_json_string():
    """A raw prisma ``Json?`` column can arrive as a JSON string; it must still
    have its reset_at preserved for an unchanged duration."""
    import json

    existing = json.dumps([{"budget_duration": "365d", "max_budget": 100.0, "reset_at": EXISTING_RESET_AT.isoformat()}])
    incoming = [{"budget_duration": "365d", "max_budget": 250.0}]

    resolved = resolve_budget_limit_windows(incoming=incoming, existing=existing)

    assert resolved[0].max_budget == 250.0
    assert resolved[0].reset_at == EXISTING_RESET_AT


def test_serialize_produces_iso_reset_at():
    windows = resolve_budget_limit_windows(
        incoming=[BudgetLimitEntry(budget_duration="365d", max_budget=100.0, reset_at=EXISTING_RESET_AT)],
        existing=[BudgetLimitEntry(budget_duration="365d", max_budget=100.0, reset_at=EXISTING_RESET_AT)],
    )

    import json

    stored = json.loads(serialize_budget_limit_windows(windows))
    assert stored[0]["reset_at"] == EXISTING_RESET_AT.isoformat()
    assert stored[0]["budget_duration"] == "365d"
    assert stored[0]["max_budget"] == 100.0
