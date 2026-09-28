from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.types.agents import agent_budget_counter_key, agent_spend_filter


@pytest.mark.parametrize("offset", [None, timezone.utc, timezone(timedelta(hours=5, minutes=30))])
def test_agent_window_key_round_trip_preserves_the_admitted_instant(offset) -> None:
    instant: Final = datetime(2030, 1, 1, 12, 30, 1, 123000, tzinfo=offset)
    expected: Final = instant.replace(tzinfo=timezone.utc) if offset is None else instant.astimezone(timezone.utc)
    key: Final = agent_budget_counter_key("agent:with:colons", instant)
    assert agent_spend_filter(key) == {"agent_id": "agent:with:colons", "spend_window": expected}
    assert key == agent_budget_counter_key("agent:with:colons", expected)


def test_unbudgeted_key_is_filtered_to_an_unbudgeted_row() -> None:
    key: Final = agent_budget_counter_key("agent-one", None)
    assert agent_spend_filter(key) == {"agent_id": "agent-one", "spend_window": None}


def test_different_budget_windows_never_share_a_settlement_filter() -> None:
    first: Final = datetime(2030, 1, 1, tzinfo=timezone.utc)
    second: Final = first + timedelta(days=1)
    assert agent_spend_filter(agent_budget_counter_key("agent-one", first)) != agent_spend_filter(
        agent_budget_counter_key("agent-one", second)
    )
