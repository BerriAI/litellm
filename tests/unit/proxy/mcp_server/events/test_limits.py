from typing import Final

import pytest
from mcp.shared.exceptions import MCPError

from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.limits import EventBudget, StreamSlots


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_event_budget_is_per_caller_per_minute() -> None:
    clock: Final = _Clock()
    budget: Final = EventBudget(clock)
    assert [budget.take("a", 2) for _ in range(3)] == [True, True, False]
    assert budget.take("b", 2) is True
    clock.now += 59.9
    assert budget.take("a", 2) is False
    clock.now += 0.2
    assert budget.take("a", 2) is True


@pytest.mark.parametrize(
    ("per_caller", "total", "second_caller", "limit_max"),
    ((1, 10, "a", 1), (5, 1, "b", 1)),
)
@pytest.mark.asyncio
async def test_stream_slots_cap_per_caller_and_per_replica(
    per_caller: int, total: int, second_caller: str, limit_max: int
) -> None:
    slots: Final = StreamSlots()
    settings: Final = EventsSettings(max_streams_per_caller=per_caller, max_streams=total)
    async with slots.hold("a", settings):
        with pytest.raises(MCPError) as rejected:
            async with slots.hold(second_caller, settings):
                pass
        assert (rejected.value.error.code, rejected.value.error.data) == (
            -32025,
            {"limit": "streams", "max": limit_max},
        )
    assert slots.live("a") == 0
    async with slots.hold(second_caller, settings):
        assert slots.live(second_caller) == 1
