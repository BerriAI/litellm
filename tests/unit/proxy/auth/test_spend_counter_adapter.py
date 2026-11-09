from typing import Final, NoReturn
from unittest.mock import AsyncMock, Mock, call

import pytest

from litellm._auth.spend_counter_policy import Spend
from litellm.proxy.auth.spend_counter_adapter import SpendIO, resolve_spend


def unavailable() -> NoReturn:
    raise RuntimeError("unverified spend")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback, window, floor, strict, expected, calls",
    (
        (0.0, False, None, (False, False), 2.0, (call.read_strict(), call.read_strict())),
        (0.0, False, 4.0, (True,), 4.0, (call.read_strict(), call.read_floor(), call.repair(4.0))),
        (5.0, False, None, (), 5.0, (call.read_floor(),)),
        (0.0, True, 1.0, (), 2.0, (call.read_floor(),)),
    ),
)
async def test_native_plan_preserves_floor_and_flag_read_order(
    fallback: float,
    window: bool,
    floor: float | None,
    strict: tuple[bool, ...],
    expected: float,
    calls: tuple[object, ...],
) -> None:
    effects: Final = Mock()
    effects.attach_mock(Mock(side_effect=strict), "read_strict")
    effects.attach_mock(AsyncMock(return_value=floor), "read_floor")
    effects.attach_mock(AsyncMock(), "repair")
    io: Final = SpendIO(effects.read_strict, effects.read_floor, effects.repair, unavailable)
    result: Final = await resolve_spend(Spend(2.0, False, fallback, 10.0, window, False), io)
    assert (result, tuple(effects.mock_calls)) == (expected, calls)


@pytest.mark.asyncio
async def test_unverified_spend_reads_the_latest_fail_closed_flag() -> None:
    strict: Final = Mock(side_effect=(False, True))
    floor: Final = AsyncMock()
    with pytest.raises(RuntimeError, match="unverified spend"):
        await resolve_spend(
            Spend(2.0, False, 0.0, 10.0, False, False), SpendIO(strict, floor, AsyncMock(), unavailable)
        )
    assert (strict.call_count, floor.await_count) == (2, 0)
