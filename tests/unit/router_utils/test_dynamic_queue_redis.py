import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Final

import pytest
from redis.crc import key_slot
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm._logging import verbose_router_logger
from litellm.router_utils.dynamic_queue import CoordinationUnavailable, DynamicQueue, Waiting, WaitRequest
from litellm.router_utils.dynamic_queue_redis import REDIS_UNAVAILABLE, RedisQueueBackend, _keys, rank_key
from tests.unit.router_utils.test_dynamic_queue import ManualTimer


def test_rank_keys_sort_byte_wise_in_numeric_order_for_any_integer() -> None:
    priorities: Final = (7, -(10**40), 0, 1001, -1, 10**40, -999, 9, -1001, 10, -10, 999, 1, -9, 10**40 + 1)

    assert sorted(priorities, key=lambda priority: rank_key(priority).encode()) == sorted(priorities)


@pytest.mark.parametrize("capacity", ["deployment-1", "a}b", "}starts-with-brace", "{already-tagged}", ""])
def test_every_key_of_a_capacity_lands_on_one_cluster_slot(capacity: str) -> None:
    slots: Final = {key_slot(key.encode()) for key in _keys(capacity)}

    assert len(slots) == 1


class UnreachableRedis:
    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]:
        async def run(keys: Sequence[str], args: Sequence[str | int]) -> object:
            raise RedisConnectionError("Error connecting to redis://queue-user:hunter2@cache.internal:6379")

        return run


async def test_an_unreachable_redis_rejects_the_wait_with_a_stable_reason_instead_of_admitting_locally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    queue: Final = DynamicQueue(RedisQueueBackend(UnreachableRedis()), ManualTimer())

    with caplog.at_level(logging.DEBUG, logger=verbose_router_logger.name):
        result: Final = await queue.admit(WaitRequest(capacity="deployment-1", waiter_id="a", priority=0), 30.0)

    logged: Final = [record.getMessage() for record in caplog.records if "coordination failed" in record.getMessage()]
    assert isinstance(result, CoordinationUnavailable)
    assert result.reason == REDIS_UNAVAILABLE
    assert "hunter2" not in repr(result)
    assert logged == [
        "dynamic queue: coordination failed (redis_unavailable) after 0.000s waiting, going on without the queue"
    ]


class AnswersPolls:
    def __init__(self, reply: object) -> None:
        self._reply = reply

    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]:
        async def run(keys: Sequence[str], args: Sequence[str | int]) -> object:
            return self._reply

        return run


@pytest.mark.parametrize(
    ("reply", "ahead", "probe_in_ms"),
    [([b"wait"], None, None), ([b"wait", 3], 3, None), ([b"wait", b"probe", 20_000], None, 20_000)],
    ids=["plain", "draining_ahead", "probe_far_ahead"],
)
async def test_a_wait_reply_is_read_with_or_without_its_probe_hint(
    reply: list[bytes | int], ahead: int | None, probe_in_ms: int | None
) -> None:
    polled: Final = await RedisQueueBackend(AnswersPolls(reply)).poll("deployment-1", "a")

    assert isinstance(polled, Waiting)
    assert (polled.ahead, polled.probe_in_ms) == (ahead, probe_in_ms)
