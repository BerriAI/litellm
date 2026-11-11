from types import MappingProxyType
from typing import Final

import pytest
from fakeredis import FakeRedis, FakeServer

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.record_scope import (
    Caller,
    RecordScope,
    returned_unchanged,
    session_claim_store_of,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GuardrailInformationScope,
    GuardrailToolParam,
)
from litellm.types.utils import GenericGuardrailAPIInputs

_USER_ROW: Final = MappingProxyType({"role": "user", "content": "hello"})
_CALLER: Final = Caller(key_hash="hash-a", team_id=None, user_id=None)
_TOOL: Final = MappingProxyType(
    {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "required": ["q"]}}}
)


def _chat_request(**extra: object) -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(texts=["hello"], structured_messages=[dict(_USER_ROW)], model="gpt-test", **extra)


@pytest.mark.parametrize(
    ("returned_tools", "unchanged"),
    [
        ([GuardrailToolParam.model_validate(dict(_TOOL))], True),
        ([GuardrailToolParam.model_validate({"type": "function", "function": {"name": "other"}})], False),
    ],
)
def test_tools_returned_as_models_compare_by_content(returned_tools: list[GuardrailToolParam], unchanged: bool) -> None:
    tool_with_a_tuple: Final = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "required": ("q",)}},
    }
    sent: Final = _chat_request(tools=[tool_with_a_tuple])
    returned: Final = GenericGuardrailAPIInputs(texts=["hello"], tools=returned_tools)

    assert returned_unchanged(sent, returned) is unchanged


def _scope(
    scope: GuardrailInformationScope = "per_session",
    *,
    redis: RedisCache | None = None,
    sessions: InMemoryCache | None = None,
    guardrail_name: str = "scoped-guardrail",
) -> RecordScope:
    return RecordScope(scope, guardrail_name=guardrail_name, shared_store=lambda: redis, recorded_sessions=sessions)


async def _records(record_scope: RecordScope, session_id: str = "s1") -> bool:
    return await record_scope.should_record_allow(session_id=session_id, caller=_CALLER, input_type="request")


@pytest.mark.asyncio
async def test_per_session_records_again_once_the_session_has_expired() -> None:
    now: Final = [0.0]  # mutable-ok: fake clock advanced by the test
    record_scope: Final = _scope(sessions=InMemoryCache(default_ttl=60, clock=lambda: now[0]))

    first, within_ttl = await _records(record_scope), await _records(record_scope)
    now[0] = 61.0

    assert (first, within_ttl, await _records(record_scope)) == (True, False, True)


@pytest.mark.asyncio
async def test_a_claim_whose_redis_reply_was_lost_still_records_the_session_once(
    redis_losing_set_replies: tuple[RedisCache, FakeServer],
) -> None:
    redis, _ = redis_losing_set_replies
    pods: Final = (_scope(redis=redis), _scope(redis=redis))

    assert [await _records(pod) for pod in (*pods, *pods)] == [True, False, False, False]


@pytest.mark.asyncio
async def test_the_proxy_shares_its_redis_for_session_claims(shared_redis: tuple[RedisCache, FakeServer]) -> None:
    redis, _ = shared_redis
    proxy_logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    proxy_logging.update_values(redis_cache=redis)

    assert session_claim_store_of(proxy_logging) is redis


@pytest.mark.asyncio
async def test_a_session_claim_in_redis_lasts_the_hour_long_session_window(
    shared_redis: tuple[RedisCache, FakeServer],
) -> None:
    redis, server = shared_redis
    for session_id in ("s1", "s2"):
        await _records(_scope(redis=redis), session_id)
    client: Final = FakeRedis(server=server)

    assert [3590 <= client.ttl(key) <= 3600 for key in client.keys()] == [True, True]


@pytest.mark.asyncio
async def test_a_redis_outage_records_each_pods_first_allow_once(shared_redis: tuple[RedisCache, FakeServer]) -> None:
    redis, server = shared_redis
    server.connected = False
    first_pod: Final = _scope(redis=redis)
    second_pod: Final = _scope(redis=redis)

    decisions: Final = [await _records(pod) for pod in (first_pod, second_pod, first_pod, second_pod)]

    assert decisions == [True, True, False, False]


@pytest.mark.asyncio
async def test_an_open_redis_circuit_breaker_still_records_each_sessions_first_allow(
    shared_redis: tuple[RedisCache, FakeServer],
) -> None:
    redis, server = shared_redis
    server.connected = False
    record_scope: Final = _scope(redis=redis)
    sessions: Final = [f"s{index}" for index in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD + 1)]

    assert [await _records(record_scope, session_id) for session_id in sessions] == [True] * len(sessions)


def test_undecodable_bytes_compare_without_raising() -> None:
    rows: Final = [{"role": "user", "content": b"\xff\xfe"}]

    assert returned_unchanged(
        GenericGuardrailAPIInputs(texts=["hello"], structured_messages=rows),
        GenericGuardrailAPIInputs(texts=["hello"], structured_messages=rows),
    )


class _Opaque:
    pass


def test_a_non_json_value_passed_through_compares_as_unchanged() -> None:
    passed_through: Final = _Opaque()

    assert returned_unchanged(
        _chat_request(tools=[passed_through]), GenericGuardrailAPIInputs(texts=["hello"], tools=[passed_through])
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("scope", "recorded"), [("per_call", True), ("off", False)])
async def test_per_call_and_off_decide_without_claiming_a_session(
    scope: GuardrailInformationScope, recorded: bool, shared_redis: tuple[RedisCache, FakeServer]
) -> None:
    redis, _ = shared_redis
    sessions: Final = InMemoryCache()
    record_scope: Final = _scope(scope, redis=redis, sessions=sessions)

    decisions: Final = [await _records(record_scope) for _ in range(2)]
    session_still_unclaimed: Final = await _records(_scope(redis=redis, sessions=sessions))

    assert (decisions, session_still_unclaimed) == ([recorded, recorded], True)


@pytest.mark.asyncio
async def test_a_huge_session_id_is_stored_under_a_fixed_size_key() -> None:
    short_id_sessions: Final = InMemoryCache()
    huge_id_sessions: Final = InMemoryCache()
    huge_session_id: Final = "s" * 1_000_000

    decisions: Final = [
        await _records(_scope(sessions=huge_id_sessions), huge_session_id),
        await _records(_scope(sessions=huge_id_sessions), huge_session_id),
        await _records(_scope(sessions=short_id_sessions), "s1"),
    ]

    assert (decisions, [len(key) for key in huge_id_sessions.cache_dict]) == (
        [True, False, True],
        [len(key) for key in short_id_sessions.cache_dict],
    ), "a caller-chosen session id must not grow the dedup key"
