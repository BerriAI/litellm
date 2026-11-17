import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.vector_stores.fallbacks import (
    Failed,
    Served,
    VectorStoreFallbackPlan,
    VectorStoreTarget,
    classify_error,
    classify_status,
    execute_with_vector_store_fallbacks,
    fallback_ids,
    served_headers,
    targets_for_store,
)


class Clock:
    def __init__(self) -> None:
        self.now: float = 100.0

    def __call__(self) -> float:
        return self.now


def target(store_id: str) -> VectorStoreTarget:
    return VectorStoreTarget(store_id, "openai", MappingProxyType({"api_base": f"https://{store_id}.example"}))


@pytest.mark.asyncio
async def test_failure_skips_primary_on_next_request_and_expiry_restores_it():
    clock: Final = Clock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    primary: Final = target("primary")
    backup: Final = target("backup")
    plan: Final = VectorStoreFallbackPlan((primary, backup), cooldown_time=0.25)
    attempts: Final[list[str]] = []

    async def attempt(candidate: VectorStoreTarget):
        attempts.append(candidate.store_id)
        if candidate == primary and clock.now < 100.25:
            return Failed(candidate, ConnectionError("unreachable"), True)
        return Served(candidate, "results")

    first: Final = await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    second: Final = await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    clock.now = 100.26
    recovered: Final = await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    assert attempts == ["primary", "backup", "backup", "primary"]
    assert first == second == Served(backup, "results", 1)
    assert recovered == Served(primary, "results", 0)


@pytest.mark.asyncio
async def test_all_cooled_returns_503_without_another_attempt():
    clock: Final = Clock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    plan: Final = VectorStoreFallbackPlan((target("primary"), target("backup")), cooldown_time=5)
    attempts: Final[list[str]] = []

    async def attempt(candidate: VectorStoreTarget):
        attempts.append(candidate.store_id)
        return Failed(candidate, httpx.ConnectError("unavailable"), True)

    await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    outcome: Final = await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    assert attempts == ["primary", "backup"]
    assert isinstance(outcome, Failed)
    assert isinstance(outcome.error, litellm.ServiceUnavailableError)
    assert outcome.error.response.status_code == 503
    assert outcome.error.response.headers["retry-after"] == "5"


@pytest.mark.asyncio
async def test_terminal_failure_is_not_retried_or_cooled():
    clock: Final = Clock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    plan: Final = VectorStoreFallbackPlan((target("primary"), target("backup")))
    attempts: Final[list[str]] = []
    error: Final = ValueError("bad query")

    async def attempt(candidate: VectorStoreTarget):
        attempts.append(candidate.store_id)
        return Failed(candidate, error, False)

    first: Final = await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    second: Final = await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    assert attempts == ["primary", "primary"]
    assert first == second == Failed(plan.targets[0], error, False)


@pytest.mark.parametrize(
    "status,expected", [(408, True), (429, True), (503, True), (400, False), (401, False), (404, False), (409, False)]
)
def test_status_classification_preserves_caller_errors(status: int, expected: bool):
    assert classify_status(status) is expected


@pytest.mark.parametrize("value", [None, "backup", [1], [""], ["backup", "backup"]])
def test_invalid_fallback_list_is_rejected(value: object):
    with pytest.raises((ValueError, ValidationError)):
        fallback_ids({"fallbacks": value})


def store(store_id: str, provider: str, refs: list[str] | None = None) -> Mapping[str, object]:
    return {"vector_store_id": store_id, "custom_llm_provider": provider, "litellm_params": {"fallbacks": refs or []}}


def test_cross_provider_targets_keep_their_own_ids_and_reject_nested_references():
    primary: Final = store("first", "openai", ["second"])
    backup: Final = store("second", "bedrock")
    targets: Final = targets_for_store(primary, (primary, backup))
    assert tuple((item.store_id, item.provider) for item in targets) == (("first", "openai"), ("second", "bedrock"))
    assert all("fallbacks" not in item.params for item in targets)
    with pytest.raises(ValueError, match="Nested"):
        targets_for_store(primary, (primary, store("second", "bedrock", ["third"])))
    with pytest.raises(ValueError, match="exactly one"):
        targets_for_store(primary, (primary,))


def test_headers_remove_url_credentials_and_report_actual_store():
    candidate: Final = VectorStoreTarget(
        "backup", "openai", {"api_base": "https://user:secret@host.example/v1?key=secret#secret"}
    )
    assert served_headers(Served(candidate, "ok", 1)) == {
        "x-litellm-vector-store-id": "backup",
        "x-litellm-attempted-fallbacks": "1",
        "x-litellm-model-api-base": "https://host.example/v1",
    }


def test_only_transport_and_backend_status_errors_are_retryable():
    candidate: Final = target("primary")
    assert classify_error(candidate, httpx.ConnectError("connection failed")).retryable
    assert not classify_error(candidate, ValueError("invalid query")).retryable


@pytest.mark.asyncio
async def test_cancellation_does_not_send_fallback_or_cool_primary():
    clock: Final = Clock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    primary: Final = target("cancelled-primary")
    plan: Final = VectorStoreFallbackPlan((primary, target("cancelled-backup")))
    cancelled_attempts: Final[list[str]] = []
    resumed_attempts: Final[list[str]] = []

    async def cancel(candidate: VectorStoreTarget):
        cancelled_attempts.append(candidate.store_id)
        raise asyncio.CancelledError

    async def resume(candidate: VectorStoreTarget):
        resumed_attempts.append(candidate.store_id)
        return Served(candidate, "results")

    with pytest.raises(asyncio.CancelledError):
        await execute_with_vector_store_fallbacks(plan, cancel, cache=cache, clock=clock)
    outcome: Final = await execute_with_vector_store_fallbacks(plan, resume, cache=cache, clock=clock)
    assert cancelled_attempts == resumed_attempts == [primary.store_id]
    assert outcome == Served(primary, "results")


@pytest.mark.asyncio
async def test_inherited_deadline_prevents_fallback_after_primary_exhausts_budget():
    from litellm.litellm_core_utils.total_deadline import total_deadline

    clock: Final = Clock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    primary: Final = target("deadline-primary")
    plan: Final = VectorStoreFallbackPlan((primary, target("deadline-backup")))
    attempts: Final[list[str]] = []

    async def attempt(candidate: VectorStoreTarget):
        attempts.append(candidate.store_id)
        clock.now += 61
        return Failed(candidate, httpx.ConnectError("unavailable"), True)

    async with total_deadline(60, from_body=False, clock=clock):
        outcome: Final = await execute_with_vector_store_fallbacks(plan, attempt, cache=cache, clock=clock)
    assert attempts == [primary.store_id]
    assert isinstance(outcome, Failed)
    assert isinstance(outcome.error, litellm.Timeout)
    assert not outcome.retryable


def test_terminal_http_status_takes_precedence_over_chained_transport_error():
    primary: Final = target("bad-request-primary")
    error: Final = litellm.BadRequestError(
        message="invalid query", model=primary.store_id, llm_provider=primary.provider
    )
    error.__cause__ = httpx.ConnectError("earlier transport error")
    outcome: Final = classify_error(primary, error)
    assert outcome == Failed(primary, error, False)
