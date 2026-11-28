"""Contract: budgets on routes that take no hold, and budgets while the counter Redis is down.

Routes. The key budget check in the builder runs for every route that runs inference checks
(`RouteChecks.runs_inference_checks`, `litellm/proxy/auth/route_checks.py:474-540`): the OpenAI and Anthropic
routes, `/queue/chat/completions` (#51753) and every mapped pass-through prefix such as `/anthropic`. A hold is
only taken when a priced model can be found for the body (`reserve_budget_for_request`,
`budget_reservation.py:255-291`), so pass-through calls, rerank and moderations can be refused for an exhausted
budget although they never hold one. Unbilled routes (`_UNBILLED_ROUTES`, route_checks.py:107-116) such as
`/utils/token_counter` skip the hold. A Rust front door that classifies routes differently would let an exhausted
key keep spending on one of them.

Outages. With Redis stopped, the admission engine skips the hold and the request is served (fail open, the
default; `engine.rs:316-327`), and `get_current_spend` falls back to the worker's own copy or the database
(`proxy_server.py:3509-3513`), so a key already over budget in Postgres is still refused. Round 1 pinned the
fail closed 503; these pin the default.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.redis_process import OwnedRedis, owned_redis
from tests.integration.spend.dec1_budgets2_support import owned
from tests.integration.spend.dec1_budgets3_support import COST, TOKENS, spend_once
from tests.integration.spend.dec1_budgets_support import call, hold_model, refused
from tests.integration.spend.dec1_spend_support import (
    hashed,
    provider_calls,
    scripted_upstream,
    served,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state


def test_an_exhausted_key_is_refused_on_the_queued_chat_route(gateway: Gateway) -> None:
    """A key with `max_budget: 0` posts to `/queue/chat/completions`. It is refused with HTTP 422
    `budget_exceeded` and the provider is never called: the queued route is one of `_QUEUED_INFERENCE_ROUTES`,
    so it runs the same budget checks as `/v1/chat/completions` (#51753)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = gateway.request(
            "POST",
            "/queue/chat/completions",
            {"model": model, "max_tokens": TOKENS, "messages": [{"role": "user", "content": "dec1 budgets3 queue"}]},
            key=key,
        )
        assert refused(response)
        assert provider_calls(wire) == ()


def test_an_exhausted_key_is_refused_on_an_anthropic_pass_through_route(gateway: Gateway) -> None:
    """A key with `max_budget: 0` posts an Anthropic body to `/anthropic/v1/messages`, a mapped pass-through route.
    It is refused with HTTP 422 before anything is forwarded: `/anthropic` is in `mapped_pass_through_routes`
    (`_types.py:411-449`) so `is_llm_api_route` is true and the key check runs.
    UNVERIFIED: the error body shape on a pass-through route (OpenAI or Anthropic envelope); only the status and
    the `budget_exceeded` marker are asserted."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(max_budget=0)
        response: Final = gateway.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": TOKENS,
                "messages": [{"role": "user", "content": "dec1 budgets3 pass-through"}],
            },
            key=key,
        )
        assert response.status_code == 422, response.text
        assert "budget_exceeded" in response.text, response.text


def test_an_exhausted_key_is_refused_on_moderations_although_moderations_take_no_hold(gateway: Gateway) -> None:
    """A key with `max_budget: 0` calls `/v1/moderations` without a model. No hold could be planned (no model, and
    moderations are an input only route, `_is_input_only_route`), but the key check in the builder still refuses
    it with HTTP 422 `budget_exceeded`."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(max_budget=0)
        response: Final = gateway.request("POST", "/v1/moderations", {"input": "dec1 budgets3 moderation"}, key=key)
        assert refused(response)


def test_an_exhausted_key_is_refused_on_rerank(gateway: Gateway) -> None:
    """A key with `max_budget: 0` calls `/v1/rerank` and is refused with HTTP 422 `budget_exceeded` before any
    provider is called: `/v1/rerank` is an OpenAI route for the budget checks."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = gateway.request(
            "POST", "/v1/rerank", {"model": model, "query": "dec1", "documents": ["a", "b"]}, key=key
        )
        assert refused(response)
        assert provider_calls(wire) == ()


def test_an_exhausted_key_is_refused_on_the_token_counter_route(gateway: Gateway) -> None:
    """A key with `max_budget: 0` calls `POST /utils/token_counter`. The route is listed in `openai_routes`
    (litellm/proxy/_types.py:326), so the key budget check of the auth path runs and the call is refused 422
    `budget_exceeded` with "Current cost: 0.0, Max budget: 0.0". `_UNBILLED_ROUTES`
    (litellm/proxy/auth/route_checks.py:107-116) only spares token counting from the header-only rate limit
    preflight (#51675, user_api_key_auth.py:2308), not from the budget check."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        response: Final = gateway.request(
            "POST",
            "/utils/token_counter",
            {"model": model, "messages": [{"role": "user", "content": "dec1 budgets3 count"}]},
            key=key,
        )
        assert response.status_code == 422, response.text
        error: Final = response.json()["error"]
        assert error["type"] == "budget_exceeded"
        assert error["message"].startswith("Budget has been exceeded! Key=")
        assert error["message"].endswith(" Current cost: 0.0, Max budget: 0.0")


@pytest.fixture
def redis_rig(gateway: Gateway, tmp_path: Path) -> Iterator[tuple[Gateway, OwnedRedis]]:
    """An owned proxy on the suite database whose response cache and spend counters live in an owned Redis that
    the test can stop."""
    with owned_redis(tmp_path / "redis") as cache:
        environment: Final = {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}
        directory: Final = tmp_path / "proxy"
        directory.mkdir()
        with owned(gateway, directory, environment=environment) as proxy:
            yield proxy, cache


def test_with_redis_down_a_budgeted_key_is_served_and_its_spend_still_reaches_the_table(
    redis_rig: tuple[Gateway, OwnedRedis],
) -> None:
    """On a proxy without `fail_closed_budget_enforcement`, the counter Redis is stopped and a key with
    `max_budget` 1.0 makes one call. The call is served (the hold is skipped, `engine.rs:316-327`) and the key's
    stored `spend` still reaches 0.02, so an outage loses the hold but not the bill."""
    proxy, cache = redis_rig
    with scripted_upstream() as wire, proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        cache.stop()
        try:
            assert served(call(proxy, model, key, max_tokens=TOKENS))
            assert spend_reaches(proxy, "key", hashed(key), COST) == pytest.approx(COST)
        finally:
            cache.start()


def test_with_redis_down_a_key_over_budget_in_the_table_is_still_refused(
    redis_rig: tuple[Gateway, OwnedRedis],
) -> None:
    """A key with `max_budget` 0.01 spends 0.02 while Redis is up, and its stored spend reaches 0.02. Redis is then
    stopped. The next call is still refused with HTTP 422 `budget_exceeded`: `get_current_spend` falls back to the
    worker's in-memory counter or the database row (`proxy_server.py:3509-3513`), never to "no spend"."""
    proxy, cache = redis_rig
    with scripted_upstream() as wire, proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.01)
        spend_once(proxy, model, key)
        spend_reaches(proxy, "key", hashed(key), COST)
        cache.stop()
        try:
            assert refused(call(proxy, model, key, max_tokens=TOKENS))
        finally:
            cache.start()
