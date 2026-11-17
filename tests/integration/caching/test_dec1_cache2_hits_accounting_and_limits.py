"""What a response cache hit costs and counts: spend rows on every route, cost headers, budgets and rate limits.

A hit is still a full proxy request. Auth, budget checks and the rate limiter's pre-call hook
(``parallel_request_limiter_v3``) run before the cache is consulted, so a hit is refused when the caller is
over a budget or a limit, and it uses up one request of every rpm window it falls in. The limiter has no
special case for hits (no ``cache_hit`` check in ``litellm/proxy/hooks/parallel_request_limiter_v3.py``), so the
stored usage of a hit is also charged to token windows. The spend side is free: the hit writes its own
``LiteLLM_SpendLogs`` row with ``cache_hit = 'True'`` and ``spend = 0`` and the response carries
``x-litellm-response-cost: 0``.

The chat-only spend cells live in ``test_dec1_cache_spend_and_callbacks.py``. This file adds the other routes
(where the Rust messages path and the Rust spend writer must keep the same row) and the limiter and budget
behaviour that the Rust ports of budgets and rate limits (Nov 10-24) are rewriting.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.caching.dec1_cache2_support import (
    CALL_TYPES,
    CHAT_ROUTE,
    EMBEDDINGS_ROUTE,
    NON_CHAT_ROUTES,
    PRICED,
    route_probe,
)
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    COST_HEADER,
    SPEND_SECONDS,
    await_named_entry,
    await_spend_rows,
    error_message,
    hashed,
)
from tests.integration.caching.response_cache_case import prompt, redis_store

pytestmark: Final = pytest.mark.shares_worker_state

ROUTE_PARAMS: Final = tuple(pytest.param(route, id=route) for route in NON_CHAT_ROUTES)
HEADER_ROUTE_PARAMS: Final = tuple(
    pytest.param(route, id=route) for route in NON_CHAT_ROUTES if route != EMBEDDINGS_ROUTE
)
REMAINING_REQUESTS: Final = "x-ratelimit-api_key-remaining-requests"


@pytest.mark.parametrize("route", ROUTE_PARAMS)
def test_a_hit_writes_a_zero_spend_hit_row_on_every_route(gateway: Gateway, route: str) -> None:
    """On each route a priced model is asked twice by one key; the second is a hit. Two spend rows must exist,
    both with the route's ``call_type`` (``atext_completion``, ``aresponses``, ``aanthropic_messages``,
    ``aembedding`` or ``arerank``); exactly one has ``cache_hit = 'True'`` and its ``spend`` must be 0, while the
    other row's ``cache_hit`` is not ``'True'``.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route, **PRICED)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        assert first.reached_upstream, first.response.text
        await_named_entry(store, target.needle(first))
        target.assert_hit(first, target.ask(text, key=key))
        rows: Final = await_spend_rows(gateway, key, 2)
        assert {row["call_type"] for row in rows} == {CALL_TYPES[route]}, rows
        hits: Final = [row for row in rows if row["cache_hit"] == "True"]
        assert len(hits) == 1, rows
        assert float(str(hits[0]["spend"])) == 0, hits


@pytest.mark.parametrize("route", ROUTE_PARAMS)
def test_a_hit_reports_zero_response_cost_on_every_route(gateway: Gateway, route: str) -> None:
    """On each route a priced model is asked twice. The hit's ``x-litellm-response-cost`` header must parse as
    0: a hit is free on every route, including the Rust messages path that computes cost natively.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route, **PRICED)
        text: Final = prompt()
        first, _ = target.fill(store, text)
        second: Final = target.ask(text)
        target.assert_hit(first, second)
        assert float(second.response.headers[COST_HEADER]) == 0, dict(second.response.headers)


@pytest.mark.parametrize("route", HEADER_ROUTE_PARAMS)
def test_a_hit_carries_the_cache_key_header_that_names_its_entry_on_every_route(gateway: Gateway, route: str) -> None:
    """On each non-embedding route the hit's ``x-litellm-cache-key`` must equal the Redis key that holds the
    entry, and the filling response must not carry the header. Embeddings are pinned separately below.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        first, key = target.fill(store, text)
        assert CACHE_KEY_HEADER not in first.response.headers, dict(first.response.headers)
        second: Final = target.ask(text)
        target.assert_hit(first, second)
        assert second.response.headers[CACHE_KEY_HEADER] == key


def test_a_full_embedding_hit_carries_no_cache_key_header(gateway: Gateway) -> None:
    """A single-input embedding request is filled and then served whole from the cache. The hit must not carry
    ``x-litellm-cache-key``: the embedding path assembles the answer from per-input entries and never stamps a
    ``cache_key`` hidden param (``_process_async_embedding_cached_response``, ``caching_handler.py:695``), unlike
    every other route. UNVERIFIED pin of current behaviour; if the Rust path adds the header (for example the key
    of a one-input request) this cell must be updated deliberately, and it is an inconsistency worth fixing.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, EMBEDDINGS_ROUTE)
        text: Final = prompt()
        first, _ = target.fill(store, text)
        second: Final = target.ask(text)
        target.assert_hit(first, second)
        assert CACHE_KEY_HEADER not in second.response.headers, dict(second.response.headers)


def test_a_hit_counts_against_the_keys_rpm_limit(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 2`` fills an entry and then gets a hit for the identical request. A third identical
    request in the same minute must be refused with 429 although its answer is cached: the limiter's pre-call
    hook counts every request before the cache is read.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model], rpm_limit=2)
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        await_named_entry(store, target.needle(first))
        target.assert_hit(first, target.ask(text, key=key))
        refused: Final = target.ask(text, key=key)
        assert refused.response.status_code == 429, refused.response.text
        assert not refused.reached_upstream


def test_a_hit_reports_the_keys_remaining_requests(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 5`` sends a filling request and then a hit. The fill must carry
    ``x-ratelimit-api_key-remaining-requests: 4`` and the hit ``3``: a hit consumes a request and reports the
    window like any other response.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model], rpm_limit=5)
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        await_named_entry(store, target.needle(first))
        second: Final = target.ask(text, key=key)
        target.assert_hit(first, second)
        assert first.response.headers.get(REMAINING_REQUESTS) == "4", dict(first.response.headers)
        assert second.response.headers.get(REMAINING_REQUESTS) == "3", dict(second.response.headers)


def test_a_hit_counts_against_the_teams_rpm_limit(gateway: Gateway) -> None:
    """A team with ``rpm_limit: 2`` has one key. The key fills an entry and gets one hit; the third identical
    request must be refused with 429: hits count against team windows too.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        team: Final = scenario.team(rpm_limit=2)
        key: Final = scenario.key(team_id=team, models=[target.model])
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        await_named_entry(store, target.needle(first))
        target.assert_hit(first, target.ask(text, key=key))
        assert target.ask(text, key=key).response.status_code == 429


def test_a_hit_counts_against_the_keys_model_rpm_limit(gateway: Gateway) -> None:
    """A key with ``model_rpm_limit: {<model>: 2}`` fills an entry and gets one hit on that model. The third
    identical request must be refused with 429: the per-model window counts hits as well.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model], model_rpm_limit={target.model: 2})
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        await_named_entry(store, target.needle(first))
        target.assert_hit(first, target.ask(text, key=key))
        assert target.ask(text, key=key).response.status_code == 429


def test_a_hits_stored_tokens_count_against_the_keys_tpm_limit(gateway: Gateway) -> None:
    """A key with ``tpm_limit: 3`` asks a model whose answer reports 2 total tokens: one fill and then hits for
    the identical request. Within a few requests, and at the latest once the fill and one hit were charged
    (4 tokens), a request must be refused with 429: the limiter charges a hit's stored usage like a provider
    answer. UNVERIFIED: whether the v3 limiter skips token settlement for cache hits was not confirmed (no
    ``cache_hit`` branch found in ``parallel_request_limiter_v3.py``); if hits are free for tpm this cell
    shows it.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model], tpm_limit=3)
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        await_named_entry(store, target.needle(first))
        statuses: Final = [target.ask(text, key=key).response.status_code for _ in range(4)]
        assert 429 in statuses, statuses
        assert target.calls() == 1


def test_a_request_refused_by_a_rate_limit_writes_no_hit_row(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` fills an entry, then sends the identical request and is refused with 429. The
    key must end with exactly one spend row, the miss: a refused request is never served from the cache and never
    logged as a hit.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE, **PRICED)
        key: Final = scenario.key(models=[target.model], rpm_limit=1)
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        await_named_entry(store, target.needle(first))
        assert target.ask(text, key=key).response.status_code == 429
        rows: Final = await_spend_rows(gateway, key, 1)
        assert rows[0]["cache_hit"] != "True", rows


def test_a_team_at_its_budget_is_refused_even_when_the_answer_is_cached(gateway: Gateway) -> None:
    """A team with ``max_budget: 0.001`` has one key on a model priced so one call costs 0.003. Once the fill's
    spend reaches the team, the identical request must be refused with 422 and a ``Budget has been exceeded``
    message instead of being served from the cache.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE, **PRICED)
        team: Final = scenario.team(max_budget=0.001)
        key: Final = scenario.key(team_id=team, models=[target.model])
        text: Final = prompt()
        first: Final = target.ask(text, key=key)
        await_named_entry(store, target.needle(first))
        refused: Final = eventually(
            lambda: target.ask(text, key=key), lambda exchange: exchange.response.status_code != 200, SPEND_SECONDS
        )
        assert refused.response.status_code == 422, refused.response.text
        assert "Budget has been exceeded" in error_message(refused.response), refused.response.text


def test_an_end_user_at_its_budget_is_refused_even_when_the_answer_is_cached(gateway: Gateway) -> None:
    """An end user is created with ``max_budget: 0.001`` through ``/end_user/new``. A request with ``user`` set to
    that end user costs 0.003 and fills an entry. Once the spend is recorded, the identical request (same
    ``user``, so the same key) must be refused instead of served from the cache. UNVERIFIED: the exact status
    (422 ``budget_exceeded`` is assumed from the key budget path); the cell only requires a non-200 refusal that
    names the budget.
    """
    store: Final = redis_store()
    end_user: Final = f"dec1-cache2-end-user-{uuid.uuid4().hex[:8]}"
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE, **PRICED)
        gateway.post("/end_user/new", {"user_id": end_user, "max_budget": 0.001})
        scenario.cleanups.callback(gateway.post, "/end_user/delete", {"user_ids": [end_user]})
        text: Final = prompt()
        first: Final = target.ask(text, user=end_user)
        await_named_entry(store, target.needle(first))
        refused: Final = eventually(
            lambda: target.ask(text, user=end_user),
            lambda exchange: exchange.response.status_code != 200,
            SPEND_SECONDS,
        )
        assert "budget" in refused.response.text.lower(), refused.response.text


def test_a_hit_row_keeps_the_end_user_and_tags_of_the_request(gateway: Gateway) -> None:
    """A key sends a request with ``user: dec1-eu`` and ``metadata.tags: ["dec1-tag"]`` twice; the second is a
    hit. The hit's spend row must carry ``end_user`` ``dec1-eu`` and ``request_tags`` containing ``dec1-tag``,
    the same attribution as the miss row: a hit is attributed to the caller of the hit, not to the original
    request. UNVERIFIED: ``request_tags`` is read as a JSON list column.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        extra: Final = {"user": "dec1-eu", "metadata": {"tags": ["dec1-tag"]}}
        first: Final = target.ask(text, key=key, **extra)
        await_named_entry(store, target.needle(first))
        target.assert_hit(first, target.ask(text, key=key, **extra))
        await_spend_rows(gateway, key, 2)
        rows: Final = gateway.rows(
            'SELECT cache_hit, end_user, request_tags FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)
        )
        hit: Final = next(row for row in rows if row["cache_hit"] == "True")
        assert hit["end_user"] == "dec1-eu", rows
        assert "dec1-tag" in str(hit["request_tags"]), rows
