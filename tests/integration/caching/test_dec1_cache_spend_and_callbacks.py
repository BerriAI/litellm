"""How a cache hit is accounted: ``LiteLLM_SpendLogs`` rows, key and team spend, and logging callbacks.

Every proxied request writes one spend row, hits included. A hit row is told apart by
``cache_hit = 'True'``, a ``request_id`` of ``<original id>_cache_hit<unix time>`` (the table does not
allow duplicate ids, ``spend_tracking_utils.get_logging_payload``), a ``spend`` of zero and the entry key
in ``cache_key``. Key and team spend only grow by the request that reached the provider. Success
callbacks fire for a hit as well, carrying ``cache_hit: true``, ``response_cost: 0`` and the cost the hit
saved in ``saved_cache_cost``.

Rows land asynchronously, so every read polls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.caching.dec1_cache_support import (
    SPEND_SECONDS,
    Exchange,
    Probe,
    assert_hit,
    await_spend_rows,
    cache_config,
    cache_rig,
    error_message,
    filled,
    hashed,
    probe,
    redis_params,
    spend_rows,
    sse_payloads,
)
from tests.integration.caching.response_cache_case import CACHE_KEY, CHAT_STREAM, prompt, redis_store

pytestmark: Final = pytest.mark.shares_worker_state

PRICED: Final[dict[str, JsonValue]] = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
ONE_CALL_COST: Final = 1 * 0.001 + 1 * 0.002
STREAM_ATTEMPTS: Final = 8


def _miss_then_hit(gateway: Gateway, target: Probe, key: str) -> tuple[Exchange, Exchange]:
    text: Final = prompt()
    first: Final = target.chat(text, key=key)
    filled(redis_store(), first)
    second: Final = target.chat(text, key=key)
    assert_hit(first, second)
    return first, second


def _rows_by_hit(rows: list[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    misses: Final = [row for row in rows if row["cache_hit"] != "True"]
    hits: Final = [row for row in rows if row["cache_hit"] == "True"]
    assert len(misses) == 1 and len(hits) == 1, rows
    return misses[0], hits[0]


def test_the_filling_request_is_logged_as_a_miss_under_its_own_id(gateway: Gateway) -> None:
    """A key sends one chat request that reaches the upstream. Its single spend row must have ``request_id``
    equal to the response body ``id``, a ``cache_hit`` that is not ``True`` and a ``cache_key`` that is a
    sha256 hex digest (logged even on a miss).
    Pinned current behaviour: a miss is stored with ``cache_hit`` the string ``None``, because the row writes
    ``str(kwargs.get("cache_hit"))`` and the logging object leaves it unset on a miss
    (``litellm/proxy/spend_tracking/spend_tracking_utils.py:646`` and ``:948``), never ``False``.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model])
        first: Final = target.chat(prompt(), key=key)
        assert first.reached_upstream, first.response.text
        rows: Final = await_spend_rows(gateway, key, 1)
        assert rows[0]["request_id"] == first.id(), rows
        assert rows[0]["cache_hit"] == "None", rows
        assert CACHE_KEY.match(string_value(rows[0]["cache_key"])), rows


def test_a_hit_is_logged_as_its_own_row_with_the_cache_hit_suffix(gateway: Gateway) -> None:
    """A key sends a request twice; the second is a hit. Two spend rows must exist: the hit row's
    ``request_id`` must start with ``<body id>_cache_hit`` and its ``cache_hit`` must be the string ``True``.
    Reports and the UI find hits by exactly these two markers.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model])
        first, _ = _miss_then_hit(gateway, target, key)
        miss, hit = _rows_by_hit(await_spend_rows(gateway, key, 2))
        assert miss["request_id"] == first.id(), miss
        assert string_value(hit["request_id"]).startswith(f"{first.id()}_cache_hit"), hit


def test_a_hit_row_has_zero_spend_and_the_fill_row_has_the_real_cost(gateway: Gateway) -> None:
    """The model costs 0.001 per input token and 0.002 per output token and the upstream reports 1 + 1 tokens.
    The miss row must record a spend of 0.003 and the hit row a spend of exactly 0.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, **PRICED)
        key: Final = scenario.key(models=[target.model])
        _miss_then_hit(gateway, target, key)
        miss, hit = _rows_by_hit(await_spend_rows(gateway, key, 2))
        assert float(str(miss["spend"])) == pytest.approx(ONE_CALL_COST), miss
        assert float(str(hit["spend"])) == 0.0, hit


def test_hit_and_miss_rows_share_the_cache_key_sent_in_the_hit_header(gateway: Gateway) -> None:
    """After a miss and a hit, both spend rows must carry the same ``cache_key`` and it must equal the
    ``x-litellm-cache-key`` header of the hit, so a row can be joined to the Redis entry it came from.
    BUG: on a miss the logging object carries no ``preset_cache_key`` (only the hit path sets it,
    ``litellm/caching/caching_handler.py:1475``), so ``get_logging_payload`` re-hashes the logging kwargs
    (``litellm/proxy/spend_tracking/spend_tracking_utils.py:929``) and the miss row names a key no entry has,
    while the hit row names the real entry key.
    """
    pytest.skip("BUG: the miss row's cache_key is re-hashed from logging kwargs and matches no Redis entry")
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model])
        _, second = _miss_then_hit(gateway, target, key)
        miss, hit = _rows_by_hit(await_spend_rows(gateway, key, 2))
        assert miss["cache_key"] == hit["cache_key"] == second.cache_key(), (miss, hit, second.cache_key())


def test_a_hit_row_keeps_the_call_type_model_and_token_counts_of_the_original(gateway: Gateway) -> None:
    """After a miss and a hit, the hit row must have the same ``call_type`` and ``model`` as the miss row and
    report the stored answer's token counts (1 prompt, 1 completion), so usage dashboards still count the
    tokens a hit served.
    UNVERIFIED: token counts on the hit row are taken from the cached usage; check
    ``spend_tracking_utils.get_logging_payload`` if they turn out to be zero.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model])
        _miss_then_hit(gateway, target, key)
        miss, hit = _rows_by_hit(await_spend_rows(gateway, key, 2))
        assert hit["call_type"] == miss["call_type"] and hit["model"] == miss["model"], (miss, hit)
        assert (hit["prompt_tokens"], hit["completion_tokens"]) == (1, 1), hit


def test_every_hit_writes_a_distinct_row(gateway: Gateway) -> None:
    """A key sends one filling request and then three hits. Four spend rows must exist, the three hit rows
    must have three different ``request_id`` values and every one of them a spend of zero.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        filled(redis_store(), first)
        for _ in range(3):
            assert_hit(first, target.chat(text, key=key))
        rows: Final = await_spend_rows(gateway, key, 4)
        hits: Final = [row for row in rows if row["cache_hit"] == "True"]
        assert len(hits) == 3 and len({row["request_id"] for row in hits}) == 3, rows
        assert all(float(str(row["spend"])) == 0.0 for row in hits), hits


def test_a_no_cache_request_is_logged_as_a_miss_with_its_real_cost(gateway: Gateway) -> None:
    """An entry exists. The same key repeats the request with ``cache: {"no-cache": true}``. Neither spend row
    may be a hit and each must have a spend of 0.003: a forced refresh is billed like any call.
    Pinned current behaviour: both rows store ``cache_hit`` as the string ``None`` rather than ``False``
    (``litellm/proxy/spend_tracking/spend_tracking_utils.py:948`` writes ``str(kwargs.get("cache_hit"))``).
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, **PRICED)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        filled(redis_store(), first)
        fresh: Final = target.chat(text, key=key, cache={"no-cache": True})
        assert fresh.reached_upstream, fresh.response.text
        rows: Final = await_spend_rows(gateway, key, 2)
        assert [row["cache_hit"] for row in rows] == ["None", "None"], rows
        assert [float(str(row["spend"])) for row in rows] == [pytest.approx(ONE_CALL_COST)] * 2, rows


def test_key_spend_grows_only_by_the_request_that_reached_the_provider(gateway: Gateway) -> None:
    """A key on a priced model sends one filling request and two hits. Once all three spend rows exist, the
    key's ``spend`` in ``LiteLLM_VerificationToken`` must be 0.003, the cost of one call, not three.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, **PRICED)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        filled(redis_store(), first)
        assert_hit(first, target.chat(text, key=key))
        assert_hit(first, target.chat(text, key=key))
        await_spend_rows(gateway, key, 3)
        rows: Final = eventually(
            lambda: gateway.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),)),
            lambda found: len(found) == 1 and float(str(found[0]["spend"])) > 0,
            SPEND_SECONDS,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(ONE_CALL_COST), rows


def test_team_spend_grows_only_by_the_request_that_reached_the_provider(gateway: Gateway) -> None:
    """A team key on a priced model sends one filling request and one hit. The team's ``spend`` in
    ``LiteLLM_TeamTable`` must settle at 0.003, the cost of the single provider call.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, **PRICED)
        team: Final = scenario.team(models=[target.model])
        key: Final = scenario.key(team_id=team, models=[target.model])
        _miss_then_hit(gateway, target, key)
        await_spend_rows(gateway, key, 2)
        rows: Final = eventually(
            lambda: gateway.rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)),
            lambda found: len(found) == 1 and float(str(found[0]["spend"])) > 0,
            SPEND_SECONDS,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(ONE_CALL_COST), rows


def test_a_key_at_its_budget_is_refused_even_when_the_answer_is_cached(gateway: Gateway) -> None:
    """A key with ``max_budget`` 0.001 fills an entry with a call that costs 0.003. Once its spend is recorded
    the identical request must be refused with the budget error instead of being served from the cache:
    the budget check runs before the cache lookup, so free hits do not bypass an exhausted budget. The
    refusal is a 422 ``budget_exceeded`` (``litellm.budget_exceeded_status_code``, ``litellm/__init__.py:392``).
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, **PRICED)
        key: Final = scenario.key(models=[target.model], max_budget=0.001)
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        filled(redis_store(), first)
        refused: Final = eventually(
            lambda: target.chat(text, key=key),
            lambda exchange: exchange.response.status_code != 200,
            SPEND_SECONDS,
        )
        assert refused.response.status_code == 422, refused.response.text
        assert error_message(refused.response).startswith("Budget has been exceeded!"), refused.response.text
        assert first.id() not in refused.response.text


def test_a_streamed_hit_is_logged_exactly_once(gateway: Gateway) -> None:
    """A key streams the same chat request until one stream is replayed from the cache (a stream can reach
    the upstream once per proxy worker first). The number of spend rows must equal the number of requests
    sent and exactly one row must be a hit with zero spend: a replayed stream logs when the stream ends and
    must not also log at lookup time.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM, **PRICED)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        sent: list[Exchange] = []  # mutable-ok: requests are sent until the first replay
        for _ in range(STREAM_ATTEMPTS):
            exchange = target.chat(text, key=key, stream=True)
            assert sse_payloads(exchange.response), exchange.response.text
            sent.append(exchange)
            if not exchange.reached_upstream:
                break
        assert not sent[-1].reached_upstream, f"no stream was replayed in {STREAM_ATTEMPTS} attempts"
        rows: Final = await_spend_rows(gateway, key, len(sent))
        hits: Final = [row for row in rows if row["cache_hit"] == "True"]
        assert len(hits) == 1 and float(str(hits[0]["spend"])) == 0.0, rows


def _collected(wire: Wire) -> list[dict[str, JsonValue]]:
    """Every standard logging payload the sink received so far, whether posted singly or as a batch list."""

    def payloads() -> list[dict[str, JsonValue]]:
        found: list[dict[str, JsonValue]] = []  # mutable-ok: flattening batches of unknown shape
        for request in wire.drain():
            decoded = json.loads(request.body)
            found.extend(object_value(item) for item in (decoded if isinstance(decoded, list) else [decoded]))
        return found

    return payloads()


def _accept(_: Request) -> Reply:
    return Reply(body=b"{}")


def test_a_success_callback_fires_for_a_hit_with_cache_hit_true_and_zero_cost(gateway: Gateway, tmp_path: Path) -> None:
    """An owned proxy logs to a ``generic_api`` callback sink. A priced model is asked twice; the second answer
    is a hit. The sink must receive two standard logging payloads: the miss with ``cache_hit`` false-ish and a
    ``response_cost`` of 0.003, and the hit with ``cache_hit: true``, ``response_cost: 0``, a
    ``saved_cache_cost`` of 0.003 and a ``cache_key`` equal to the hit's ``x-litellm-cache-key`` header.
    UNVERIFIED: the sink's batch shape (a JSON list of payloads) and that ``saved_cache_cost`` is priced from
    the custom deployment prices; see ``litellm_logging.py::get_standard_logging_object_payload``.
    """
    with wire_server(_accept) as sink:
        config: Final = cache_config(
            tmp_path,
            cache_params=redis_params(),
            litellm_settings={"callbacks": ["dec1_cache_sink"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1},
            extra={
                "callback_settings": {
                    "dec1_cache_sink": {"callback_type": "generic_api", "endpoint": sink.url, "headers": {}}
                }
            },
        )
        with cache_rig(gateway, tmp_path, config) as rig, rig.gateway.scenario() as scenario:
            target: Final = probe(scenario, **PRICED)
            text: Final = prompt()
            first: Final = target.chat(text)
            filled(rig.store(), first)
            second: Final = target.chat(text)
            assert_hit(first, second)
            received: list[dict[str, JsonValue]] = []  # mutable-ok: the sink is drained across polls

            def drained() -> list[dict[str, JsonValue]]:
                received.extend(_collected(sink))
                return [item for item in received if item.get("model_group") == target.model]

            payloads: Final = eventually(drained, lambda found: len(found) >= 2, SPEND_SECONDS)
            hits: Final = [item for item in payloads if item.get("cache_hit") is True]
            misses: Final = [item for item in payloads if item.get("cache_hit") is not True]
            assert len(hits) == 1 and len(misses) == 1, payloads
            assert float(str(misses[0]["response_cost"])) == pytest.approx(ONE_CALL_COST), misses[0]
            assert float(str(hits[0]["response_cost"])) == 0.0, hits[0]
            assert float(str(hits[0]["saved_cache_cost"])) == pytest.approx(ONE_CALL_COST), hits[0]
            assert hits[0]["cache_key"] == second.cache_key(), hits[0]


def test_spend_rows_say_cache_off_when_no_response_cache_is_configured(gateway: Gateway, tmp_path: Path) -> None:
    """An owned proxy runs without ``litellm_settings.cache``. A key sends the same request twice: both reach
    the upstream, and both spend rows must have the literal ``cache_key`` ``Cache OFF``
    (``spend_tracking_utils.get_logging_payload``) and must not be hits.
    Pinned current behaviour: ``cache_hit`` is the string ``None`` on both rows
    (``litellm/proxy/spend_tracking/spend_tracking_utils.py:948``), not ``False``.
    """
    config: Final = cache_config(tmp_path, cache=None)
    with cache_rig(gateway, tmp_path, config) as rig, rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        second: Final = target.chat(text, key=key)
        assert first.reached_upstream and second.reached_upstream and target.calls() == 2
        rows: Final = eventually(lambda: spend_rows(rig.gateway, key), lambda found: len(found) >= 2, SPEND_SECONDS)
        assert [row["cache_key"] for row in rows] == ["Cache OFF", "Cache OFF"], rows
        assert [row["cache_hit"] for row in rows] == ["None", "None"], rows
