"""Per-request cache controls: the ``cache`` object in the request body and the ``Cache-Control`` header.

A caller steers the response cache for one request with ``"cache": {...}`` in the JSON body
(``DynamicCacheControl`` in ``litellm/types/caching.py``):

``no-cache`` skips the lookup but still stores the fresh answer, ``no-store`` skips the write but still
reads, ``ttl`` sets the entry's lifetime in seconds, ``s-maxage`` (also spelled ``s-max-age``) refuses
entries older than that many seconds, ``namespace`` prefixes the entry key with ``<namespace>:``.
Every cell uses the shared rig (Redis, ``default_on``) and proves the effect from the outside: upstream
call counts, the Redis key and its TTL, and the hit header.

The ``Cache-Control`` HTTP header form is read in ``add_litellm_data_to_request`` and maps ``s-maxage``
to the entry TTL; see the BUG skip below for its current state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    SETTLE_SECONDS,
    assert_hit,
    assert_miss,
    await_named_entry,
    filled,
    probe,
    sse_payloads,
    unique_embedding,
)
from tests.integration.caching.response_cache_case import (
    CACHE_KEY,
    CHAT_STREAM,
    MESSAGE,
    MESSAGES_MODEL,
    RERANK,
    TEXT,
    TEXT_MODEL,
    await_entry,
    json_response,
    message_body,
    prompt,
    redis_store,
)
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse

NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
NO_STORE: Final[dict[str, JsonValue]] = {"no-store": True}


def test_no_cache_skips_the_lookup_and_returns_a_fresh_answer_without_the_hit_header(gateway: Gateway) -> None:
    """An entry exists. The same request with ``cache: {"no-cache": true}`` must reach the upstream, return
    a new id and carry no ``x-litellm-cache-key`` header.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        fresh: Final = target.chat(text, cache=NO_CACHE)
        assert_miss(first, fresh)
        assert CACHE_KEY_HEADER not in fresh.response.headers, dict(fresh.response.headers)


def test_no_cache_still_stores_the_fresh_answer_over_the_old_entry(gateway: Gateway) -> None:
    """An entry exists under key K. A ``no-cache`` request for the same prompt gets a fresh answer, and that
    fresh answer must replace the value stored under K in Redis (``no-cache`` only disables the read; the
    write path checks ``no-store`` alone). This is how callers refresh a stale entry.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        key: Final = filled(store, first)
        fresh: Final = target.chat(text, cache=NO_CACHE)
        assert fresh.reached_upstream, fresh.response.text
        assert await_entry(store, fresh.id()) == key


def test_no_cache_false_behaves_like_no_control_at_all(gateway: Gateway) -> None:
    """An entry exists. The same request with ``cache: {"no-cache": false}`` must be a hit: only a literal
    ``true`` disables the lookup.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        assert_hit(first, target.chat(text, cache={"no-cache": False}))


@pytest.mark.parametrize(
    ("path", "answer", "litellm_model"),
    (
        pytest.param("/v1/completions", TEXT, TEXT_MODEL, id="completions"),
        pytest.param("/v1/messages", MESSAGE, MESSAGES_MODEL, id="messages"),
        pytest.param("/v1/rerank", RERANK, "cohere/rerank-v4.0", id="rerank"),
    ),
)
def test_no_cache_skips_the_lookup_on_every_cached_endpoint(
    gateway: Gateway, path: str, answer: Mapping[str, JsonValue], litellm_model: str
) -> None:
    """On ``/v1/completions``, ``/v1/messages`` and ``/v1/rerank`` an entry is filled and the identical request is
    repeated with ``cache: {"no-cache": true}`` in the body. Each repeat must reach the upstream and return a
    new id, so the control is honoured outside chat completions too.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(answer), model=litellm_model)
        text: Final = prompt()
        bodies: Final[dict[str, dict[str, JsonValue]]] = {
            "/v1/completions": {"model": target.model, "prompt": text},
            "/v1/messages": message_body(target.model, text),
            "/v1/rerank": {"model": target.model, "query": text, "documents": ["a", "b"]},
        }
        first: Final = target.send(path, bodies[path])
        filled(store, first)
        assert_miss(first, target.send(path, {**bodies[path], "cache": NO_CACHE}))


def test_no_cache_on_embeddings_sends_every_input_upstream_again(gateway: Gateway) -> None:
    """An embedding entry exists for an input, and a plain repeat is served from it. The same
    ``/v1/embeddings`` request with ``cache: {"no-cache": true}`` must reach the upstream again, so the
    provider has been called twice for three requests.
    The entry is found by the unique vector the provider returned: the answer's ``model`` is restamped to the
    public model name (``litellm/proxy/common_request_processing.py:3689``), so it cannot tell a fresh answer
    from a stored one; the upstream call count does.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        answer, marker = unique_embedding()
        target: Final = probe(scenario, answer, model="openai/text-embedding-3-small")
        body: Final[dict[str, JsonValue]] = {"model": target.model, "input": prompt()}
        first: Final = target.send("/v1/embeddings", body)
        assert first.reached_upstream, first.response.text
        await_entry(store, marker)
        assert not target.send("/v1/embeddings", body).reached_upstream
        fresh: Final = target.send("/v1/embeddings", {**body, "cache": NO_CACHE})
        assert fresh.reached_upstream, fresh.response.text
        assert fresh.json()["data"] == first.json()["data"], fresh.response.text
        assert target.calls() == 2


def test_no_store_answers_normally_and_leaves_no_entry(gateway: Gateway) -> None:
    """A request with ``cache: {"no-store": true}`` is answered 200 from the upstream. A following plain
    identical request must reach the upstream too, and only then does an entry appear: the ``no-store``
    answer was never written.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        unstored: Final = target.chat(text, cache=NO_STORE)
        assert unstored.reached_upstream and unstored.response.status_code == 200, unstored.response.text
        stored: Final = target.chat(text)
        assert_miss(unstored, stored)
        filled(store, stored)
        assert target.calls() == 2


def test_no_store_still_reads_an_existing_entry(gateway: Gateway) -> None:
    """An entry exists. The same request with ``cache: {"no-store": true}`` must be served from it without an
    upstream call: ``no-store`` only disables the write (``should_store_result_in_cache``), not the lookup.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        assert_hit(first, target.chat(text, cache=NO_STORE))


def test_no_store_on_a_stream_leaves_no_entry(gateway: Gateway) -> None:
    """A streamed chat request with ``cache: {"no-store": true}`` completes. The identical streamed request
    without the control must reach the upstream again, so the assembled stream was not written.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        text: Final = prompt()
        first: Final = target.chat(text, stream=True, cache=NO_STORE)
        assert first.reached_upstream and sse_payloads(first.response), first.response.text
        second: Final = target.chat(text, stream=True)
        assert second.reached_upstream, second.response.text
        assert target.calls() == 2


def test_no_cache_and_no_store_together_neither_read_nor_write(gateway: Gateway) -> None:
    """An entry exists under key K holding answer A. A request with both ``no-cache`` and ``no-store`` reaches
    the upstream and gets answer B, and the value under K must still be A afterwards.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        key: Final = filled(store, first)
        bypass: Final = target.chat(text, cache={"no-cache": True, "no-store": True})
        assert_miss(first, bypass)
        raw: Final = store.get(key)
        assert raw is not None and first.id().encode() in raw and bypass.id().encode() not in raw


def test_request_ttl_longer_than_the_default_sets_the_redis_expiry(gateway: Gateway) -> None:
    """A request with ``cache: {"ttl": 300}`` fills an entry. Its Redis TTL must be above the 60 second default
    and at most 300, proving the per-request value replaced the default.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = filled(store, target.chat(prompt(), cache={"ttl": 300}))
        assert 60 < store.ttl(key) <= 300, store.ttl(key)


def test_an_entry_with_a_short_ttl_expires_and_the_next_request_reaches_upstream(gateway: Gateway) -> None:
    """An entry is filled with ``cache: {"ttl": 3}`` and hit once. The Redis key must disappear by itself, and
    after that an identical plain request must reach the upstream again and get a new id. The worker's
    in-memory copy takes the same TTL, so no worker may keep serving the expired answer.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text, cache={"ttl": 3})
        key: Final = filled(store, first)
        eventually(lambda: store.exists(key), lambda present: present == 0, SETTLE_SECONDS)
        after: Final = eventually(lambda: target.chat(text), lambda exchange: exchange.reached_upstream, SETTLE_SECONDS)
        assert after.id() != first.id(), after.response.text


def test_top_level_ttl_in_the_body_sets_the_entry_expiry(gateway: Gateway) -> None:
    """A request carries ``"ttl": 300`` at the top level of the body instead of inside ``cache``. The entry's
    Redis TTL must be above 60 and at most 300, and ``ttl`` must not be forwarded to the provider.
    UNVERIFIED: ``ttl`` is a litellm-owned kwarg that ``Cache._add_cache_logic`` passes to the backend; confirm
    the router keeps it on the call.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = filled(store, target.chat(prompt(), ttl=300))
        assert 60 < store.ttl(key) <= 300, store.ttl(key)
        assert all("ttl" not in body for body in target.upstream_bodies()), target.upstream_bodies()


@pytest.mark.parametrize("spelling", ("s-maxage", "s-max-age"))
def test_s_maxage_refuses_an_entry_older_than_the_given_age(gateway: Gateway, spelling: str) -> None:
    """An entry is filled and hit by a plain request. Requests with ``cache: {"s-maxage": 1}`` (and the
    ``s-max-age`` spelling) are then repeated: once the entry is older than one second such a request must
    reach the upstream, although the entry itself lives for 60 seconds. The age comes from the ``timestamp``
    stored with the entry.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        assert_hit(first, target.chat(text))
        stale: Final = eventually(
            lambda: target.chat(text, cache={spelling: 1}), lambda exchange: exchange.reached_upstream, SETTLE_SECONDS
        )
        assert stale.id() != first.id(), stale.response.text


def test_s_maxage_larger_than_the_entry_age_is_a_hit(gateway: Gateway) -> None:
    """An entry was filled moments ago. A request with ``cache: {"s-maxage": 600}`` must be served from it."""
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        assert_hit(first, target.chat(text, cache={"s-maxage": 600}))


def test_request_namespace_prefixes_the_entry_key_and_the_hit_header(gateway: Gateway) -> None:
    """A request with ``cache: {"namespace": "<ns>"}`` fills an entry. The Redis key must be ``<ns>:`` followed
    by a sha256 hex digest, and a repeat with the same namespace must be a hit whose ``x-litellm-cache-key``
    is that full namespaced key.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        namespace: Final = f"dec1-ns-{scenario.gateway.key[-4:]}-{prompt()[-8:]}"
        first: Final = target.chat(text, cache={"namespace": namespace})
        assert first.reached_upstream, first.response.text
        key: Final = await_named_entry(store, first.id())
        prefix, _, digest = key.partition(":")
        assert prefix == namespace and CACHE_KEY.match(digest), key
        second: Final = target.chat(text, cache={"namespace": namespace})
        assert not second.reached_upstream and second.id() == first.id(), second.response.text
        assert second.cache_key() == key


def test_request_namespaces_isolate_identical_requests(gateway: Gateway) -> None:
    """The same request is sent under namespace A, then under namespace B, then with no namespace. All three
    must reach the upstream, and a repeat under A must still be a hit on A's answer.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        suffix: Final = prompt()[-8:]
        in_a: Final = target.chat(text, cache={"namespace": f"dec1-a-{suffix}"})
        await_named_entry(store, in_a.id())
        in_b: Final = target.chat(text, cache={"namespace": f"dec1-b-{suffix}"})
        bare: Final = target.chat(text)
        assert in_a.reached_upstream and in_b.reached_upstream and bare.reached_upstream
        assert len({in_a.id(), in_b.id(), bare.id()}) == 3
        again: Final = target.chat(text, cache={"namespace": f"dec1-a-{suffix}"})
        assert not again.reached_upstream and again.id() == in_a.id(), again.response.text


def test_the_cache_object_is_never_forwarded_to_the_provider(gateway: Gateway) -> None:
    """A request carries ``cache: {"ttl": 120, "namespace": "dec1-forward"}``. The body the upstream receives
    must not contain a ``cache`` field: the control is consumed by the proxy.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        sent: Final = target.chat(prompt(), cache={"ttl": 120, "namespace": "dec1-forward"})
        assert sent.reached_upstream and sent.response.status_code == 200, sent.response.text
        bodies: Final = target.upstream_bodies()
        assert len(bodies) == 1 and "cache" not in bodies[0], bodies


def test_an_unknown_cache_control_is_ignored(gateway: Gateway) -> None:
    """A request carries ``cache: {"dec1-unknown-control": true}``. It must answer 200, fill the cache as a
    plain request would, and the identical repeat must be a hit.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        control: Final[dict[str, JsonValue]] = {"dec1-unknown-control": True}
        first: Final = target.chat(text, cache=control)
        filled(store, first)
        assert_hit(first, target.chat(text, cache=control))


@pytest.mark.parametrize("value", ("no-cache", True, 5, ["no-cache"]), ids=("string", "bool", "int", "list"))
def test_a_cache_field_that_is_not_an_object_never_causes_a_server_error(gateway: Gateway, value: JsonValue) -> None:
    """A request carries a ``cache`` field that is a string, a boolean, a number or a list instead of an
    object. The proxy must answer either 200 (control ignored) or a 4xx validation error, never a 5xx.
    UNVERIFIED: ``LLMCachingHandler.async_get_cache`` calls ``kwargs.get("cache", {}).get("no-cache")``, which
    raises ``AttributeError`` on a non-dict, so this may currently be a 500.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        answered: Final = target.chat(prompt(), cache=value)
        assert answered.response.status_code < 500, answered.response.text


def test_top_level_caching_false_bypasses_the_cache_for_that_request(gateway: Gateway) -> None:
    """An entry exists. The same request with ``"caching": false`` at the top level of the body must reach the
    upstream, since the SDK only consults the cache when ``caching`` is unset or true.
    UNVERIFIED: the router may overwrite ``caching`` with its own ``cache_responses`` setting before the call;
    see ``ProxyConfig.switch_on_llm_response_caching`` and ``LLMCachingHandler.async_get_cache``.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        assert_miss(first, target.chat(text, caching=False))


def test_cache_control_header_s_maxage_sets_the_entry_ttl(gateway: Gateway) -> None:
    """A request with the HTTP header ``Cache-Control: s-maxage=300`` and no ``cache`` body field fills an
    entry. Its Redis TTL must be above 60 and at most 300: ``add_litellm_data_to_request`` maps the header's
    ``s-maxage`` directive to the call's ``ttl``.
    """
    pytest.skip(
        "BUG: the Cache-Control request header is never honoured; add_litellm_data_to_request looks up "
        '_headers.get("Cache-Control") on a plain dict whose keys are the lowercased ASGI header names'
    )
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first: Final = target.chat(prompt(), headers={"Cache-Control": "s-maxage=300"})
        key: Final = filled(store, first)
        assert 60 < store.ttl(key) <= 300, store.ttl(key)


def test_cache_control_header_no_cache_alone_does_not_bypass_the_response_cache(gateway: Gateway) -> None:
    """An entry exists. The same request with the HTTP header ``Cache-Control: no-cache`` and no ``cache``
    body field must still be a hit: the header form only ever maps ``s-maxage``, and ``no-cache`` /
    ``no-store`` are honoured in the body ``cache`` object alone. Clients that send ``Cache-Control: no-cache``
    by default (browsers, some SDK transports) must not silently lose caching after the rewrite.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        assert_hit(first, target.chat(text, headers={"Cache-Control": "no-cache"}))


def test_a_malformed_cache_control_header_does_not_fail_the_request(gateway: Gateway) -> None:
    """A request carries ``Cache-Control: s-maxage=1=2, ,=`` which ``parse_cache_control`` cannot split into
    one key and one value. The request must still be answered 200.
    UNVERIFIED: today the header is not read at all (see the BUG skip above); once it is, ``key, value =
    directive.split("=")`` raises ``ValueError`` on this input.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        answered: Final = target.chat(prompt(), headers={"Cache-Control": "s-maxage=1=2, ,="})
        assert answered.response.status_code == 200, answered.response.text


def test_use_cache_is_accepted_and_harmless_when_the_cache_is_already_default_on(gateway: Gateway) -> None:
    """On the ``default_on`` rig a request with ``cache: {"use-cache": true}`` fills an entry and a plain
    request for the same prompt is a hit on it: the opt-in flag is not part of the key and changes nothing
    when caching is already on.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text, cache={"use-cache": True})
        filled(store, first)
        assert_hit(first, target.chat(text))


def test_request_ttl_on_a_messages_request_sets_the_redis_expiry(gateway: Gateway) -> None:
    """An Anthropic ``/v1/messages`` request with ``cache: {"ttl": 300}`` fills an entry whose Redis TTL must be
    above 60 and at most 300, so the control works on the Messages endpoint as on chat.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, _message(), model=MESSAGES_MODEL)
        key: Final = filled(store, target.messages(prompt(), cache={"ttl": 300}))
        assert 60 < store.ttl(key) <= 300, store.ttl(key)


def _message() -> JsonResponse:
    return json_response(MESSAGE)
