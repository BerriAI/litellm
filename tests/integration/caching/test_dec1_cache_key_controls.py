"""Cache controls attached to a virtual key: ``allowed_cache_controls`` and ``metadata.cache``.

``allowed_cache_controls`` on a key is an allow-list enforced by the ``cache_control_check`` pre-call hook
(``litellm/proxy/hooks/cache_control_check.py``): when the list is non-empty, every name inside the
request's ``cache`` object must be in it, otherwise the request is refused with 403 before it reaches
the provider. An empty list (the default) allows everything.

``metadata.cache`` on a key (``LiteLLMProxyRequestSetup.add_key_level_controls``) is the opposite
direction: the admin forces cache controls onto every request of that key. It replaces whatever
``cache`` object the caller sent and keeps only ``ttl``, ``s-maxage``, ``no-cache`` and ``no-store``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration.caching.dec1_cache_support import (
    assert_hit,
    assert_miss,
    await_named_entry,
    error_message,
    filled,
    hashed,
    probe,
)
from tests.integration.caching.response_cache_case import (
    CACHE_KEY,
    EMBEDDING,
    MESSAGE,
    MESSAGES_MODEL,
    json_response,
    message_body,
    prompt,
    redis_store,
)


def _denied(control: str) -> str:
    return f"Not allowed to set {control} as a cache control. Contact admin to change permissions."


def test_a_control_outside_the_keys_allow_list_is_refused_with_403(gateway: Gateway) -> None:
    """A key is created with ``allowed_cache_controls: ["no-cache"]``. It sends a chat request with
    ``cache: {"ttl": 5}``. The proxy must answer 403 with the message "Not allowed to set ttl as a cache
    control. Contact admin to change permissions." and must not call the upstream.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], allowed_cache_controls=["no-cache"])
        refused: Final = target.chat(prompt(), key=key, cache={"ttl": 5})
        assert refused.response.status_code == 403, refused.response.text
        assert _denied("ttl") in error_message(refused.response), refused.response.text
        assert not refused.reached_upstream and target.calls() == 0


def test_a_control_inside_the_keys_allow_list_is_honoured(gateway: Gateway) -> None:
    """A key allowed only ``no-cache`` sends a request with ``cache: {"no-cache": true}`` for a prompt that is
    already cached. It must answer 200 with a fresh upstream answer.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], allowed_cache_controls=["no-cache"])
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        filled(store, first)
        assert_miss(first, target.chat(text, key=key, cache={"no-cache": True}))


@pytest.mark.parametrize("control", ("ttl", "s-maxage", "no-store", "namespace", "use-cache"))
def test_every_control_name_is_checked_against_the_allow_list(gateway: Gateway, control: str) -> None:
    """A key allowed only ``no-cache`` sends ``cache: {<control>: ...}`` for each other control name
    (``ttl``, ``s-maxage``, ``no-store``, ``namespace``, ``use-cache``). Each must be refused with 403 naming
    that control: the hook checks every key of the ``cache`` object, not a fixed list.
    """
    values: Final[Mapping[str, JsonValue]] = {
        "ttl": 30,
        "s-maxage": 30,
        "no-store": True,
        "namespace": "dec1-denied",
        "use-cache": True,
    }
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], allowed_cache_controls=["no-cache"])
        refused: Final = target.chat(prompt(), key=key, cache={control: values[control]})
        assert refused.response.status_code == 403, refused.response.text
        assert _denied(control) in error_message(refused.response), refused.response.text
        assert target.calls() == 0


def test_one_disallowed_control_refuses_a_request_that_also_carries_allowed_ones(gateway: Gateway) -> None:
    """A key allowed ``ttl`` and ``s-maxage`` sends ``cache: {"ttl": 30, "no-store": true}``. The request must be
    refused with 403 naming ``no-store`` and the upstream must not be called.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], allowed_cache_controls=["ttl", "s-maxage"])
        refused: Final = target.chat(prompt(), key=key, cache={"ttl": 30, "no-store": True})
        assert refused.response.status_code == 403, refused.response.text
        assert _denied("no-store") in error_message(refused.response), refused.response.text
        assert target.calls() == 0


def test_a_key_with_an_allow_list_can_still_send_requests_without_cache_controls(gateway: Gateway) -> None:
    """A key allowed only ``ttl`` sends two plain identical requests. Both must answer 200 and the second
    must be a normal cache hit: the allow-list restricts controls, not caching itself.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], allowed_cache_controls=["ttl"])
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        filled(store, first)
        assert_hit(first, target.chat(text, key=key))


def test_a_key_without_an_allow_list_may_use_any_control(gateway: Gateway) -> None:
    """A key created without ``allowed_cache_controls`` (stored as an empty list) sends
    ``cache: {"ttl": 300, "no-cache": true}``. It must answer 200 and the entry it writes must carry the 300
    second TTL, because an empty allow-list means no restriction.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model])
        answered: Final = target.chat(prompt(), key=key, cache={"ttl": 300, "no-cache": True})
        entry: Final = filled(store, answered)
        assert 60 < store.ttl(entry) <= 300, store.ttl(entry)


def test_the_master_key_is_not_restricted_by_any_allow_list(gateway: Gateway) -> None:
    """The master key sends ``cache: {"ttl": 300, "namespace": ...}``. It must answer 200: the master key has no
    ``allowed_cache_controls`` and so no control is refused.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        answered: Final = target.chat(prompt(), cache={"ttl": 300, "namespace": f"dec1-master-{prompt()[-8:]}"})
        assert answered.response.status_code == 200 and answered.reached_upstream, answered.response.text


@pytest.mark.parametrize("path", ("/v1/embeddings", "/v1/messages", "/v1/completions"))
def test_the_allow_list_is_enforced_on_other_endpoints(gateway: Gateway, path: str) -> None:
    """A key allowed only ``no-cache`` sends ``cache: {"no-store": true}`` on ``/v1/embeddings``,
    ``/v1/messages`` and ``/v1/completions``. Each must be refused with 403 naming ``no-store`` and the
    upstream must not be called, because the hook runs for every call type.
    """
    with gateway.scenario() as scenario:
        answers: Final = {"/v1/embeddings": EMBEDDING, "/v1/messages": MESSAGE, "/v1/completions": MESSAGE}
        models: Final = {
            "/v1/embeddings": "openai/text-embedding-3-small",
            "/v1/messages": MESSAGES_MODEL,
            "/v1/completions": "openai/gpt-3.5-turbo-instruct",
        }
        target: Final = probe(scenario, json_response(answers[path]), model=models[path])
        key: Final = scenario.key(models=[target.model], allowed_cache_controls=["no-cache"])
        bodies: Final[dict[str, dict[str, JsonValue]]] = {
            "/v1/embeddings": {"model": target.model, "input": prompt()},
            "/v1/messages": message_body(target.model, prompt()),
            "/v1/completions": {"model": target.model, "prompt": prompt()},
        }
        refused: Final = target.send(path, {**bodies[path], "cache": {"no-store": True}}, key=key)
        assert refused.response.status_code == 403, refused.response.text
        assert _denied("no-store") in error_message(refused.response), refused.response.text
        assert target.calls() == 0


def test_key_info_returns_the_allow_list_that_was_set(gateway: Gateway) -> None:
    """A key is generated with ``allowed_cache_controls: ["no-cache", "ttl"]``. ``GET /key/info`` for it must
    return exactly that list and the ``LiteLLM_VerificationToken`` row must store it.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(allowed_cache_controls=["no-cache", "ttl"])
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert info["allowed_cache_controls"] == ["no-cache", "ttl"], info
        rows: Final = gateway.rows(
            'SELECT allowed_cache_controls FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),)
        )
        assert rows == [{"allowed_cache_controls": ["no-cache", "ttl"]}], rows


def test_updating_the_allow_list_changes_what_the_key_may_send(gateway: Gateway) -> None:
    """A key starts with ``allowed_cache_controls: ["no-cache"]`` and a ``ttl`` control is refused with 403.
    After ``POST /key/update`` sets the list to ``["no-cache", "ttl"]`` the same request must be answered 200
    (polled, since other workers learn the change through the auth cache).
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], allowed_cache_controls=["no-cache"])
        assert target.chat(prompt(), key=key, cache={"ttl": 30}).response.status_code == 403
        gateway.post("/key/update", {"key": key, "allowed_cache_controls": ["no-cache", "ttl"]})
        allowed: Final = eventually(
            lambda: target.chat(prompt(), key=key, cache={"ttl": 30}),
            lambda exchange: exchange.response.status_code == 200,
            seconds=30,
        )
        assert allowed.reached_upstream, allowed.response.text


def test_key_metadata_no_cache_forces_every_request_of_that_key_past_the_cache(gateway: Gateway) -> None:
    """A key is created with ``metadata: {"cache": {"no-cache": true}}``. An entry exists for a prompt (filled
    by the master key). The restricted key sends the same plain request, with no ``cache`` field, and must
    reach the upstream. The master key repeating it must still be a hit.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"no-cache": True}})
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        forced: Final = target.chat(text, key=key)
        assert_miss(first, forced)
        again: Final = target.chat(text, key=key)
        assert again.reached_upstream, again.response.text


def test_key_metadata_ttl_applies_to_entries_written_by_that_key(gateway: Gateway) -> None:
    """A key is created with ``metadata: {"cache": {"ttl": 300}}``. A plain request by that key fills an entry
    whose Redis TTL must be above the 60 second default and at most 300.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"ttl": 300}})
        entry: Final = filled(store, target.chat(prompt(), key=key))
        assert 60 < store.ttl(entry) <= 300, store.ttl(entry)


def test_key_metadata_cache_replaces_the_cache_object_the_caller_sent(gateway: Gateway) -> None:
    """A key has ``metadata: {"cache": {"ttl": 300}}``. An entry exists and the key sends the same request with
    its own ``cache: {"no-cache": true}``. The request must be a hit: the key-level block replaces the
    caller's object wholesale (``data["cache"] = {}`` and then only the key's controls), so the caller's
    ``no-cache`` is dropped.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"ttl": 300}})
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        filled(store, first)
        assert_hit(first, target.chat(text, key=key, cache={"no-cache": True}))


def test_key_metadata_cache_drops_controls_outside_the_supported_four(gateway: Gateway) -> None:
    """A key has ``metadata: {"cache": {"namespace": "dec1-forced", "ttl": 300}}``. Its request fills an entry
    whose key must be a bare sha256 with no ``dec1-forced:`` prefix, while the TTL of 300 is applied: key
    metadata only carries ``ttl``, ``s-maxage``, ``no-cache`` and ``no-store`` (``SupportedCacheControls``).
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(
            models=[target.model], metadata={"cache": {"namespace": "dec1-forced", "ttl": 300}}
        )
        answered: Final = target.chat(prompt(), key=key)
        entry: Final = await_named_entry(store, answered.id())
        assert CACHE_KEY.match(entry), entry
        assert 60 < store.ttl(entry) <= 300, store.ttl(entry)


def test_key_metadata_no_store_keeps_that_keys_answers_out_of_the_cache(gateway: Gateway) -> None:
    """A key has ``metadata: {"cache": {"no-store": true}}``. It sends a request twice. Both must reach the
    upstream because nothing it receives is written, and a third identical request by the master key must
    reach the upstream as well.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"no-store": True}})
        text: Final = prompt()
        first: Final = target.chat(text, key=key)
        second: Final = target.chat(text, key=key)
        third: Final = target.chat(text)
        assert first.reached_upstream and second.reached_upstream and third.reached_upstream
        assert target.calls() == 3
