"""Limits changed at runtime, and how the rate-limit window opens and resets.

The contract for updates: ``/key/update`` changes a key's ``rpm_limit``, ``tpm_limit``,
``max_parallel_requests`` and ``model_rpm_limit`` without a restart. The new value is used for the key's next
requests in the SAME window: the counters are not reset by an update, only the limit they are compared to.
The stored values are the ``rpm_limit``, ``tpm_limit`` and ``max_parallel_requests`` columns of
``LiteLLM_VerificationToken``. A regenerated key is a new identity with a fresh window.

The contract for windows: the window is fixed, not sliding. It opens at the first request of a scope and
lasts ``LITELLM_RATE_LIMIT_WINDOW_SIZE`` seconds (default 60). When it has elapsed the next request opens a
new window with both the request and token counters at zero. ``retry-after`` is always the window size. The
concurrency gauge is not windowed and survives a window reset.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``window_size``, ``should_rate_limit``) and
the Lua in ``litellm-rust/crates/redis-store/src/rate_limits/lua.rs``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway, eventually, held, object_value, string_value
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    counter_key,
    hashed,
    limits_proxy,
    posts,
    provider,
    scope_headers,
    shared_redis,
    statuses,
)

SHORT_WINDOW: Final = 3
SHORT_WINDOW_ENVIRONMENT: Final = {"LITELLM_RATE_LIMIT_WINDOW_SIZE": str(SHORT_WINDOW)}


def test_raising_a_keys_rpm_limit_admits_it_again_in_the_same_window(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is served once and refused once. ``/key/update`` then sets rpm_limit=5. The key must
    be served again without waiting for the window, and that response must report ``limit-requests: 5`` and
    ``remaining-requests: 2``, because the two earlier requests still count in the unchanged window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert statuses([chat(gateway, model, key), chat(gateway, model, key)]) == [200, 429]
        gateway.post("/key/update", {"key": key, "rpm_limit": 5})
        served: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 200)
        assert scope_headers(served, "api_key") == {"limit-requests": "5", "remaining-requests": "2"}
        assert len(posts(wire)) == 2


def test_lowering_a_keys_rpm_limit_refuses_it_in_the_same_window(gateway: Gateway) -> None:
    """A key with rpm_limit=100 is served once, then ``/key/update`` sets rpm_limit=1. The key must start being
    refused with the 429 for ``api_key`` requests showing ``Current limit: 1``, without a restart."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=100)
        assert chat(gateway, model, key).status_code == 200
        gateway.post("/key/update", {"key": key, "rpm_limit": 1})
        refused: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 429)
        assert_limited(refused, "api_key", hashed(key), "requests", 1)


def test_clearing_a_keys_rpm_limit_with_null_removes_the_limit_and_its_headers(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is exhausted, then ``/key/update`` sends ``rpm_limit: null``. The key must be served
    again and its responses must carry no ``x-ratelimit-api_key-*`` header, and the ``rpm_limit`` column must be
    NULL. UNVERIFIED: assumes an explicit null clears the column on ``/key/update`` as it does on
    ``/user/update``; see ``prepare_key_update_data`` in ``key_management_endpoints.py``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert statuses([chat(gateway, model, key), chat(gateway, model, key)]) == [200, 429]
        gateway.post("/key/update", {"key": key, "rpm_limit": None})
        served: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 200)
        assert scope_headers(served, "api_key") == {}
        rows: Final = gateway.rows('SELECT rpm_limit FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),))
        assert rows == [{"rpm_limit": None}]


def test_raising_a_keys_tpm_limit_admits_a_request_that_was_too_large(gateway: Gateway) -> None:
    """A key with tpm_limit=20 has a 60 token request refused. ``/key/update`` sets tpm_limit=1000 and the same
    size of request must then be served with ``limit-tokens: 1000`` and ``remaining-tokens: 940``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=20)
        assert_limited(chat(gateway, model, key, max_tokens=50), "api_key", hashed(key), "tokens", 20, 20)
        gateway.post("/key/update", {"key": key, "tpm_limit": 1000})
        served: Final = eventually(
            lambda: chat(gateway, model, key, max_tokens=50), lambda response: response.status_code == 200
        )
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "940"}


def test_raising_a_keys_parallel_limit_from_zero_lets_it_through(gateway: Gateway) -> None:
    """A key with max_parallel_requests=0 is refused. ``/key/update`` sets max_parallel_requests=2 and the key must
    then be served with ``limit-max_parallel_requests: 2`` and ``remaining-max_parallel_requests: 1``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=0)
        assert_limited(chat(gateway, model, key), "api_key", hashed(key), "max_parallel_requests", 0)
        gateway.post("/key/update", {"key": key, "max_parallel_requests": 2})
        served: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 200)
        assert scope_headers(served, "api_key") == {
            "limit-max_parallel_requests": "2",
            "remaining-max_parallel_requests": "1",
        }


def test_a_per_model_limit_added_by_key_update_is_enforced(gateway: Gateway) -> None:
    """A key without limits is served, then ``/key/update`` adds ``model_rpm_limit: {<model>: 1}``. The key must
    start being refused with ``Rate limit exceeded for model_per_key: <hash>:<model>`` and limit 1."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        assert chat(gateway, model, key).status_code == 200
        gateway.post("/key/update", {"key": key, "model_rpm_limit": {model: 1}})
        refused: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 429)
        assert_limited(refused, "model_per_key", f"{hashed(key)}:{model}", "requests", 1)


def test_updated_limits_are_stored_on_the_key_row_and_returned_by_key_info(gateway: Gateway) -> None:
    """A key is created with rpm_limit=3, tpm_limit=300, max_parallel_requests=2 and then updated to 4, 400 and
    5. ``GET /key/info`` must return the new values and the ``LiteLLM_VerificationToken`` row must hold them in
    its ``rpm_limit``, ``tpm_limit`` and ``max_parallel_requests`` columns."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=3, tpm_limit=300, max_parallel_requests=2)
        gateway.post("/key/update", {"key": key, "rpm_limit": 4, "tpm_limit": 400, "max_parallel_requests": 5})
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        expected: Final = {"rpm_limit": 4, "tpm_limit": 400, "max_parallel_requests": 5}
        assert {name: info[name] for name in expected} == expected
        rows: Final = gateway.rows(
            'SELECT rpm_limit, tpm_limit, max_parallel_requests FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (hashed(key),),
        )
        assert rows == [expected]


def test_a_regenerated_key_keeps_its_limit_but_starts_a_fresh_window(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is exhausted and then regenerated with ``POST /key/<key>/regenerate``. The new key
    value must be served once (``limit-requests: 1``, ``remaining-requests: 0``) and then refused naming the
    NEW key's hash, because windows are keyed by the token hash."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        original: Final = string_value(gateway.post("/key/generate", {"models": [model], "rpm_limit": 1})["key"])
        assert statuses([chat(gateway, model, original), chat(gateway, model, original)]) == [200, 429]
        regenerated: Final = string_value(gateway.post(f"/key/{original}/regenerate", {})["key"])
        scenario.cleanups.callback(scenario.delete_key, regenerated)
        assert regenerated != original
        served: Final = eventually(
            lambda: chat(gateway, model, regenerated), lambda response: response.status_code == 200
        )
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}
        assert_limited(chat(gateway, model, regenerated), "api_key", hashed(regenerated), "requests", 1)


def test_the_window_start_does_not_move_while_the_window_is_open(gateway: Gateway) -> None:
    """A key with rpm_limit=10 sends three requests. The Redis value ``{api_key:<hash>}:window`` (the epoch second
    the window opened) must be the same after the first and after the third request: later requests join the
    open window instead of sliding it."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10)
        window: Final = counter_key("api_key", hashed(key), "window")
        assert chat(gateway, model, key).status_code == 200
        opened: Final = redis.get(window)
        assert opened is not None
        assert statuses([chat(gateway, model, key), chat(gateway, model, key)]) == [200, 200]
        assert redis.get(window) == opened
        assert len(posts(wire)) == 3


def test_retry_after_is_the_configured_window_size(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy runs with ``LITELLM_RATE_LIMIT_WINDOW_SIZE=3`` and a key with rpm_limit=1 is refused. The 429 must
    carry ``retry-after: 3`` and the Redis request counter must expire within 3 seconds."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, environment=SHORT_WINDOW_ENVIRONMENT) as candidate,
        candidate.scenario() as scenario,
        shared_redis() as redis,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(candidate, model, key).status_code == 200
        refused: Final = chat(candidate, model, key)
        assert refused.status_code == 429, refused.text
        assert refused.headers["retry-after"] == str(SHORT_WINDOW)
        ttl: Final = int(redis.ttl(counter_key("api_key", hashed(key), "requests")))
        assert ttl <= SHORT_WINDOW, ttl


def test_a_key_is_admitted_again_once_its_window_has_elapsed(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy runs with a 3 second window and a key with rpm_limit=1 is served and then refused. Polling without
    changing anything, the key must be served again within a few seconds, and that response must report
    ``remaining-requests: 0`` of a brand new window. Refusals while waiting must not push the reset further out."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, environment=SHORT_WINDOW_ENVIRONMENT) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert statuses([chat(candidate, model, key), chat(candidate, model, key)]) == [200, 429]
        served: Final = eventually(
            lambda: chat(candidate, model, key), lambda response: response.status_code == 200, seconds=5 * SHORT_WINDOW
        )
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}
        assert len(posts(wire)) == 2


def test_the_token_window_resets_together_with_the_request_window(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy runs with a 3 second window and a key has tpm_limit=100. Requests reserving 70 tokens fill the
    window until one is refused. Polling on, a request must again be served with ``remaining-tokens: 30``, which
    is only possible in a new window whose token counter started from zero."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, environment=SHORT_WINDOW_ENVIRONMENT) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        first: Final = chat(candidate, model, key, max_tokens=60)
        assert first.status_code == 200, first.text
        assert scope_headers(first, "api_key") == {"limit-tokens": "100", "remaining-tokens": "30"}
        eventually(
            lambda: chat(candidate, model, key, max_tokens=60),
            lambda response: response.status_code == 429,
            seconds=5 * SHORT_WINDOW,
        )
        fresh: Final = eventually(
            lambda: chat(candidate, model, key, max_tokens=60),
            lambda response: (
                response.status_code == 200 and response.headers.get("x-ratelimit-api_key-remaining-tokens") == "30"
            ),
            seconds=5 * SHORT_WINDOW,
        )
        assert scope_headers(fresh, "api_key") == {"limit-tokens": "100", "remaining-tokens": "30"}


def test_a_held_request_keeps_its_parallel_slot_across_a_window_reset(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy runs with a 3 second window and a key has max_parallel_requests=1. One request is held at the
    provider for longer than two windows. Every other request in that time must be refused for
    max_parallel_requests: the concurrency gauge is not reset when the rpm window rolls over."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        limits_proxy(gateway, tmp_path, environment=SHORT_WINDOW_ENVIRONMENT) as candidate,
        candidate.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        first: Final = pool.submit(chat, candidate, model, key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            refused: Final = held(
                lambda: chat(candidate, model, key),
                lambda response: response.status_code == 429,
                holding=2 * SHORT_WINDOW + 1,
                seconds=4 * SHORT_WINDOW,
            )
            assert_limited(refused, "api_key", hashed(key), "max_parallel_requests", 1)
        finally:
            release.set()
        assert first.result().status_code == 200
