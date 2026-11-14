"""Requests-per-minute limits on a virtual key (``rpm_limit``).

The contract: a key created with ``rpm_limit=N`` is admitted N times inside one 60 second window and every
later request in that window is refused with HTTP 429 before the provider is called. The refusal is an
OpenAI-shaped error body (``error.type`` ``throttling_error``, ``error.code`` the string ``"429"``) whose
message names the scope (``api_key``), the SHA-256 hash of the key (never the key itself), the limit kind
(``requests``), the limit, the remaining count and the reset time. The refusal carries ``retry-after`` (the
window length in seconds), ``rate_limit_type`` and ``reset_at`` headers. An admitted response carries
``x-ratelimit-api_key-limit-requests`` and ``x-ratelimit-api_key-remaining-requests``, plus the configured
limits as ``x-litellm-key-rpm-limit`` and ``x-litellm-key-tpm-limit``.

The counters live in Redis under ``{api_key:<hash>}:requests`` with a sibling ``{api_key:<hash>}:window``
holding the epoch second the window opened. The first refused request still increments the request counter.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``should_rate_limit``,
``_handle_rate_limit_error``, ``_merge_ratelimit_statuses_into_additional_headers``),
``litellm/proxy/common_utils/proxy_rate_limit_error.py`` and
``litellm-rust/crates/redis-store/src/rate_limits/``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Final

from tests.integration._support.client import Gateway, object_value
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    RESET_AT,
    WINDOW_SECONDS,
    assert_limited,
    chat,
    counter_key,
    error_message,
    hashed,
    limit_message,
    posts,
    provider,
    ratelimit_header_names,
    redis_integer,
    scope_headers,
    shared_redis,
    statuses,
)


def test_a_key_is_admitted_up_to_its_rpm_limit_and_then_refused_with_429(gateway: Gateway) -> None:
    """A key has rpm_limit=3 and sends five chat requests one after another. The first three must answer 200
    and the last two 429, and the provider must have received exactly three requests."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=3)
        responses: Final = [chat(gateway, model, key) for _ in range(5)]
        assert statuses(responses) == [200, 200, 200, 429, 429], [response.text for response in responses]
        assert len(posts(wire)) == 3


def test_the_rpm_refusal_body_names_the_hashed_key_the_limit_and_the_reset_time(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends a second request in the same window. The 429 body must be
    ``{"error": {"message": "Rate limit exceeded for api_key: <sha256 of key>. Limit type: requests. Current
    limit: 1, Remaining: 0. Limit resets at: <YYYY-MM-DD HH:MM:SS UTC>", "type": "throttling_error", "param":
    null, "code": "429"}}``. Client SDKs and dashboards parse this text, so a rewrite must keep it byte for byte."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        refused: Final = chat(gateway, model, key)
        assert_limited(refused, "api_key", hashed(key), "requests", 1)
        assert set(object_value(refused.json())) == {"error"}, refused.text
        assert set(object_value(refused.json()["error"])) == {"message", "type", "param", "code"}, refused.text


def test_the_rpm_refusal_never_echoes_the_raw_key(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is refused on its second request. Neither the body nor any response header may
    contain the secret key string, only its SHA-256 hash."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        refused: Final = chat(gateway, model, key)
        assert refused.status_code == 429, refused.text
        assert key not in refused.text
        assert all(key not in value for value in refused.headers.values()), dict(refused.headers)
        assert hashed(key) in refused.text


def test_the_rpm_refusal_carries_retry_after_rate_limit_type_and_reset_at_headers(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is refused on its second request. The 429 must carry ``retry-after: 60`` (the
    window length, not the time left in it), ``rate_limit_type: requests`` and ``reset_at`` holding the same
    ``YYYY-MM-DD HH:MM:SS UTC`` time the body ends with."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        refused: Final = chat(gateway, model, key)
        assert refused.status_code == 429, refused.text
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        assert refused.headers["rate_limit_type"] == "requests"
        reset_at: Final = refused.headers["reset_at"]
        assert RESET_AT.match(reset_at), reset_at
        assert error_message(refused) == limit_message("api_key", hashed(key), "requests", 1) + reset_at


def test_the_reset_time_in_an_rpm_refusal_is_one_window_after_the_refusal(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is refused. ``reset_at`` must be the moment of the refusal plus 60 seconds, give or
    take the request's own duration, because the limiter reports ``now + window`` and not the true window end."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        before: Final = time.time()
        refused: Final = chat(gateway, model, key)
        after: Final = time.time()
        assert refused.status_code == 429, refused.text
        reset: Final = (
            datetime.strptime(refused.headers["reset_at"], "%Y-%m-%d %H:%M:%S UTC")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        assert before + WINDOW_SECONDS - 2 <= reset <= after + WINDOW_SECONDS + 1, (before, reset, after)


def test_the_rpm_refusal_still_carries_the_call_id_and_key_limit_headers(gateway: Gateway) -> None:
    """A key with rpm_limit=1 and tpm_limit=5000 is refused for requests. The 429 must still carry a non-empty
    ``x-litellm-call-id`` and the configured limits as ``x-litellm-key-rpm-limit: 1`` and
    ``x-litellm-key-tpm-limit: 5000``, the same headers a served response has."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, tpm_limit=5000)
        assert chat(gateway, model, key, max_tokens=5).status_code == 200
        refused: Final = chat(gateway, model, key, max_tokens=5)
        assert refused.status_code == 429, refused.text
        assert refused.headers["x-litellm-call-id"]
        assert refused.headers["x-litellm-key-rpm-limit"] == "1"
        assert refused.headers["x-litellm-key-tpm-limit"] == "5000"


def test_a_served_response_reports_the_key_request_limit_and_what_is_left(gateway: Gateway) -> None:
    """A key with rpm_limit=5 sends three requests. Each 200 must carry
    ``x-ratelimit-api_key-limit-requests: 5`` and ``x-ratelimit-api_key-remaining-requests`` counting down 4, 3,
    2, and no token headers because the key has no tpm limit."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        responses: Final = [chat(gateway, model, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 200], [response.text for response in responses]
        assert [scope_headers(response, "api_key") for response in responses] == [
            {"limit-requests": "5", "remaining-requests": "4"},
            {"limit-requests": "5", "remaining-requests": "3"},
            {"limit-requests": "5", "remaining-requests": "2"},
        ]


def test_a_served_response_reports_the_configured_key_limits_as_x_litellm_key_headers(gateway: Gateway) -> None:
    """A key with rpm_limit=7 and tpm_limit=9000 gets a 200. It must carry ``x-litellm-key-rpm-limit: 7`` and
    ``x-litellm-key-tpm-limit: 9000``. These are the configured limits, not the remaining counts."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=7, tpm_limit=9000)
        served: Final = chat(gateway, model, key, max_tokens=5)
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-key-rpm-limit"] == "7"
        assert served.headers["x-litellm-key-tpm-limit"] == "9000"


def test_a_key_without_limits_gets_no_api_key_ratelimit_headers(gateway: Gateway) -> None:
    """A key with no rpm, tpm or parallel limit gets a 200. It must carry no ``x-ratelimit-api_key-*`` header
    and no ``x-litellm-key-rpm-limit`` or ``x-litellm-key-tpm-limit`` header, because unset limits are omitted
    rather than sent as the string ``None``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {}
        assert "x-litellm-key-rpm-limit" not in served.headers
        assert "x-litellm-key-tpm-limit" not in served.headers


def test_an_rpm_limit_of_zero_refuses_every_request(gateway: Gateway) -> None:
    """A key has rpm_limit=0. Its very first request must be refused with the 429 for ``api_key`` requests with
    ``Current limit: 0, Remaining: 0`` and the provider must receive nothing."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=0)
        assert_limited(chat(gateway, model, key), "api_key", hashed(key), "requests", 0)
        assert posts(wire) == ()


def test_two_keys_with_the_same_rpm_limit_are_counted_separately(gateway: Gateway) -> None:
    """Two keys each have rpm_limit=1. After the first key used its one request and is refused, the second key
    must still be served, and its refusal must name its own hash."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        first: Final = scenario.key(models=[model], rpm_limit=1)
        second: Final = scenario.key(models=[model], rpm_limit=1)
        assert statuses([chat(gateway, model, first), chat(gateway, model, first)]) == [200, 429]
        assert chat(gateway, model, second).status_code == 200
        assert_limited(chat(gateway, model, second), "api_key", hashed(second), "requests", 1)
        assert len(posts(wire)) == 2


def test_the_master_key_is_not_limited_by_another_keys_rpm_limit(gateway: Gateway) -> None:
    """A virtual key with rpm_limit=1 is exhausted. The master key calling the same model must keep getting 200
    and its responses must carry no ``x-ratelimit-api_key-*`` headers."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert statuses([chat(gateway, model, key), chat(gateway, model, key)]) == [200, 429]
        admin: Final = [chat(gateway, model, gateway.key) for _ in range(3)]
        assert statuses(admin) == [200, 200, 200], [response.text for response in admin]
        assert all(scope_headers(response, "api_key") == {} for response in admin)


def test_the_request_counter_stops_one_past_the_limit_on_a_single_instance(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends four requests to one proxy instance. The Redis counter
    ``{api_key:<hash>}:requests`` must read 2: the first refusal increments it past the limit, and later
    refusals are answered from the instance's local mirror without touching Redis. A rewrite that increments on
    every refusal, or never past the limit, changes what a second instance sees."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        responses: Final = [chat(gateway, model, key) for _ in range(4)]
        assert statuses(responses) == [200, 429, 429, 429]
        assert redis_integer(redis, counter_key("api_key", hashed(key), "requests")) == 2
        assert len(posts(wire)) == 1


def test_the_request_counter_and_its_window_live_in_redis_with_the_window_ttl(gateway: Gateway) -> None:
    """A key with rpm_limit=5 sends one request. Redis must then hold ``{api_key:<hash>}:requests`` equal to 1
    and ``{api_key:<hash>}:window`` equal to the epoch second the window opened, both expiring within 60
    seconds. Other proxy instances and the Rust limiter read these exact keys."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        before: Final = int(time.time())
        assert chat(gateway, model, key).status_code == 200
        after: Final = int(time.time())
        requests_key: Final = counter_key("api_key", hashed(key), "requests")
        window_key: Final = counter_key("api_key", hashed(key), "window")
        assert redis_integer(redis, requests_key) == 1
        opened: Final = redis_integer(redis, window_key)
        assert opened is not None and before - 1 <= opened <= after + 1, (before, opened, after)
        assert 0 < int(redis.ttl(requests_key)) <= WINDOW_SECONDS
        assert 0 < int(redis.ttl(window_key)) <= WINDOW_SECONDS


def test_a_key_with_only_an_rpm_limit_reports_only_the_two_request_headers(gateway: Gateway) -> None:
    """A key with rpm_limit=5 and no tpm or parallel limit is served once. The only ``x-ratelimit-*`` headers on
    the response must be ``x-ratelimit-api_key-limit-requests`` and ``x-ratelimit-api_key-remaining-requests``:
    no token header, no parallel header and no header for a scope that has no limit."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert ratelimit_header_names(served) == {
            "x-ratelimit-api_key-limit-requests",
            "x-ratelimit-api_key-remaining-requests",
        }


def test_a_streaming_request_over_the_rpm_limit_gets_a_json_429_and_no_event_stream(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends two ``stream: true`` chat requests. The first must stream; the second must be
    a plain JSON 429 with ``content-type: application/json`` and the usual error body, not a 200 event stream
    carrying an error frame."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        streamed: Final = chat(gateway, model, key, stream=True)
        assert streamed.status_code == 200, streamed.text
        assert streamed.headers["content-type"].startswith("text/event-stream")
        refused: Final = chat(gateway, model, key, stream=True)
        assert refused.headers["content-type"].startswith("application/json"), refused.headers
        assert_limited(refused, "api_key", hashed(key), "requests", 1)
        assert len(posts(wire)) == 1


def test_a_streaming_response_carries_no_key_rate_limit_headers_but_still_counts(gateway: Gateway) -> None:
    """A key with rpm_limit=4 sends one ``stream: true`` request and then one plain request.

    Pinned current behaviour: the limiter writes its ``x-ratelimit-*`` values into the response's hidden
    params in ``async_post_call_success_hook``, which runs after a stream's headers are already sent, and for
    streams it only mirrors them into the logging payload (``async_logging_hook`` in
    litellm/proxy/hooks/parallel_request_limiter_v3.py). So the 200 event stream carries no
    ``x-ratelimit-api_key-*`` header at all. The stream still uses one request of the window, so the plain
    request that follows must report ``limit-requests: 4`` and ``remaining-requests: 2``.
    """
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=4)
        streamed: Final = chat(gateway, model, key, stream=True)
        assert streamed.status_code == 200, streamed.text
        assert scope_headers(streamed, "api_key") == {}
        plain: Final = chat(gateway, model, key)
        assert plain.status_code == 200, plain.text
        assert scope_headers(plain, "api_key") == {"limit-requests": "4", "remaining-requests": "2"}


def test_a_request_the_provider_fails_still_uses_one_request_of_the_window(gateway: Gateway) -> None:
    """A key has rpm_limit=2 and the provider answers 400 to everything. The first two requests must relay the
    provider's 400 and the third must be the proxy's own 429, because a request counts when it is admitted, not
    when it succeeds."""
    with wire_server(provider(status=400)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=2)
        responses: Final = [chat(gateway, model, key) for _ in range(3)]
        assert statuses(responses) == [400, 400, 429], [response.text for response in responses]
        assert_limited(responses[2], "api_key", hashed(key), "requests", 2)
        assert len(posts(wire)) == 2


def test_a_request_denied_for_model_access_does_not_use_the_rpm_window(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is restricted to one model and first asks for a model it may not use. That request
    is refused by authorization (401 or 403), not by the limiter, and the key's one request must still be
    available afterwards, so access checks run before the window is charged."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        denied: Final = chat(gateway, other, key)
        assert denied.status_code in (401, 403), denied.text
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}


def test_management_calls_with_the_key_do_not_use_its_rpm_window(gateway: Gateway) -> None:
    """A key with rpm_limit=1 calls ``GET /key/info`` and ``GET /v1/models`` several times and then sends one chat
    request. The chat request must be served with ``remaining-requests: 0``, because only LLM routes are charged
    to the window. UNVERIFIED: assumed from the limiter being a pre-call hook of LLM routes only; check
    ``ProxyLogging.pre_call_hook`` callers if this fails."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        for _ in range(3):
            assert gateway.request("GET", "/key/info", key=key).status_code == 200
            assert gateway.request("GET", "/v1/models", key=key).status_code == 200
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}
