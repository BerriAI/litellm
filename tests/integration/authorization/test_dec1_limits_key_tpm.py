"""Tokens-per-minute limits on a virtual key (``tpm_limit``): reservation, settlement and refusal.

The contract: before the provider is called the proxy reserves an estimate of the request against the key's
token window, and refuses with 429 when the window cannot hold the estimate. The estimate is input plus
output. Input is the prompt text length divided by 4 (a 40 character probe is 10 tokens). Output is the
caller's ``max_tokens`` (times ``n``) when given. When no output cap is given the output estimate is
``max(input, floor)`` where the floor is ``min(1024, max(1, tpm_limit // 4))``, and when that floor is below
1024 the proxy also writes it into the request as ``max_tokens`` so the provider cannot overshoot the window.

After the provider answers, the reservation is settled to the usage the provider reported (total tokens minus
cached prompt tokens by default, or only input or only output under ``general_settings.token_rate_limit_type``).
A failed provider call refunds the whole reservation. A refused request reserves nothing.

Served responses carry ``x-ratelimit-api_key-limit-tokens`` and ``x-ratelimit-api_key-remaining-tokens``, where
remaining is the limit minus everything reserved or settled in the window, this request's reservation included.
The counter is the Redis key ``{api_key:<hash>}:tokens``.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``async_pre_call_hook``,
``_estimate_input_and_output_tokens``, ``_apply_implicit_output_cap``, ``async_log_success_event``,
``async_log_failure_event``) and ``litellm/constants.py`` (``NO_MAX_TOKENS_TPM_RESERVATION_FRACTION`` = 0.25).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    DEFAULT_USAGE,
    WINDOW_SECONDS,
    Usage,
    assert_limited,
    chat,
    counter_key,
    hashed,
    limits_proxy,
    posts,
    probe,
    provider,
    redis_integer,
    scope_headers,
    shared_redis,
    statuses,
)


def _settled(key: str, expected: int) -> None:
    with shared_redis() as redis:
        name: Final = counter_key("api_key", hashed(key), "tokens")
        eventually(lambda: redis_integer(redis, name), lambda value: value == expected, seconds=20)


def test_a_served_response_reports_the_token_limit_and_what_the_reservation_left(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and sends a 40 character prompt with max_tokens=20. The reservation is 10 + 20 =
    30 tokens, so the 200 must carry ``x-ratelimit-api_key-limit-tokens: 1000`` and
    ``x-ratelimit-api_key-remaining-tokens: 970``, and no request headers because the key has no rpm limit."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "970"}


def test_a_request_whose_estimate_exceeds_the_whole_tpm_limit_is_never_admitted(gateway: Gateway) -> None:
    """A key has tpm_limit=20 and asks for max_tokens=50, an estimate of 60 tokens. Even against an empty window
    the request must be refused with the 429 for ``api_key`` tokens reading ``Current limit: 20, Remaining: 20``,
    with ``rate_limit_type: tokens`` and ``retry-after: 60``, and the provider must receive nothing."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=20)
        refused: Final = chat(gateway, model, key, max_tokens=50)
        assert_limited(refused, "api_key", hashed(key), "tokens", 20, remaining=20)
        assert refused.headers["rate_limit_type"] == "tokens"
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        assert posts(wire) == ()


def test_a_tpm_limit_of_zero_refuses_every_request(gateway: Gateway) -> None:
    """A key has tpm_limit=0. Its first request must be refused for ``api_key`` tokens with ``Current limit: 0,
    Remaining: 0`` and never reach the provider, because every request reserves at least one token."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=0)
        assert_limited(chat(gateway, model, key, max_tokens=1), "api_key", hashed(key), "tokens", 0)
        assert posts(wire) == ()


def test_a_refused_token_request_reserves_nothing(gateway: Gateway) -> None:
    """A key has tpm_limit=100. A request estimated at 510 tokens is refused, then a request estimated at 30 is
    served with ``remaining-tokens: 70``. The refusal must not have moved the token counter at all."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        assert_limited(chat(gateway, model, key, max_tokens=500), "api_key", hashed(key), "tokens", 100, 100)
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "100", "remaining-tokens": "70"}


def test_the_token_window_settles_to_the_usage_the_provider_reported(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and reserves 510 tokens for a request the provider bills at 11 + 7 = 18 tokens.
    After the response, the Redis counter ``{api_key:<hash>}:tokens`` must settle at exactly 18, releasing the
    unused 492. A rewrite that leaves the reservation in place would throttle callers 28 times too early."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert chat(gateway, model, key, max_tokens=500).status_code == 200
        _settled(key, DEFAULT_USAGE.total)
        assert len(posts(wire)) == 1


def test_capacity_released_by_settlement_admits_the_next_request(gateway: Gateway) -> None:
    """A key has tpm_limit=100. The first request reserves 70 tokens and is billed 18. Once the window has settled
    at 18 a second request reserving 70 must be served with ``remaining-tokens: 12`` (100 - 18 - 70), although
    two unsettled reservations of 70 would not fit."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        assert chat(gateway, model, key, max_tokens=60).status_code == 200
        _settled(key, DEFAULT_USAGE.total)
        second: Final = chat(gateway, model, key, max_tokens=60)
        assert second.status_code == 200, second.text
        assert scope_headers(second, "api_key") == {"limit-tokens": "100", "remaining-tokens": "12"}
        assert len(posts(wire)) == 2


def test_an_unsettled_reservation_refuses_a_second_request_and_reports_what_is_left(gateway: Gateway) -> None:
    """A key has tpm_limit=100. While a request that reserved 70 tokens is still waiting on the provider, a second
    request needing 70 must be refused for tokens with ``Current limit: 100, Remaining: 30``. After the first is
    released both are accounted: one provider call, one 200."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        first: Final = pool.submit(chat, gateway, model, key, max_tokens=60)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            refused: Final = chat(gateway, model, key, max_tokens=60)
            assert_limited(refused, "api_key", hashed(key), "tokens", 100, remaining=30)
        finally:
            release.set()
        assert first.result().status_code == 200
        assert len(posts(wire)) == 1


def test_a_failed_provider_call_refunds_its_whole_token_reservation(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and the provider answers 400. The request reserved 510 tokens, the caller gets the
    provider's 400, and the Redis token counter must return to 0. Failed calls must not eat the token budget."""
    with wire_server(provider(status=400)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        failed: Final = chat(gateway, model, key, max_tokens=500)
        assert failed.status_code == 400, failed.text
        _settled(key, 0)
        assert len(posts(wire)) == 1


def test_a_streamed_response_is_charged_the_usage_in_its_final_chunk(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and streams a chat completion whose last provider chunk reports 18 total tokens.
    After the stream has been read to the end the Redis token counter must settle at 18, the same as a
    non-streamed call."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        streamed: Final = chat(gateway, model, key, max_tokens=500, stream=True)
        assert streamed.status_code == 200, streamed.text
        assert "[DONE]" in streamed.text
        _settled(key, DEFAULT_USAGE.total)
        assert len(posts(wire)) == 1


def test_cached_prompt_tokens_are_not_charged_to_the_token_window(gateway: Gateway) -> None:
    """The provider reports 30 prompt tokens of which 20 were cached, and 7 completion tokens. With the default
    ``total`` accounting the key's window must settle at 37 - 20 = 17 tokens, because provider-cached prompt
    tokens do not count toward the limit."""
    with wire_server(provider(Usage(prompt=30, completion=7, cached=20))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert chat(gateway, model, key, max_tokens=100).status_code == 200
        _settled(key, 17)
        assert len(posts(wire)) == 1


def test_n_multiplies_the_output_part_of_the_reservation(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and sends max_tokens=10 with n=3. The reservation must be 10 input + 3 x 10
    output = 40 tokens, so the response carries ``remaining-tokens: 960``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, max_tokens=10, n=3)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "960"}


def test_max_completion_tokens_is_reserved_like_max_tokens(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and sends ``max_completion_tokens: 50`` with no ``max_tokens``. The reservation
    must be 10 + 50 = 60 tokens (``remaining-tokens: 940``) and the provider must receive the caller's
    ``max_completion_tokens`` with no ``max_tokens`` added by the proxy."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, max_completion_tokens=50)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "940"}
        (forwarded,) = posts(wire)
        assert forwarded.get("max_completion_tokens") == 50
        assert "max_tokens" not in forwarded


def test_a_small_tpm_limit_caps_an_uncapped_request_at_a_quarter_of_the_limit(gateway: Gateway) -> None:
    """A key has tpm_limit=100 and sends a request with no output cap. The proxy must forward it to the provider
    with ``max_tokens: 25`` (a quarter of the limit) and reserve 10 + 25 = 35 tokens (``remaining-tokens: 65``).
    A rewrite that drops the injected cap lets one unbounded answer blow through a small tenant limit."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "100", "remaining-tokens": "65"}
        (forwarded,) = posts(wire)
        assert forwarded["max_tokens"] == 25


def test_a_large_tpm_limit_leaves_an_uncapped_request_uncapped_and_reserves_1024_output_tokens(
    gateway: Gateway,
) -> None:
    """A key has tpm_limit=10000 and sends a request with no output cap. The provider must receive no
    ``max_tokens`` at all, and the reservation is 10 input + 1024 output (a quarter of the 4096 default) = 1034,
    so the response carries ``remaining-tokens: 8966``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000)
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "10000", "remaining-tokens": "8966"}
        (forwarded,) = posts(wire)
        assert "max_tokens" not in forwarded
        assert "max_completion_tokens" not in forwarded


def test_an_explicit_max_tokens_reaches_the_provider_unchanged_under_a_small_tpm_limit(gateway: Gateway) -> None:
    """A key has tpm_limit=100 and the caller sends max_tokens=40, above the quarter-of-limit floor of 25. The
    provider must receive ``max_tokens: 40`` exactly: the proxy only adds a cap when the caller gave none."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        served: Final = chat(gateway, model, key, max_tokens=40)
        assert served.status_code == 200, served.text
        (forwarded,) = posts(wire)
        assert forwarded["max_tokens"] == 40


def test_a_token_refusal_still_uses_one_request_of_the_rpm_window(gateway: Gateway) -> None:
    """A key has rpm_limit=10 and tpm_limit=100. Its first request is refused for tokens (estimate 510), its
    second is served. The served response must report ``remaining-requests: 8``: the request check runs before
    the token reservation, so the token refusal already used one request."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10, tpm_limit=100)
        assert_limited(chat(gateway, model, key, max_tokens=500), "api_key", hashed(key), "tokens", 100, 100)
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {
            "limit-requests": "10",
            "remaining-requests": "8",
            "limit-tokens": "100",
            "remaining-tokens": "70",
        }
        assert len(posts(wire)) == 1


def test_tokens_are_counted_for_a_key_before_it_has_a_tpm_limit(gateway: Gateway) -> None:
    """A key with only rpm_limit=10 is served once (18 tokens billed). The Redis counter
    ``{api_key:<hash>}:tokens`` must still settle at 18: usage is recorded for every scope of the request whether
    or not it has a token limit, so a limit added later sees the tokens already spent in the window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10)
        assert chat(gateway, model, key).status_code == 200
        _settled(key, DEFAULT_USAGE.total)
        assert len(posts(wire)) == 1


def test_a_response_served_from_the_response_cache_is_still_charged_tokens(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and sends the same prompt twice, so the second answer comes from the proxy's
    response cache and the provider is called once. Both answers must be 200 and the token window must settle at
    2 x 18 = 36. UNVERIFIED: the limiter has no cache-hit branch, so the cached response's stored usage is
    assumed to be charged again; check ``_build_success_event_pipeline_operations`` if the counter reads 18."""
    text: Final = probe()
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        responses: Final = [chat(gateway, model, key, text=text, max_tokens=20) for _ in range(2)]
        assert statuses(responses) == [200, 200], [response.text for response in responses]
        assert len(posts(wire)) == 1
        _settled(key, 2 * DEFAULT_USAGE.total)


@pytest.mark.parametrize(("accounting", "charged"), [("input", 11), ("output", 7), ("total", 18)])
def test_token_rate_limit_type_chooses_which_tokens_are_charged(
    gateway: Gateway, tmp_path: Path, accounting: str, charged: int
) -> None:
    """A proxy is started with ``general_settings.token_rate_limit_type`` set to input, output or total, and the
    provider bills 11 prompt and 7 completion tokens. The key's token window must settle at 11, 7 or 18
    respectively. Operators rely on this to mirror providers that only meter one side. The candidate runs on the
    shared Redis (``shared=True``) because the window is read back through ``shared_redis()``."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(
            gateway, tmp_path, general_settings={"token_rate_limit_type": accounting}, shared=True
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert chat(candidate, model, key, max_tokens=100).status_code == 200
        _settled(key, charged)
        assert len(posts(wire)) == 1


def test_an_unknown_token_rate_limit_type_falls_back_to_total(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy is started with ``general_settings.token_rate_limit_type: sideways``. It must boot, serve the
    request, and charge the full 18 total tokens, treating the unknown value as ``total``. The candidate runs on
    the shared Redis (``shared=True``) because the window is read back through ``shared_redis()``."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(
            gateway, tmp_path, general_settings={"token_rate_limit_type": "sideways"}, shared=True
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert chat(candidate, model, key, max_tokens=100).status_code == 200
        _settled(key, DEFAULT_USAGE.total)
        assert len(posts(wire)) == 1


def test_with_reservation_disabled_a_large_request_is_admitted_and_the_next_one_is_refused(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy runs with ``LITELLM_TPM_TOKEN_RESERVATION_ENABLED=false`` and a key has tpm_limit=20. A request
    with max_tokens=500 must be served (nothing is reserved up front) and billed 40 tokens. Once that usage has
    landed, the next request must be refused for ``api_key`` tokens with ``Current limit: 20, Remaining: 0``."""
    with (
        wire_server(provider(Usage(prompt=30, completion=10))) as wire,
        limits_proxy(gateway, tmp_path, environment={"LITELLM_TPM_TOKEN_RESERVATION_ENABLED": "false"}) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=20)
        assert chat(candidate, model, key, max_tokens=500).status_code == 200
        refused: Final = eventually(
            lambda: chat(candidate, model, key, max_tokens=500),
            lambda response: response.status_code == 429,
            seconds=20,
        )
        assert_limited(refused, "api_key", hashed(key), "tokens", 20)
        assert len(posts(wire)) >= 1
