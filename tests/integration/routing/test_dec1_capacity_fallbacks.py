"""Model capacity and fallbacks: what happens when a model group's window is full or saturated.

A model capacity refusal is a ``ProxyRateLimitError`` raised from the proxy pre-call hook, and
``_pre_call_with_fallbacks`` catches it: when the request has fallbacks (``router_settings.fallbacks`` in the config,
or the key's or team's own ``router_settings``) and the body does not set ``disable_fallbacks``, each fallback group
is admitted through the same hooks in turn. The first one that admits serves the request, and the response carries
``x-litellm-attempted-fallbacks: <n>`` and ``x-litellm-fallback-reason: rate_limit:<descriptor_key>``. The capacity
refusal is raised without a ``descriptor_key``, so its reason reads ``rate_limit:unknown``. When no fallback admits,
the caller gets the original model's refusal.

``router_settings.fallback_on_saturation: true`` on a key or team (#51439) goes further: once the requested model is
at or above ``saturation_threshold``, the walk runs before the model refuses anything, with reason
``rate_limit:model_saturation_fallback``. If no fallback admits, the request is served on the requested model,
which still has room. A client cannot turn this on through ``router_settings_override`` in the body.

``routing/test_fallback_on_saturation.py`` pins the team opt-in, and ``test_per_model_rate_limit_fallbacks.py`` pins
that a capacity refusal falls back at all. This file pins the rest of the walk for capacity refusals.

Read from ``litellm/proxy/common_request_processing.py`` (``_pre_call_with_fallbacks``,
``_fallback_refusal_ends_the_walk``), ``litellm/proxy/common_utils/proxy_rate_limit_error.py``
(``RateLimitFallback``, ``rate_limit_fallback_headers``, ``drop_client_saturation_opt_in``) and
``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` (``_admit``, ``_saturation_fallback_error``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Wire, wire_server
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    capacity_message,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    model_counter,
    posts,
    provider,
    respond,
    scope_headers,
    settled,
    shared_redis,
)

ATTEMPTED: Final = "x-litellm-attempted-fallbacks"
REASON: Final = "x-litellm-fallback-reason"
CAPACITY_REASON: Final = "rate_limit:unknown"
SATURATION_REASON: Final = "rate_limit:model_saturation_fallback"


def _drain(*wires: Wire) -> None:
    for wire in wires:
        wire.drain()


def _fallbacks(primary: str, *targets: str) -> dict[str, JsonValue]:
    return {"fallbacks": [{primary: list(targets)}]}


def test_a_capacity_refusal_is_served_by_the_configured_fallback_group(gateway: Gateway, tmp_path: Path) -> None:
    """``primary`` has ``rpm: 1`` and ``router_settings.fallbacks: [{primary: [spare]}]``. After one served request
    the second request for ``primary`` must be answered 200 by ``spare``'s provider with
    ``x-litellm-attempted-fallbacks: 1``, and ``primary``'s provider must not be called again."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url, rpm=100)],
            router_settings=_fallbacks(primary, spare),
        ) as candidate,
    ):
        assert chat(candidate, primary, candidate.key).status_code == 200
        moved: Final = chat(candidate, primary, candidate.key)
        assert moved.status_code == 200, moved.text
        assert moved.headers.get(ATTEMPTED) == "1", dict(moved.headers)
        assert (len(posts(first)), len(posts(second))) == (1, 1)


def test_a_capacity_fallback_names_an_unknown_rate_limit_as_its_reason(gateway: Gateway, tmp_path: Path) -> None:
    """The capacity refusal of the dynamic limiter carries no ``descriptor_key``, so a request it sends to a fallback
    is answered with ``x-litellm-fallback-reason: rate_limit:unknown``. Pinned so a rewrite that starts naming the
    descriptor (for example ``rate_limit:model_saturation_check``) is a deliberate, visible change for dashboards."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url, rpm=100)],
            router_settings=_fallbacks(primary, spare),
        ) as candidate,
    ):
        assert chat(candidate, primary, candidate.key).status_code == 200
        moved: Final = chat(candidate, primary, candidate.key)
        assert moved.status_code == 200, moved.text
        assert moved.headers.get(REASON) == CAPACITY_REASON, dict(moved.headers)


def test_a_capacity_fallback_is_counted_in_the_fallback_groups_window(gateway: Gateway, tmp_path: Path) -> None:
    """The request ``spare`` serves for a full ``primary`` reports ``spare``'s window: model-wide ``limit-requests:
    100`` and ``remaining-requests: 99``. The refused admission on ``primary`` used nothing, so a later direct
    request to ``spare`` reports 98."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url, rpm=100)],
            router_settings=_fallbacks(primary, spare),
        ) as candidate,
    ):
        assert chat(candidate, primary, candidate.key).status_code == 200
        moved: Final = chat(candidate, primary, candidate.key)
        direct: Final = chat(candidate, spare, candidate.key)
        assert scope_headers(moved, MODEL_SCOPE) == {"limit-requests": "100", "remaining-requests": "99"}
        assert scope_headers(direct, MODEL_SCOPE) == {"limit-requests": "100", "remaining-requests": "98"}


def test_when_the_fallback_is_also_full_the_caller_gets_the_primarys_refusal(gateway: Gateway, tmp_path: Path) -> None:
    """``primary`` and ``spare`` both have ``rpm: 1`` and each has served one request. A request for ``primary`` is
    refused with ``primary``'s own capacity text (``Model capacity reached for <primary>. ... Model RPM: 1``), not
    ``spare``'s, and neither provider is called."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url, rpm=1)],
            router_settings=_fallbacks(primary, spare),
        ) as candidate,
    ):
        assert chat(candidate, spare, candidate.key).status_code == 200
        assert chat(candidate, primary, candidate.key).status_code == 200
        _drain(first, second)
        refused: Final = chat(candidate, primary, candidate.key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == capacity_message(primary, None, "requests", rpm=1)
        assert ATTEMPTED not in refused.headers
        assert (posts(first), posts(second)) == ((), ())


def test_the_walk_skips_a_full_fallback_and_counts_both_attempts(gateway: Gateway, tmp_path: Path) -> None:
    """``primary`` falls back to ``[busy, spare]``. ``primary`` and ``busy`` (both ``rpm: 1``) are full. A request for
    ``primary`` is served by ``spare`` with ``x-litellm-attempted-fallbacks: 2``, and ``busy``'s provider is not
    called."""
    primary, busy, spare = group_name("primary"), group_name("busy"), group_name("spare")
    with wire_server(provider()) as first, wire_server(provider()) as second, wire_server(provider()) as third:
        with capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(busy, second.url, rpm=1), entry(spare, third.url)],
            router_settings=_fallbacks(primary, busy, spare),
        ) as candidate:
            assert chat(candidate, primary, candidate.key).status_code == 200
            assert chat(candidate, busy, candidate.key).status_code == 200
            _drain(first, second)
            moved: Final = chat(candidate, primary, candidate.key)
            assert moved.status_code == 200, moved.text
            assert moved.headers.get(ATTEMPTED) == "2", dict(moved.headers)
            assert (len(posts(first)), len(posts(second)), len(posts(third))) == (0, 0, 1)


def test_disable_fallbacks_in_the_body_answers_the_capacity_refusal_directly(gateway: Gateway, tmp_path: Path) -> None:
    """``primary`` is full and has a configured fallback. A request with ``"disable_fallbacks": true`` in the body is
    refused with ``primary``'s capacity text and the fallback's provider is not called."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url)],
            router_settings=_fallbacks(primary, spare),
        ) as candidate,
    ):
        assert chat(candidate, primary, candidate.key).status_code == 200
        refused: Final = chat(candidate, primary, candidate.key, disable_fallbacks=True)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == capacity_message(primary, None, "requests", rpm=1)
        assert posts(second) == ()


def test_a_keys_own_router_settings_fallbacks_serve_a_capacity_refusal(gateway: Gateway, tmp_path: Path) -> None:
    """The config has no fallbacks. A key has ``router_settings: {"fallbacks": [{primary: [spare]}]}``. Once
    ``primary`` is full, that key's request is served by ``spare``, while the master key without fallbacks gets the
    capacity 429."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url)]
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(router_settings=_fallbacks(primary, spare))
        assert chat(candidate, primary, candidate.key).status_code == 200
        plain: Final = chat(candidate, primary, candidate.key)
        moved: Final = chat(candidate, primary, key)
        assert plain.status_code == 429, plain.text
        assert moved.status_code == 200, moved.text
        assert moved.headers.get(ATTEMPTED) == "1"
        assert len(posts(second)) == 1


def test_a_streamed_request_at_capacity_is_streamed_by_the_fallback(gateway: Gateway, tmp_path: Path) -> None:
    """``primary`` is full and falls back to ``spare``. A ``stream: true`` request for ``primary`` must be answered 200
    with an SSE body that ends in ``data: [DONE]``, produced by ``spare``'s provider, and carry
    ``x-litellm-attempted-fallbacks: 1`` on the stream's response headers."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url)],
            router_settings=_fallbacks(primary, spare),
        ) as candidate,
    ):
        assert chat(candidate, primary, candidate.key).status_code == 200
        moved: Final = chat(candidate, primary, candidate.key, stream=True)
        assert moved.status_code == 200, moved.text
        assert moved.text.rstrip().endswith("data: [DONE]"), moved.text
        assert moved.headers.get(ATTEMPTED) == "1", dict(moved.headers)
        assert len(posts(second)) == 1


def test_a_responses_api_request_at_capacity_is_served_by_the_fallback(gateway: Gateway, tmp_path: Path) -> None:
    """``primary`` is full and falls back to ``spare``. A ``POST /v1/responses`` for ``primary`` must be answered 200
    from ``spare``'s provider (a ``response`` object) with ``x-litellm-attempted-fallbacks: 1``. The Responses route
    shares the pre-call walk with chat, and the Rust port of the limiter is wired per route."""
    primary, spare = group_name("primary"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1), entry(spare, second.url)],
            router_settings=_fallbacks(primary, spare),
        ) as candidate,
    ):
        assert respond(candidate, primary, candidate.key).status_code == 200
        moved: Final = respond(candidate, primary, candidate.key)
        assert moved.status_code == 200, moved.text
        assert json.loads(moved.content)["object"] == "response"
        assert moved.headers.get(ATTEMPTED) == "1", dict(moved.headers)
        assert [request.target for request in second.drain()] == ["/v1/responses"]


def test_a_capacity_fallbacks_tokens_settle_on_the_fallback_groups_counter(gateway: Gateway, tmp_path: Path) -> None:
    """Shared Redis. ``primary`` serves one request (18 tokens settle on ``{model_saturation_check:<primary>}:tokens``),
    then a full ``primary`` falls back to ``spare``. The 18 tokens of that request must settle on ``spare``'s
    tokens counter, and ``primary``'s counter must stay at 18 (#51082)."""
    primary, spare = group_name("primary"), group_name("spare")
    with wire_server(provider()) as first, wire_server(provider()) as second, shared_redis() as redis:
        with capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(primary, first.url, rpm=1, tpm=10_000), entry(spare, second.url, tpm=10_000)],
            router_settings=_fallbacks(primary, spare),
            shared=True,
        ) as candidate:
            assert chat(candidate, primary, candidate.key).status_code == 200
            assert settled(redis, model_counter(primary, "tokens"), 18) == 18
            assert chat(candidate, primary, candidate.key).status_code == 200
            assert settled(redis, model_counter(spare, "tokens"), 18) == 18
            assert settled(redis, model_counter(primary, "tokens"), 18) == 18


def test_below_saturation_an_opted_in_key_stays_on_the_requested_model(gateway: Gateway, tmp_path: Path) -> None:
    """A key has ``router_settings: {"fallbacks": [{preferred: [spare]}], "fallback_on_saturation": true}``.
    ``preferred`` has ``rpm: 4`` and no traffic, so it is at 0%. The key's request is served by ``preferred`` with no
    ``x-litellm-fallback-reason`` header."""
    preferred, spare = group_name("preferred"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(preferred, first.url, rpm=4), entry(spare, second.url, rpm=100)]
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(router_settings={**_fallbacks(preferred, spare), "fallback_on_saturation": True})
        served: Final = chat(candidate, preferred, key)
        assert served.status_code == 200, served.text
        assert REASON not in served.headers
        assert (len(posts(first)), len(posts(second))) == (1, 0)


def test_a_key_with_fallback_on_saturation_moves_to_its_fallback_at_the_threshold(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The opt-in on a key, not a team. ``preferred`` (``rpm: 4``) has served two requests from the master key, so it
    is at 50%. The opted-in key's request is served by ``spare`` with ``x-litellm-fallback-reason:
    rate_limit:model_saturation_fallback`` and ``x-litellm-attempted-fallbacks: 1``, although ``preferred`` still had
    room."""
    preferred, spare = group_name("preferred"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(preferred, first.url, rpm=4), entry(spare, second.url, rpm=100)]
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(router_settings={**_fallbacks(preferred, spare), "fallback_on_saturation": True})
        assert [chat(candidate, preferred, candidate.key).status_code for _ in range(2)] == [200, 200]
        moved: Final = chat(candidate, preferred, key)
        assert moved.status_code == 200, moved.text
        assert (moved.headers.get(REASON), moved.headers.get(ATTEMPTED)) == (SATURATION_REASON, "1")
        assert len(posts(second)) == 1


def test_when_no_fallback_admits_a_saturated_request_it_is_served_on_the_requested_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An opted-in key, ``preferred`` (``rpm: 4``) at 50% and its only fallback ``spare`` (``rpm: 1``) full. The walk
    finds no fallback below the threshold, so the request goes back to ``preferred`` and is served there: 200 from
    ``preferred``'s provider, no ``x-litellm-fallback-reason`` header, and model-wide ``remaining-requests: 1``."""
    preferred, spare = group_name("preferred"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(preferred, first.url, rpm=4), entry(spare, second.url, rpm=1)]
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(router_settings={**_fallbacks(preferred, spare), "fallback_on_saturation": True})
        assert chat(candidate, spare, candidate.key).status_code == 200
        assert [chat(candidate, preferred, candidate.key).status_code for _ in range(2)] == [200, 200]
        _drain(first, second)
        kept: Final = chat(candidate, preferred, key)
        assert kept.status_code == 200, kept.text
        assert REASON not in kept.headers
        assert scope_headers(kept, MODEL_SCOPE)["remaining-requests"] == "1"
        assert (len(posts(first)), len(posts(second))) == (1, 0)


def test_a_client_cannot_opt_into_fallback_on_saturation_through_the_request_body(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The config has ``fallbacks: [{preferred: [spare]}]`` and ``preferred`` (``rpm: 4``) is at 50%. The master key,
    whose saved router settings do not opt in, sends ``router_settings_override: {"fallback_on_saturation": true}``.
    The field is dropped, so the request is served by ``preferred`` with no fallback headers."""
    preferred, spare = group_name("preferred"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(preferred, first.url, rpm=4), entry(spare, second.url, rpm=100)],
            router_settings=_fallbacks(preferred, spare),
        ) as candidate,
    ):
        assert [chat(candidate, preferred, candidate.key).status_code for _ in range(2)] == [200, 200]
        _drain(first)
        kept: Final = chat(
            candidate, preferred, candidate.key, router_settings_override={"fallback_on_saturation": True}
        )
        assert kept.status_code == 200, kept.text
        assert REASON not in kept.headers and ATTEMPTED not in kept.headers
        assert (len(posts(first)), len(posts(second))) == (1, 0)


def test_an_opted_in_key_at_full_capacity_without_room_anywhere_gets_the_capacity_refusal(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An opted-in key, ``preferred`` (``rpm: 2``) full and ``spare`` (``rpm: 1``) full. The walk admits nothing and the
    request goes back to ``preferred``, where the model-wide window refuses it with ``preferred``'s capacity text.
    UNVERIFIED: read from the exhausted branch of ``_pre_call_with_fallbacks``, which re-runs the pre-call hooks on
    the requested model."""
    preferred, spare = group_name("preferred"), group_name("spare")
    with (
        wire_server(provider()) as first,
        wire_server(provider()) as second,
        capacity_proxy(
            gateway, tmp_path, model_list=[entry(preferred, first.url, rpm=2), entry(spare, second.url, rpm=1)]
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(router_settings={**_fallbacks(preferred, spare), "fallback_on_saturation": True})
        assert chat(candidate, spare, candidate.key).status_code == 200
        assert [chat(candidate, preferred, candidate.key).status_code for _ in range(2)] == [200, 200]
        refused: Final = chat(candidate, preferred, key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == capacity_message(preferred, None, "requests", rpm=2)
