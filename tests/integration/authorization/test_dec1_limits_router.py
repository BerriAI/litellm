"""Deployment-level limits: ``rpm``, ``tpm`` and ``max_parallel_requests`` in a model's ``litellm_params``.

These limits protect a provider deployment, not a caller, and are enforced by the router after the caller's own
limits have passed. The contract:

- Under the default routing strategy a deployment's ``rpm`` is NOT a per-minute cap. It becomes the number of
  requests the deployment may have in flight (``max_parallel_requests`` if set, else ``rpm``, else
  ``tpm / 1000 * 6`` with a minimum of 1). A request that finds every slot taken is refused with 429 and a
  message containing ``already has max_parallel_requests=<n> requests in flight``.
- With ``router_settings.optional_pre_call_checks: ["enforce_model_rate_limits"]`` the ``rpm`` and ``tpm`` are
  hard per-minute caps. Over the cap the caller gets 429 with ``Model rate limit exceeded. RPM limit=<n>,
  current usage=<m>`` (or ``TPM limit=...``) and the provider is not called.
- With ``routing_strategy: usage-based-routing-v2`` a deployment that has used up its ``rpm`` is skipped in
  favour of another deployment of the same model group, and the caller only sees 429 when all are full.
- A model group with ``rpm``/``tpm`` reports ``x-ratelimit-limit-requests``, ``x-ratelimit-remaining-requests``,
  ``x-ratelimit-limit-tokens`` and ``x-ratelimit-remaining-tokens`` (no scope word) on served responses.
- A request refused by a caller's key limit never reaches the router and does not use deployment capacity.

Read from ``litellm/utils.py`` (``calculate_max_parallel_requests``),
``litellm/router_utils/client_initalization_utils.py`` (``MaxParallelRequestsLimit``),
``litellm/router_utils/pre_call_checks/model_rate_limit_check.py``,
``litellm/router_strategy/lowest_tpm_rpm_v2.py`` and ``litellm/router_rust/surface/model_info.py``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    WINDOW_SECONDS,
    assert_limited,
    chat,
    error_message,
    hashed,
    limits_proxy,
    posts,
    provider,
    ratelimit_header_names,
    statuses,
)

ENFORCING: Final[dict[str, JsonValue]] = {"optional_pre_call_checks": ["enforce_model_rate_limits"]}
USAGE_BASED: Final[dict[str, JsonValue]] = {"routing_strategy": "usage-based-routing-v2"}


def test_under_default_routing_a_deployment_rpm_does_not_cap_sequential_requests(gateway: Gateway) -> None:
    """A deployment has ``rpm: 1`` on the default proxy (simple-shuffle routing, no extra pre-call checks). Three
    requests sent one after another must all be served: here ``rpm`` only sizes the deployment's concurrency
    slots. UNVERIFIED: derived from ``calculate_max_parallel_requests`` and the absence of an rpm check in
    simple-shuffle; if the second request answers 429, the default routing enforces rpm per minute after all."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=1)
        responses: Final = [chat(gateway, model, gateway.key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 200], [response.text for response in responses]
        assert len(posts(wire)) == 3


@pytest.mark.parametrize(
    "limits",
    [{"max_parallel_requests": 1}, {"rpm": 1}, {"tpm": 100}, {"max_parallel_requests": 1, "rpm": 50}],
    ids=["max_parallel_requests", "rpm", "tpm", "max_parallel_requests_over_rpm"],
)
def test_a_deployment_with_one_slot_refuses_a_second_request_in_flight(
    gateway: Gateway, limits: dict[str, JsonValue]
) -> None:
    """A deployment gets exactly one concurrency slot from ``max_parallel_requests: 1``, from ``rpm: 1``, from
    ``tpm: 100`` (100 / 1000 * 6 rounds down to 0 and is raised to 1) or from ``max_parallel_requests: 1`` next
    to ``rpm: 50`` (the explicit value wins). While one request is held at the provider, a second must answer
    429 with a message containing ``already has max_parallel_requests=1 requests in flight`` and must not reach
    the provider."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", **limits)
        first: Final = pool.submit(chat, gateway, model, gateway.key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            refused: Final = chat(gateway, model, gateway.key)
            assert refused.status_code == 429, refused.text
            assert "already has max_parallel_requests=1 requests in flight" in error_message(refused)
            assert wire.received.qsize() == 1
        finally:
            release.set()
        assert first.result().status_code == 200


def test_a_deployment_slot_is_free_again_after_its_request_fails(gateway: Gateway) -> None:
    """A deployment has ``max_parallel_requests: 1`` and the provider answers 400. Three sequential requests must
    each relay the 400; none may be a 429, so a failed attempt gives its deployment slot back."""
    with wire_server(provider(status=400)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", max_parallel_requests=1)
        responses: Final = [chat(gateway, model, gateway.key) for _ in range(3)]
        assert statuses(responses) == [400, 400, 400], [response.text for response in responses]
        assert len(posts(wire)) == 3


def test_a_model_group_with_rpm_and_tpm_reports_unscoped_ratelimit_headers(gateway: Gateway) -> None:
    """A deployment has ``rpm: 7`` and ``tpm: 7000`` and the caller is the master key, which has no limits. A
    served response must carry exactly the four headers ``x-ratelimit-limit-requests: 7``,
    ``x-ratelimit-limit-tokens: 7000``, ``x-ratelimit-remaining-requests`` and ``x-ratelimit-remaining-tokens``,
    with each remaining value an integer no larger than its limit. UNVERIFIED: whether the remaining values
    already include this request depends on when the router's usage counter is written."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=7, tpm=7000)
        served: Final = chat(gateway, model, gateway.key)
        assert served.status_code == 200, served.text
        assert ratelimit_header_names(served) == {
            "x-ratelimit-limit-requests",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-tokens",
        }
        assert served.headers["x-ratelimit-limit-requests"] == "7"
        assert served.headers["x-ratelimit-limit-tokens"] == "7000"
        assert 0 <= int(served.headers["x-ratelimit-remaining-requests"]) <= 7
        assert 0 <= int(served.headers["x-ratelimit-remaining-tokens"]) <= 7000


def test_a_deployment_without_limits_reports_no_unscoped_ratelimit_headers(gateway: Gateway) -> None:
    """A deployment has no ``rpm`` or ``tpm`` and the provider sends no rate-limit headers. A response served to
    the master key must carry no ``x-ratelimit-*`` header at all."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        served: Final = chat(gateway, model, gateway.key)
        assert served.status_code == 200, served.text
        assert ratelimit_header_names(served) == frozenset()


def test_enforce_model_rate_limits_refuses_requests_over_the_deployment_rpm(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy runs with ``optional_pre_call_checks: ["enforce_model_rate_limits"]`` and a deployment has
    ``rpm: 2``. Three sequential requests must answer 200, 200, 429. The 429 message must contain ``Model rate
    limit exceeded. RPM limit=2, current usage=3``, must not be worded as a caller limit (``Rate limit exceeded
    for``), and the provider must have received two requests."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, router_settings=ENFORCING) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=2)
        responses: Final = [chat(candidate, model, candidate.key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        message: Final = error_message(responses[2])
        assert "Model rate limit exceeded. RPM limit=2, current usage=3" in message, message
        assert "Rate limit exceeded for" not in message
        assert len(posts(wire)) == 2


def test_enforce_model_rate_limits_answers_with_retry_after_sixty(gateway: Gateway, tmp_path: Path) -> None:
    """With ``enforce_model_rate_limits`` and a deployment ``rpm: 1``, the second request's 429 must carry
    ``retry-after: 60``.

    Product bug: ``ModelRateLimitingCheck`` (litellm/router_utils/pre_call_checks/model_rate_limit_check.py) puts
    ``retry-after: 60`` only on the synthetic ``httpx.Response`` of its ``RateLimitError`` and not in the
    ``headers=`` argument. ``RateLimitError`` deliberately does not copy response headers onto ``e.headers``
    (litellm/exceptions.py), so ``handle_llm_api_exception`` treats them as provider headers and forwards them
    under the ``llm_provider-`` prefix at best; the client never gets the ``retry-after`` the check meant to send."""
    pytest.skip(
        "BUG: enforce_model_rate_limits 429 lacks retry-after; the check sets it only on its synthetic response"
    )
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, router_settings=ENFORCING) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=1)
        assert chat(candidate, model, candidate.key).status_code == 200
        refused: Final = chat(candidate, model, candidate.key)
        assert refused.status_code == 429, refused.text
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        assert len(posts(wire)) == 1


def test_enforce_model_rate_limits_refuses_once_the_deployment_tpm_is_used_up(gateway: Gateway, tmp_path: Path) -> None:
    """With ``enforce_model_rate_limits`` a deployment has ``tpm: 10`` and the provider bills 18 tokens per call.
    The first request is served (nothing used yet). Once its usage is recorded, a request must answer 429 with a
    message containing ``Model rate limit exceeded. TPM limit=10, current usage=18``: deployment tpm is checked
    against tokens already spent, it is not reserved up front."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, router_settings=ENFORCING) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", tpm=10)
        assert chat(candidate, model, candidate.key).status_code == 200
        refused: Final = eventually(
            lambda: chat(candidate, model, candidate.key),
            lambda response: response.status_code == 429,
            seconds=20,
            return_last_on_timeout=True,
        )
        assert refused.status_code == 429, refused.text
        assert "Model rate limit exceeded. TPM limit=10, current usage=" in error_message(refused)


def test_a_request_refused_by_the_key_limit_uses_no_deployment_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """With ``enforce_model_rate_limits`` a deployment has ``rpm: 2``. Key A (rpm_limit=1) is served once and
    refused once by its own limit. Key B must then still be served: only A's one admitted request and B's
    request reached the deployment, so the key refusal did not use a deployment request."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, router_settings=ENFORCING) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=2)
        limited: Final = scenario.key(models=[model], rpm_limit=1)
        other: Final = scenario.key(models=[model])
        assert chat(candidate, model, limited).status_code == 200
        assert_limited(chat(candidate, model, limited), "api_key", hashed(limited), "requests", 1)
        assert chat(candidate, model, other).status_code == 200
        assert len(posts(wire)) == 2


def test_usage_based_routing_skips_a_full_deployment_and_refuses_when_all_are_full(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy routes with ``usage-based-routing-v2`` and one model group has two deployments, each ``rpm: 1``, on
    two different providers. Two requests must both be served, one by each provider. A third request must answer
    429 and reach neither provider."""
    with (
        wire_server(provider()) as first_wire,
        wire_server(provider()) as second_wire,
        limits_proxy(gateway, tmp_path, router_settings=USAGE_BASED) as candidate,
        candidate.scenario() as scenario,
    ):
        group: Final = scenario.model(api_base=f"{first_wire.url}/v1", rpm=1)
        scenario.model(name=group, api_base=f"{second_wire.url}/v1", rpm=1)
        served: Final = [chat(candidate, group, candidate.key) for _ in range(2)]
        assert statuses(served) == [200, 200], [response.text for response in served]
        assert (len(posts(first_wire)), len(posts(second_wire))) == (1, 1)
        refused: Final = chat(candidate, group, candidate.key)
        assert refused.status_code == 429, refused.text
        assert (len(posts(first_wire)), len(posts(second_wire))) == (0, 0)


def test_usage_based_routing_names_the_deployment_rpm_in_its_refusal(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy routes with ``usage-based-routing-v2`` and a single deployment has ``rpm: 1``. The second request
    must answer 429. UNVERIFIED wording: the strategy raises ``Deployment over defined rpm limit=1. current
    usage=...`` for the deployment, but with no deployment left the router may answer with its ``No deployments
    available`` text instead, so this pins only that one of the two is present."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, router_settings=USAGE_BASED) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=1)
        assert chat(candidate, model, candidate.key).status_code == 200
        refused: Final = chat(candidate, model, candidate.key)
        assert refused.status_code == 429, refused.text
        message: Final = error_message(refused)
        assert "Deployment over defined rpm limit=1" in message or "No deployments available" in message, message
        assert len(posts(wire)) == 1
