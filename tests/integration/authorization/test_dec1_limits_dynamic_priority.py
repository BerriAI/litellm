"""Dynamic, priority-aware rate limiting (``callbacks: ["dynamic_rate_limiter_v3"]``, ``priority_reservation``).

With the ``dynamic_rate_limiter_v3`` callback a model group's ``rpm``/``tpm`` become a capacity shared by every
caller of that model, counted per 60 second window in the same store as key limits. The contract:

- Total capacity is always enforced. The request that would exceed the model's ``rpm`` is refused with 429 and
  the message ``Model capacity reached for <model>. Priority: <priority>, Rate limit type: requests, Model
  TPM: <tpm or 'not configured'>, Model RPM: <rpm>, Remaining: 0``, with headers ``retry-after: 60``,
  ``rate_limit_type: requests`` and ``x-litellm-priority``.
- A caller's priority is ``metadata.priority`` of its team if set, otherwise of its key, otherwise none, which
  is reported as ``default``.
- Served responses carry ``x-litellm-priority``, ``x-litellm-rate-limiter-version: v3`` and
  ``x-ratelimit-model_saturation_check-limit-requests`` / ``-remaining-requests`` for the model-wide window.
- A model with no ``rpm`` and no ``tpm`` is not limited by this callback.

``routing/test_priority_rate_limit_headers.py`` already pins the ``x-ratelimit-priority_model-*`` headers of a
reserved priority on ``/v1/messages``. This file pins capacity enforcement, the refusal and priority lookup.

Read from ``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` (``_check_rate_limits``,
``_get_priority_from_user_api_key_dict``, ``async_post_call_success_hook``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, object_value
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    WINDOW_SECONDS,
    chat,
    error_message,
    error_of,
    limits_proxy,
    posts,
    provider,
    scope_headers,
    statuses,
)


@contextmanager
def _dynamic_proxy(
    gateway: Gateway, directory: Path, priority_reservation: dict[str, JsonValue] | None = None
) -> Iterator[Gateway]:
    settings: Final[dict[str, JsonValue]] = {
        "callbacks": ["dynamic_rate_limiter_v3"],
        **({"priority_reservation": priority_reservation} if priority_reservation is not None else {}),
    }
    with limits_proxy(gateway, directory, litellm_settings=settings) as candidate:
        yield candidate


def _capacity_message(model: str, priority: str | None, rpm: int) -> str:
    return (
        f"Model capacity reached for {model}. Priority: {priority}, Rate limit type: requests, "
        f"Model TPM: not configured, Model RPM: {rpm}, Remaining: 0"
    )


def test_requests_beyond_the_models_rpm_are_refused_with_the_capacity_message(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on and a model has ``rpm: 2``. A key with no limits and no priority sends three
    requests: 200, 200, 429. The 429 body must be OpenAI-shaped with ``error.type`` ``throttling_error``,
    ``error.code`` ``"429"`` and the message ``Model capacity reached for <model>. Priority: None, Rate limit
    type: requests, Model TPM: not configured, Model RPM: 2, Remaining: 0``, and the provider sees two calls."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=2)
        key: Final = scenario.key(models=[model])
        responses: Final = [chat(candidate, model, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        error: Final = error_of(responses[2])
        assert error["message"] == _capacity_message(model, None, 2)
        assert (error["type"], error["code"]) == ("throttling_error", "429")
        assert len(posts(wire)) == 2


def test_the_capacity_refusal_keeps_the_structured_detail_next_to_the_message(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter refuses a request for model capacity. Besides ``message`` the error object must carry
    ``provider_specific_fields`` equal to ``{"error": <the same message>}``, the limiter's original structured
    detail, so clients that read the old ``detail.error`` shape keep working."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=1)
        key: Final = scenario.key(models=[model])
        assert chat(candidate, model, key).status_code == 200
        refused: Final = chat(candidate, model, key)
        assert refused.status_code == 429, refused.text
        assert object_value(error_of(refused)["provider_specific_fields"]) == {
            "error": _capacity_message(model, None, 1)
        }


def test_the_capacity_refusal_carries_retry_after_limit_type_and_priority_headers(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The dynamic limiter refuses a request from a key without a priority. The 429 must carry ``retry-after:
    60``, ``rate_limit_type: requests`` and ``x-litellm-priority: default``."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=1)
        key: Final = scenario.key(models=[model])
        assert chat(candidate, model, key).status_code == 200
        refused: Final = chat(candidate, model, key)
        assert refused.status_code == 429, refused.text
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        assert refused.headers["rate_limit_type"] == "requests"
        assert refused.headers["x-litellm-priority"] == "default"
        assert len(posts(wire)) == 1


def test_model_capacity_is_shared_by_every_key_calling_the_model(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on and a model has ``rpm: 2``. Two unrelated keys each send one request and are
    served. A third request from either key must be refused with the capacity message, although that key has
    only made one request itself."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=2)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        assert statuses([chat(candidate, model, first), chat(candidate, model, second)]) == [200, 200]
        refused: Final = chat(candidate, model, first)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == _capacity_message(model, None, 2)


def test_a_served_response_reports_the_model_wide_window_and_the_limiter_version(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The dynamic limiter is on, a model has ``rpm: 40`` and a key without priority sends one request. The 200
    must carry ``x-litellm-rate-limiter-version: v3``, ``x-litellm-priority: default``,
    ``x-ratelimit-model_saturation_check-limit-requests: 40`` and ``-remaining-requests: 39``."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=40)
        served: Final = chat(candidate, model, scenario.key(models=[model]))
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-rate-limiter-version"] == "v3"
        assert served.headers["x-litellm-priority"] == "default"
        assert scope_headers(served, "model_saturation_check") == {
            "limit-requests": "40",
            "remaining-requests": "39",
        }


def test_a_model_without_rpm_or_tpm_is_not_limited_by_the_dynamic_limiter(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on and a model has neither ``rpm`` nor ``tpm``. Five requests must all be served
    and carry no ``x-ratelimit-model_saturation_check-*`` header."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        responses: Final = [chat(candidate, model, key) for _ in range(5)]
        assert statuses(responses) == [200] * 5, [response.text for response in responses]
        assert all(scope_headers(response, "model_saturation_check") == {} for response in responses)
        assert len(posts(wire)) == 5


def test_the_priority_in_key_metadata_is_reported_and_named_in_the_refusal(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on, a model has ``rpm: 1`` and a key has ``metadata: {"priority": "batch"}``. Its
    served response must carry ``x-litellm-priority: batch`` and its refusal must read ``Model capacity reached
    for <model>. Priority: batch, ...`` with ``x-litellm-priority: batch`` as well."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=1)
        key: Final = scenario.key(models=[model], metadata={"priority": "batch"})
        served: Final = chat(candidate, model, key)
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-priority"] == "batch"
        refused: Final = chat(candidate, model, key)
        assert refused.status_code == 429, refused.text
        assert error_message(refused) == _capacity_message(model, "batch", 1)
        assert refused.headers["x-litellm-priority"] == "batch"


def test_the_teams_priority_takes_precedence_over_the_keys(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on. A team has ``metadata: {"priority": "gold"}`` and its key has ``metadata:
    {"priority": "silver"}``. A served response must carry ``x-litellm-priority: gold``: team metadata is read
    first and the key's value is only a fallback."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=10)
        team: Final = scenario.team(models=[model], metadata={"priority": "gold"})
        key: Final = scenario.key(team_id=team, metadata={"priority": "silver"})
        served: Final = chat(candidate, model, key)
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-priority"] == "gold"


def test_a_reserved_priority_reports_its_share_of_the_model_rpm(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on with ``priority_reservation: {"premium": 0.5}`` and a model has ``rpm: 40``. A key
    with priority ``premium`` sends one chat request. The 200 must carry
    ``x-ratelimit-priority_model-limit-requests: 20`` and ``-remaining-requests: 19`` next to the model-wide
    ``x-ratelimit-model_saturation_check-limit-requests: 40``. The share is 0.5 with or without a license,
    because an unlicensed proxy falls back to the default priority weight, which is also 0.5."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path, {"premium": 0.5}) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=40)
        key: Final = scenario.key(models=[model], metadata={"priority": "premium"})
        served: Final = chat(candidate, model, key)
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-priority"] == "premium"
        assert scope_headers(served, "priority_model") == {"limit-requests": "20", "remaining-requests": "19"}
        assert scope_headers(served, "model_saturation_check") == {
            "limit-requests": "40",
            "remaining-requests": "39",
        }


def test_below_saturation_a_priority_may_borrow_beyond_its_reserved_share(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on with ``priority_reservation: {"premium": 0.5}`` and a model has ``rpm: 10``, so
    ``premium`` is reserved 5 requests. With no other traffic a ``premium`` key sends 7 requests. All 7 must be
    served: the model is below the 80% saturation threshold (7 of 10), where only total capacity is enforced.
    UNVERIFIED: assumes the saturation read is not ahead of the counters; see ``_check_model_saturation``."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path, {"premium": 0.5}) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=10)
        key: Final = scenario.key(models=[model], metadata={"priority": "premium"})
        responses: Final = [chat(candidate, model, key) for _ in range(7)]
        assert statuses(responses) == [200] * 7, [response.text for response in responses]
        assert len(posts(wire)) == 7


def test_the_key_limit_and_the_model_capacity_are_both_enforced(gateway: Gateway, tmp_path: Path) -> None:
    """The dynamic limiter is on, a model has ``rpm: 5`` and a key has rpm_limit=1. The key's second request must
    be refused by its own limit (``Rate limit exceeded for api_key``), and a different key must still be served
    on the model, because the model's capacity is far from used."""
    with (
        wire_server(provider()) as wire,
        _dynamic_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=5)
        limited: Final = scenario.key(models=[model], rpm_limit=1)
        other: Final = scenario.key(models=[model])
        assert chat(candidate, model, limited).status_code == 200
        refused: Final = chat(candidate, model, limited)
        assert refused.status_code == 429, refused.text
        assert error_message(refused).startswith("Rate limit exceeded for api_key: ")
        assert chat(candidate, model, other).status_code == 200
