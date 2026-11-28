"""Which big provider errors put a deployment in cooldown, with no `allowed_fails` setting.

The cooldown rules in `litellm/rust/crates/router-core/src/health.rs` read the status of the exception litellm
raised, not the provider's raw status. `is_cooldown_required` (`health.rs:88`) lets a 4xx cool a deployment
only for 401, 402, 404, 408 and 429, never for a connection error. Without `allowed_fails` the default rule
(`health.rs:437`) cools a deployment of a multi deployment group at once for a 429 or any non-retryable
status it allows, and for a 5xx or 408 only once more than half of at least five requests in the minute
failed.

Run through the provider mapping, that gives: every provider's throttle cools at once (Bedrock
`ThrottlingException` and `ModelNotReadyException`, Google `RESOURCE_EXHAUSTED`, Azure, Anthropic and both
OpenAI 429 codes). Bedrock's 403 `UnrecognizedClientException` (an `AuthenticationError`, 401) cools at once,
but its 403 `AccessDeniedException` (a `PermissionDeniedError`, 403) never does. A Vertex 403 is answered as
a 400 and never cools. Anthropic 529, Google's overload 503, Bedrock 500 and Azure 408 need the five
request failure rate. A Google 502 is an `APIConnectionError` and never cools at all.

One owned proxy runs with `num_retries: 0` and `cooldown_time: 60`. Each group pairs a failing deployment
with `weight: 1` and a healthy OpenAI sibling without weight, so the failing one takes every request until
it is in cooldown. The number of requests the failing deployment received shows when the cooldown began.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_reliab_support import (
    ANTHROPIC_RATE_TEXT,
    AZURE_TIMEOUT_TEXT,
    BEDROCK_ACCESS_TEXT,
    BEDROCK_INTERNAL_TEXT,
    BEDROCK_NOT_READY_TEXT,
    BEDROCK_THROTTLE_TEXT,
    BEDROCK_TOKEN_TEXT,
    BEDROCK_VALIDATION_TEXT,
    GOOGLE_BAD_GATEWAY_TEXT,
    GOOGLE_EXHAUSTED_TEXT,
    GOOGLE_INVALID_TEXT,
    GOOGLE_OVERLOADED_TEXT,
    GOOGLE_PERMISSION_TEXT,
    NO_DEPLOYMENTS,
    Rig,
    anthropic_error,
    anthropic_overloaded,
    azure_content_filter,
    azure_rate_limit,
    bedrock_deployment,
    bedrock_error,
    burst,
    chat,
    error_message,
    failing_then_healthy,
    google_error,
    openai_error,
    openai_rate_limit,
    reliab_rig,
)
from pydantic import JsonValue

from tests.integration._support.client import eventually
from tests.integration._support.wire import Reply

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {"num_retries": 0, "cooldown_time": 60}

# group -> (provider of the failing deployment, its fixed reply)
FAILING: Final[dict[str, tuple[str, Reply]]] = {
    "now-bedrock-throttle": ("bedrock", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT)),
    "now-bedrock-not-ready": ("bedrock", bedrock_error(429, "ModelNotReadyException", BEDROCK_NOT_READY_TEXT)),
    "now-vertex-exhausted": ("vertex", google_error(429, "RESOURCE_EXHAUSTED", GOOGLE_EXHAUSTED_TEXT)),
    "now-gemini-exhausted": ("gemini", google_error(429, "RESOURCE_EXHAUSTED", GOOGLE_EXHAUSTED_TEXT)),
    "now-azure-429": ("azure", azure_rate_limit()),
    "now-anthropic-429": ("anthropic", anthropic_error(429, "rate_limit_error", ANTHROPIC_RATE_TEXT)),
    "now-openai-quota": ("openai", openai_rate_limit("insufficient_quota")),
    "now-openai-rate": ("openai", openai_rate_limit("rate_limit_exceeded")),
    "now-bedrock-token": ("bedrock", bedrock_error(403, "UnrecognizedClientException", BEDROCK_TOKEN_TEXT)),
    "rate-anthropic-529": ("anthropic", anthropic_overloaded()),
    "rate-gemini-overloaded": ("gemini", google_error(503, "UNAVAILABLE", GOOGLE_OVERLOADED_TEXT)),
    "rate-bedrock-500": ("bedrock", bedrock_error(500, "InternalServerException", BEDROCK_INTERNAL_TEXT)),
    "rate-azure-408": ("azure", openai_error(408, AZURE_TIMEOUT_TEXT, kind="timeout")),
    "never-bedrock-access": ("bedrock", bedrock_error(403, "AccessDeniedException", BEDROCK_ACCESS_TEXT)),
    "never-vertex-permission": ("vertex", google_error(403, "PERMISSION_DENIED", GOOGLE_PERMISSION_TEXT)),
    "never-bedrock-validation": ("bedrock", bedrock_error(400, "ValidationException", BEDROCK_VALIDATION_TEXT)),
    "never-vertex-invalid": ("vertex", google_error(400, "INVALID_ARGUMENT", GOOGLE_INVALID_TEXT)),
    "never-azure-filter": ("azure", azure_content_filter()),
    "never-gemini-502": ("gemini", google_error(502, "UNAVAILABLE", GOOGLE_BAD_GATEWAY_TEXT)),
    "now-bedrock-short": ("bedrock", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT)),
}
SHORT_COOLDOWN_SECONDS: Final = 3


def _failing(group: str) -> str:
    return f"{group}-bad"


def _healthy(group: str) -> str:
    return f"{group}-ok"


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    entries: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: built once for the config
    for group, (provider, _) in FAILING.items():
        extra: dict[str, JsonValue] = {"cooldown_time": SHORT_COOLDOWN_SECONDS} if group == "now-bedrock-short" else {}
        entries.extend(failing_then_healthy(provider, group, _failing(group), _healthy(group), base, **extra))
    entries.append(bedrock_deployment("lone-bedrock-throttle", "lone-bedrock-throttle", base))
    entries.append(bedrock_deployment("lone-bedrock-token", "lone-bedrock-token", base))
    return entries


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with reliab_rig(tmp_path_factory.mktemp("dec1-reliab-cooldown-shapes"), _models, ROUTER_SETTINGS) as value:
        for group, (_, reply) in FAILING.items():
            value.peer.always(_failing(group), reply)
        value.peer.always("lone-bedrock-throttle", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT))
        value.peer.always("lone-bedrock-token", bedrock_error(403, "UnrecognizedClientException", BEDROCK_TOKEN_TEXT))
        yield value


@pytest.mark.parametrize(
    ("group", "status"),
    [
        ("now-bedrock-throttle", 429),
        ("now-bedrock-not-ready", 429),
        ("now-vertex-exhausted", 429),
        ("now-gemini-exhausted", 429),
        ("now-azure-429", 429),
        ("now-anthropic-429", 429),
        ("now-openai-quota", 429),
        ("now-openai-rate", 429),
    ],
)
def test_every_providers_throttle_cools_the_deployment_on_its_first_failure(rig: Rig, group: str, status: int) -> None:
    """The weighted deployment answers the provider's throttle; its OpenAI sibling is healthy.

    The first request returns HTTP 429 to the caller and the next four return 200 from the sibling. The
    failing deployment must have received exactly one request: a `RateLimitError` in a multi deployment group
    starts a 60 second cooldown at once (`health.rs:444`). OpenAI `insufficient_quota`, which will not clear
    within a minute, is cooled for the same 60 seconds as a per-minute limit.
    """
    assert burst(rig.gateway, group, 5) == (status, 200, 200, 200, 200)
    assert rig.peer.count(_failing(group)) == 1
    assert rig.peer.count(_healthy(group)) == 4


def test_a_bedrock_invalid_security_token_cools_the_deployment_at_once(rig: Rig) -> None:
    """The weighted Bedrock deployment answers 403 `UnrecognizedClientException` with `The security token
    included in the request is invalid.`.

    The text maps to `AuthenticationError`, status 401, which cools at once: statuses (401, 200, 200) and one
    request at the Bedrock deployment, although Bedrock itself answered 403.
    """
    assert burst(rig.gateway, "now-bedrock-token", 3) == (401, 200, 200)
    assert rig.peer.count(_failing("now-bedrock-token")) == 1


@pytest.mark.parametrize(
    ("group", "status"),
    [
        ("rate-anthropic-529", 503),
        ("rate-gemini-overloaded", 500),
        ("rate-bedrock-500", 503),
        ("rate-azure-408", 408),
    ],
)
def test_a_provider_overload_cools_only_after_five_straight_failures(rig: Rig, group: str, status: int) -> None:
    """The weighted deployment answers the provider's overload or timeout every time: Anthropic 529, Google's
    503 overload text, Bedrock 500 `InternalServerException`, Azure 408.

    These map to statuses 500, 500, 503 and 408, which cool only by the failure rate rule: requests one to
    five return the caller-facing status (503, 500, 503, 408) and requests six and seven return 200 from the
    sibling. The failing deployment receives exactly five requests.
    """
    assert burst(rig.gateway, group, 7) == (status, status, status, status, status, 200, 200)
    assert rig.peer.count(_failing(group)) == 5


@pytest.mark.parametrize(
    ("group", "status"),
    [
        ("never-bedrock-access", 403),
        ("never-vertex-permission", 403),
        ("never-bedrock-validation", 400),
        ("never-vertex-invalid", 400),
        ("never-azure-filter", 400),
    ],
)
def test_a_provider_refusal_never_cools_the_deployment(rig: Rig, group: str, status: int) -> None:
    """The weighted deployment refuses every request: Bedrock 403 `AccessDeniedException`, Vertex 403
    `PERMISSION_DENIED` (answered as 403 since the CI merge ref of 2026-11-17, it was 400), Bedrock 400
    `ValidationException`, Google 400 `INVALID_ARGUMENT`,
    Azure 400 `content_filter`.

    A 403 or 400 is not in `is_cooldown_required`'s list (`health.rs:93`), so all six requests return the
    same error status and all six reach the failing deployment; the sibling receives none. Bedrock's model
    access 403 in particular does not move traffic away, even though every request to it fails.
    """
    assert burst(rig.gateway, group, 6) == (status,) * 6
    assert rig.peer.count(_failing(group)) == 6
    assert rig.peer.count(_healthy(group)) == 0


def test_a_google_502_never_cools_the_deployment_because_it_is_a_connection_error(rig: Rig) -> None:
    """The weighted Gemini deployment answers 502 every time.

    `_map_vertex_exception` raises `APIConnectionError` for a 502, and a connection error never cools a
    deployment (`health.rs:89`), not even by the failure rate rule: seven requests all return HTTP 500 and all
    seven reach the Gemini deployment. A 500 or 503 from the same upstream would have cooled it after five.
    """
    assert burst(rig.gateway, "never-gemini-502", 7) == (500,) * 7
    assert rig.peer.count(_failing("never-gemini-502")) == 7
    assert rig.peer.count(_healthy("never-gemini-502")) == 0


def test_a_lone_bedrock_deployment_is_not_cooled_by_throttling(rig: Rig) -> None:
    """The group has only one Bedrock deployment and it throttles every time.

    A 429 cools only a deployment that has siblings (`health.rs:444`), so three requests all return 429 with
    Bedrock's text and all three reach Bedrock: the caller never sees `No deployments available`.
    """
    responses: Final = tuple(chat(rig.gateway, "lone-bedrock-throttle") for _ in range(3))
    assert tuple(response.status_code for response in responses) == (429, 429, 429)
    assert all(BEDROCK_THROTTLE_TEXT in error_message(response) for response in responses)
    assert rig.peer.count("lone-bedrock-throttle") == 3


def test_a_lone_bedrock_deployment_with_an_invalid_token_is_cooled_and_then_refused_without_a_call(rig: Rig) -> None:
    """The group has only one Bedrock deployment and it answers 403 `UnrecognizedClientException`.

    The mapped 401 is a non-retryable status `is_cooldown_required` allows, so even a lone deployment is
    cooled. The first request returns 401; the second returns HTTP 429 `No deployments available for selected
    model` without reaching Bedrock, which saw one request.
    """
    first: Final = chat(rig.gateway, "lone-bedrock-token")
    second: Final = chat(rig.gateway, "lone-bedrock-token")
    assert first.status_code == 401, first.text
    assert second.status_code == 429, second.text
    assert NO_DEPLOYMENTS in error_message(second)
    assert rig.peer.count("lone-bedrock-token") == 1


def test_a_deployment_cooldown_time_ends_a_bedrock_throttle_cooldown(rig: Rig) -> None:
    """The weighted Bedrock deployment sets `cooldown_time: 3` and throttles once, then answers normally.

    The first request returns 429 and the second 200 from the sibling. Once three seconds have passed the
    Bedrock deployment is picked again: polling until a request is served by it must succeed, and it must
    then have received exactly two requests. A rewrite that used the router's 60 seconds would never serve it.
    """
    label: Final = _failing("now-bedrock-short")
    rig.peer.fixed.pop(label)
    rig.peer.plan(label, bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT))
    assert burst(rig.gateway, "now-bedrock-short", 2) == (429, 200)

    eventually(
        lambda: chat(rig.gateway, "now-bedrock-short"),
        lambda response: response.status_code == 200 and rig.peer.count(label) == 2,
        seconds=20,
    )
    assert rig.peer.count(label) == 2
