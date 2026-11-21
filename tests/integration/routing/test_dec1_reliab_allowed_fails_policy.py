"""`allowed_fails_policy` against the big providers' real error shapes.

`router_settings.allowed_fails_policy` names a failure allowance per exception class, and a deployment may
carry its own policy and `allowed_fails` in `model_info`. The router core
(`litellm/rust/crates/router-core/src/health.rs:388`, `config.rs:154`) picks the allowance by the class litellm's
provider mapping raised, counts failures per deployment in a window of `cooldown_time`, and cools the
deployment once the count exceeds the allowance.

The contracts that matter for the providers: an Anthropic 529 and Google's 503 overload text are
`InternalServerError`, so `InternalServerErrorAllowedFails` governs them, while a Bedrock 500 is a
`ServiceUnavailableError` and reads `ServiceUnavailableErrorAllowedFails`. A router level allowance never
cools a 400 or 403 at all, because `_should_run_cooldown_logic` (`health.rs:362`) consults only the
deployment's own policy before letting a non-cooling status through: Azure's content filter refusal and a
Bedrock `ValidationException` stay in rotation even with `ContentPolicyViolationErrorAllowedFails: 0` and
`BadRequestErrorAllowedFails: 0` on the router, but a deployment level policy with the same field cools them
on the first refusal. A Google 502 (`APIConnectionError`) is never cooled by any policy. With a router policy
that names the error, even a group's only deployment is cooled, which it never is by default.

One owned proxy runs with `num_retries: 0`, `cooldown_time: 60` and the router policy
`RateLimitErrorAllowedFails: 2`, `InternalServerErrorAllowedFails: 0`, `ServiceUnavailableErrorAllowedFails: 3`,
`ContentPolicyViolationErrorAllowedFails: 0`, `BadRequestErrorAllowedFails: 0`. Groups pair a weighted
failing deployment with a healthy OpenAI sibling, so the request count at the failing deployment shows
when its cooldown started.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_reliab_support import (
    AZURE_GENERIC_500_TEXT,
    BEDROCK_ACCESS_TEXT,
    BEDROCK_INTERNAL_TEXT,
    BEDROCK_THROTTLE_TEXT,
    BEDROCK_VALIDATION_TEXT,
    GOOGLE_BAD_GATEWAY_TEXT,
    GOOGLE_OVERLOADED_TEXT,
    GOOGLE_UNAVAILABLE_TEXT,
    NO_DEPLOYMENTS,
    Rig,
    anthropic_deployment,
    anthropic_overloaded,
    azure_content_filter,
    bedrock_deployment,
    bedrock_error,
    burst,
    chat,
    error_message,
    failing_then_healthy,
    google_error,
    openai_error,
    reliab_rig,
)
from pydantic import JsonValue

from tests.integration._support.wire import Reply

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "cooldown_time": 60,
    "allowed_fails_policy": {
        "RateLimitErrorAllowedFails": 2,
        "InternalServerErrorAllowedFails": 0,
        "ServiceUnavailableErrorAllowedFails": 3,
        "ContentPolicyViolationErrorAllowedFails": 0,
        "BadRequestErrorAllowedFails": 0,
    },
}

# group -> (provider, fixed reply of the failing deployment, extra params for it)
FAILING: Final[dict[str, tuple[str, Reply, dict[str, JsonValue]]]] = {
    "afp-anthropic-529": ("anthropic", anthropic_overloaded(), {}),
    "afp-gemini-overloaded": ("gemini", google_error(503, "UNAVAILABLE", GOOGLE_OVERLOADED_TEXT), {}),
    "afp-gemini-unavailable": ("gemini", google_error(503, "UNAVAILABLE", GOOGLE_UNAVAILABLE_TEXT), {}),
    "afp-bedrock-500": ("bedrock", bedrock_error(500, "InternalServerException", BEDROCK_INTERNAL_TEXT), {}),
    "afp-bedrock-throttle": ("bedrock", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT), {}),
    "afp-azure-filter": ("azure", azure_content_filter(), {}),
    "afp-bedrock-validation": ("bedrock", bedrock_error(400, "ValidationException", BEDROCK_VALIDATION_TEXT), {}),
    "afp-azure-500-generic": ("azure", openai_error(500, AZURE_GENERIC_500_TEXT, kind="server_error"), {}),
    "afp-gemini-502": ("gemini", google_error(502, "UNAVAILABLE", GOOGLE_BAD_GATEWAY_TEXT), {}),
    "afp-bedrock-access": ("bedrock", bedrock_error(403, "AccessDeniedException", BEDROCK_ACCESS_TEXT), {}),
    "dep-azure-filter": (
        "azure",
        azure_content_filter(),
        {"model_info": {"allowed_fails_policy": {"ContentPolicyViolationErrorAllowedFails": 0}}},
    ),
    "dep-bedrock-validation": (
        "bedrock",
        bedrock_error(400, "ValidationException", BEDROCK_VALIDATION_TEXT),
        {"model_info": {"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}},
    ),
    "dep-gemini-502": (
        "gemini",
        google_error(502, "UNAVAILABLE", GOOGLE_BAD_GATEWAY_TEXT),
        {
            "model_info": {
                "allowed_fails_policy": {"BadGatewayErrorAllowedFails": 0, "InternalServerErrorAllowedFails": 0}
            }
        },
    ),
    "dep-anthropic-allowed-one": ("anthropic", anthropic_overloaded(), {"model_info": {"allowed_fails": 1}}),
    "dep-bedrock-500-policy": (
        "bedrock",
        bedrock_error(500, "InternalServerException", BEDROCK_INTERNAL_TEXT),
        {"model_info": {"allowed_fails_policy": {"ServiceUnavailableErrorAllowedFails": 0}}},
    ),
}


def _failing(group: str) -> str:
    return f"{group}-bad"


def _healthy(group: str) -> str:
    return f"{group}-ok"


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    entries: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: built once for the config
    for group, (provider, _, extra) in FAILING.items():
        info: JsonValue = extra.get("model_info")
        failing, healthy = failing_then_healthy(provider, group, _failing(group), _healthy(group), base)
        if isinstance(info, dict):
            existing: JsonValue = failing["model_info"]
            assert isinstance(existing, dict)
            failing["model_info"] = {**existing, **info}
        entries.extend((failing, healthy))
    entries.append(anthropic_deployment("lone-anthropic-529", "lone-anthropic-529", base))
    entries.append(bedrock_deployment("lone-bedrock-throttle", "lone-bedrock-throttle", base))
    entries.append(bedrock_deployment("lone-bedrock-access", "lone-bedrock-access", base))
    return entries


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with reliab_rig(tmp_path_factory.mktemp("dec1-reliab-allowed-fails"), _models, ROUTER_SETTINGS) as value:
        for group, (_, reply, _) in FAILING.items():
            value.peer.always(_failing(group), reply)
        value.peer.always("lone-anthropic-529", anthropic_overloaded())
        value.peer.always("lone-bedrock-throttle", bedrock_error(429, "ThrottlingException", BEDROCK_THROTTLE_TEXT))
        value.peer.always("lone-bedrock-access", bedrock_error(403, "AccessDeniedException", BEDROCK_ACCESS_TEXT))
        yield value


def test_internal_server_error_allowed_fails_zero_cools_an_anthropic_529_on_the_first_overload(rig: Rig) -> None:
    """The weighted Anthropic deployment answers 529 `overloaded_error`.

    A 529 is an `InternalServerError`, so `InternalServerErrorAllowedFails: 0` cools it on the first failure:
    statuses (503, 200, 200) and one request at the Anthropic deployment.
    """
    assert burst(rig.gateway, "afp-anthropic-529", 3) == (503, 200, 200)
    assert rig.peer.count(_failing("afp-anthropic-529")) == 1


def test_internal_server_error_allowed_fails_zero_cools_a_google_overload_503_on_the_first_failure(rig: Rig) -> None:
    """The weighted Gemini deployment answers 503 with `The model is overloaded.`.

    The overload text is an `InternalServerError` (status 500), so the internal server allowance of 0
    applies: statuses (500, 200, 200) and one request at Gemini.
    """
    assert burst(rig.gateway, "afp-gemini-overloaded", 3) == (500, 200, 200)
    assert rig.peer.count(_failing("afp-gemini-overloaded")) == 1


def test_a_google_503_with_other_text_reads_the_service_unavailable_allowance(rig: Rig) -> None:
    """The weighted Gemini deployment answers 503 `The service is currently unavailable.`.

    That is a `ServiceUnavailableError`, allowed three failures: requests one to four return 503, request five
    returns 200 from the sibling, and Gemini received exactly four requests.
    """
    assert burst(rig.gateway, "afp-gemini-unavailable", 5) == (503, 503, 503, 503, 200)
    assert rig.peer.count(_failing("afp-gemini-unavailable")) == 4


def test_a_bedrock_500_reads_the_service_unavailable_allowance_not_the_internal_server_one(rig: Rig) -> None:
    """The weighted Bedrock deployment answers 500 `InternalServerException`.

    Bedrock's 500 is a `ServiceUnavailableError`, so it gets three allowed failures, not the internal server
    allowance of 0: statuses (503, 503, 503, 503, 200) and four requests at Bedrock.
    """
    assert burst(rig.gateway, "afp-bedrock-500", 5) == (503, 503, 503, 503, 200)
    assert rig.peer.count(_failing("afp-bedrock-500")) == 4


def test_rate_limit_error_allowed_fails_two_cools_a_bedrock_throttle_on_the_third(rig: Rig) -> None:
    """The weighted Bedrock deployment throttles every request.

    `RateLimitErrorAllowedFails: 2` replaces the default immediate cooldown for a 429: statuses
    (429, 429, 429, 200) and three requests at Bedrock.
    """
    assert burst(rig.gateway, "afp-bedrock-throttle", 4) == (429, 429, 429, 200)
    assert rig.peer.count(_failing("afp-bedrock-throttle")) == 3


def test_a_router_content_policy_allowance_never_cools_an_azure_content_filter_refusal(rig: Rig) -> None:
    """The weighted Azure deployment refuses every request with the 400 `content_filter` body.

    `ContentPolicyViolationErrorAllowedFails: 0` is set on the router, but a 400 never passes
    `is_cooldown_required` and only a deployment's own policy can override that (`health.rs:369`). All five
    requests return 400 and reach Azure; the sibling gets none.
    """
    assert burst(rig.gateway, "afp-azure-filter", 5) == (400,) * 5
    assert rig.peer.count(_failing("afp-azure-filter")) == 5
    assert rig.peer.count(_healthy("afp-azure-filter")) == 0


def test_a_router_bad_request_allowance_never_cools_a_bedrock_validation_exception(rig: Rig) -> None:
    """The weighted Bedrock deployment answers 400 `ValidationException` every time, with
    `BadRequestErrorAllowedFails: 0` on the router.

    For the same reason as the content filter, all five requests return 400 and reach Bedrock.
    """
    assert burst(rig.gateway, "afp-bedrock-validation", 5) == (400,) * 5
    assert rig.peer.count(_failing("afp-bedrock-validation")) == 5


def test_an_azure_500_the_policy_does_not_name_falls_back_to_three_allowed_failures(rig: Rig) -> None:
    """The weighted Azure deployment answers 500 `The service encountered a transient error.`, a bare
    `APIError` no policy field names.

    With a policy set, an unnamed class is allowed `litellm.allowed_fails` (3) failures (`health.rs:458`):
    statuses (500, 500, 500, 500, 200) and four requests at Azure. UNVERIFIED: assumes no
    `allowed_fails` in the proxy's `litellm_settings`; see `Settings::effective_allowed_fails`.
    """
    assert burst(rig.gateway, "afp-azure-500-generic", 5) == (500, 500, 500, 500, 200)
    assert rig.peer.count(_failing("afp-azure-500-generic")) == 4


def test_a_google_502_is_never_cooled_even_with_an_internal_server_allowance_of_zero(rig: Rig) -> None:
    """The weighted Gemini deployment answers 502 every time.

    It is an `APIConnectionError`, which `_should_run_cooldown_logic` excludes before any policy is read: six
    requests return HTTP 500 and all six reach Gemini.
    """
    assert burst(rig.gateway, "afp-gemini-502", 6) == (500,) * 6
    assert rig.peer.count(_failing("afp-gemini-502")) == 6


def test_a_bedrock_access_denied_is_never_cooled_with_a_policy_set(rig: Rig) -> None:
    """The weighted Bedrock deployment answers 403 `AccessDeniedException`.

    A 403 is not a cooling status and `PermissionDeniedError` has no policy field, so five requests return
    403 and all reach Bedrock.
    """
    assert burst(rig.gateway, "afp-bedrock-access", 5) == (403,) * 5
    assert rig.peer.count(_failing("afp-bedrock-access")) == 5


def test_a_deployment_content_policy_allowance_cools_an_azure_content_filter_on_the_first_refusal(rig: Rig) -> None:
    """The weighted Azure deployment carries `model_info.allowed_fails_policy.ContentPolicyViolationErrorAllowedFails: 0`
    and refuses with the content filter 400.

    A deployment's own policy lets a 400 into the cooldown logic (`health.rs:369`): statuses (400, 200, 200)
    and one request at Azure. The same field on the router does nothing.
    """
    assert burst(rig.gateway, "dep-azure-filter", 3) == (400, 200, 200)
    assert rig.peer.count(_failing("dep-azure-filter")) == 1


def test_a_deployment_bad_request_allowance_cools_a_bedrock_validation_exception(rig: Rig) -> None:
    """The weighted Bedrock deployment carries `BadRequestErrorAllowedFails: 0` in its own policy and answers
    400 `ValidationException`.

    Statuses (400, 200, 200) and one request at Bedrock.
    """
    assert burst(rig.gateway, "dep-bedrock-validation", 3) == (400, 200, 200)
    assert rig.peer.count(_failing("dep-bedrock-validation")) == 1


def test_a_deployment_policy_cannot_cool_a_google_502(rig: Rig) -> None:
    """The weighted Gemini deployment carries `BadGatewayErrorAllowedFails: 0` and
    `InternalServerErrorAllowedFails: 0` in its own policy and answers 502.

    The `APIConnectionError` is neither class and connection errors never cool: five requests return 500 and
    all five reach Gemini.
    """
    assert burst(rig.gateway, "dep-gemini-502", 5) == (500,) * 5
    assert rig.peer.count(_failing("dep-gemini-502")) == 5


def test_a_deployment_allowed_fails_wins_over_the_router_policy_for_an_anthropic_529(rig: Rig) -> None:
    """The weighted Anthropic deployment carries `model_info.allowed_fails: 1` and answers 529.

    The deployment's count replaces the router's `InternalServerErrorAllowedFails: 0`
    (`health.rs:418`): statuses (503, 503, 200) and two requests at Anthropic.
    """
    assert burst(rig.gateway, "dep-anthropic-allowed-one", 3) == (503, 503, 200)
    assert rig.peer.count(_failing("dep-anthropic-allowed-one")) == 2


def test_a_deployment_service_unavailable_allowance_wins_over_the_router_one_for_a_bedrock_500(rig: Rig) -> None:
    """The weighted Bedrock deployment carries `ServiceUnavailableErrorAllowedFails: 0` while the router
    allows 3, and answers 500 `InternalServerException`.

    The deployment policy is read first: statuses (503, 200, 200) and one request at Bedrock.
    """
    assert burst(rig.gateway, "dep-bedrock-500-policy", 3) == (503, 200, 200)
    assert rig.peer.count(_failing("dep-bedrock-500-policy")) == 1


def test_a_router_policy_naming_the_error_cools_a_lone_anthropic_deployment(rig: Rig) -> None:
    """The group has only one Anthropic deployment and it answers 529.

    By default a lone deployment is never cooled for a 5xx, but a router policy entry that names the class
    does cool it (`health.rs:454`): the first request returns 503, the second HTTP 429 `No deployments
    available for selected model` without reaching Anthropic, which saw one request.
    """
    first: Final = chat(rig.gateway, "lone-anthropic-529")
    second: Final = chat(rig.gateway, "lone-anthropic-529")
    assert first.status_code == 503, first.text
    assert second.status_code == 429, second.text
    assert NO_DEPLOYMENTS in error_message(second)
    assert rig.peer.count("lone-anthropic-529") == 1


def test_a_router_rate_limit_allowance_cools_a_lone_bedrock_deployment_after_three_throttles(rig: Rig) -> None:
    """The group has only one Bedrock deployment and it throttles every time.

    With `RateLimitErrorAllowedFails: 2` the lone deployment is cooled on its third throttle: statuses
    (429, 429, 429, 429), the last one being `No deployments available` with no upstream call, and Bedrock
    saw exactly three requests.
    """
    responses: Final = tuple(chat(rig.gateway, "lone-bedrock-throttle") for _ in range(4))
    assert tuple(response.status_code for response in responses) == (429, 429, 429, 429)
    assert BEDROCK_THROTTLE_TEXT in error_message(responses[2])
    assert NO_DEPLOYMENTS in error_message(responses[3])
    assert rig.peer.count("lone-bedrock-throttle") == 3


def test_a_lone_bedrock_deployment_answering_access_denied_is_never_cooled_with_a_policy_set(rig: Rig) -> None:
    """The group has only one Bedrock deployment and it answers 403 `AccessDeniedException`.

    No policy field names a `PermissionDeniedError`, so a lone deployment stays warm: four requests return
    403 and all four reach Bedrock.
    """
    assert burst(rig.gateway, "lone-bedrock-access", 4) == (403,) * 4
    assert rig.peer.count("lone-bedrock-access") == 4
