"""`router_settings.retry_policy`, `model_group_retry_policy` and `retry_after` as seen through the proxy.

One owned proxy runs with `num_retries: 1`, cooldowns disabled, `retry_after: 3`, and this policy:
`RateLimitErrorRetries: 3`, `BadRequestErrorRetries: 1`, `ContentPolicyViolationErrorRetries: 2`,
`AuthenticationErrorRetries: 0`, `InternalServerErrorRetries: 2`, `NotFoundErrorRetries: 1`,
`TimeoutErrorRetries: 0`. A policy entry replaces both the retry count and the "is this error retryable"
decision for its error class, so a 400 that is normally never retried is retried once here. An error class
the policy does not name keeps `num_retries`. A `model_group_retry_policy` entry replaces the whole global
policy for its group, not just the fields it sets.

Each group has two deployments that call one scripted upstream model, so the number of requests that model
received is the number of attempts. Expected attempts are always the policy count plus one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import Rig, chat, deployment, pair, routing_rig
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

RETRY_AFTER_SECONDS: Final = 3
ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 1,
    "disable_cooldowns": True,
    "retry_after": RETRY_AFTER_SECONDS,
    "retry_policy": {
        "RateLimitErrorRetries": 3,
        "BadRequestErrorRetries": 1,
        "ContentPolicyViolationErrorRetries": 2,
        "AuthenticationErrorRetries": 0,
        "InternalServerErrorRetries": 2,
        "NotFoundErrorRetries": 1,
        "TimeoutErrorRetries": 0,
    },
    "model_group_retry_policy": {
        "policy-group-500": {"InternalServerErrorRetries": 0},
        "policy-group-429": {"InternalServerErrorRetries": 0},
    },
}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        *pair("policy-429", "status-429-policy", api_base),
        *pair("policy-400", "status-400-policy", api_base),
        *pair("policy-content", "policy-retried", api_base),
        *pair("policy-401", "status-401-policy", api_base),
        *pair("policy-500", "status-500-policy", api_base),
        *pair("policy-404", "status-404-policy", api_base),
        *pair("policy-timeout", "slow-policy", api_base, timeout=1),
        *pair("policy-503", "status-503-policy", api_base),
        *pair("policy-group-500", "status-500-grouppolicy", api_base),
        *pair("policy-group-429", "status-429-grouppolicy", api_base),
        *pair("policy-header-zero", "status-429-headerzero", api_base),
        deployment("policy-retry-after", "script-retryafter-setting", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-retry-policy"), _models, ROUTER_SETTINGS) as value:
        yield value


def test_rate_limit_error_retries_three_gives_a_429_four_attempts(rig: Rig) -> None:
    """The policy sets `RateLimitErrorRetries: 3` while `num_retries` is 1, and the upstream always answers 429.

    The upstream must receive exactly four requests and the caller gets HTTP 429. The policy count wins over
    `num_retries` for its error class.
    """
    response: Final = chat(rig.gateway, "policy-429")
    assert response.status_code == 429, response.text
    assert rig.upstream.count("status-429-policy") == 4


def test_bad_request_error_retries_one_retries_a_400_that_is_otherwise_never_retried(rig: Rig) -> None:
    """The policy sets `BadRequestErrorRetries: 1` and the upstream always answers 400.

    Without a policy a 400 is attempted once. With it the upstream must receive exactly two requests, and the
    caller still gets HTTP 400.
    """
    response: Final = chat(rig.gateway, "policy-400")
    assert response.status_code == 400, response.text
    assert rig.upstream.count("status-400-policy") == 2


def test_content_policy_violation_retries_wins_over_bad_request_retries_for_a_content_refusal(rig: Rig) -> None:
    """The upstream answers the 400 `content_policy_violation` body; the policy has
    `ContentPolicyViolationErrorRetries: 2` and `BadRequestErrorRetries: 1`.

    A content policy error is also a bad request, and the more specific class must decide: the upstream
    receives exactly three requests, and the caller gets HTTP 400.
    """
    response: Final = chat(rig.gateway, "policy-content")
    assert response.status_code == 400, response.text
    assert rig.upstream.count("policy-retried") == 3


def test_authentication_error_retries_zero_stops_a_401_from_being_retried_on_the_sibling(rig: Rig) -> None:
    """The policy sets `AuthenticationErrorRetries: 0` and the group has two deployments that answer 401.

    Without a policy the sibling would be tried. With a count of zero the upstream must receive exactly one
    request, and the caller gets HTTP 401.
    """
    response: Final = chat(rig.gateway, "policy-401")
    assert response.status_code == 401, response.text
    assert rig.upstream.count("status-401-policy") == 1


def test_internal_server_error_retries_two_gives_a_500_three_attempts(rig: Rig) -> None:
    """The policy sets `InternalServerErrorRetries: 2` while `num_retries` is 1, and the upstream answers 500.

    The upstream must receive exactly three requests and the caller gets HTTP 500.
    """
    response: Final = chat(rig.gateway, "policy-500")
    assert response.status_code == 500, response.text
    assert rig.upstream.count("status-500-policy") == 3


def test_not_found_error_retries_one_retries_a_404(rig: Rig) -> None:
    """The policy sets `NotFoundErrorRetries: 1` and the upstream always answers 404.

    Without a policy a 404 is attempted once. With it the upstream must receive exactly two requests, and the
    caller gets HTTP 404.
    """
    response: Final = chat(rig.gateway, "policy-404")
    assert response.status_code == 404, response.text
    assert rig.upstream.count("status-404-policy") == 2


def test_timeout_error_retries_zero_stops_a_timeout_from_being_retried(rig: Rig) -> None:
    """The policy sets `TimeoutErrorRetries: 0`; both deployments time out after 1 second against a 4 second
    upstream.

    `num_retries: 1` alone would give two attempts. The upstream must receive exactly one request, and the
    caller gets HTTP 408.
    """
    response: Final = chat(rig.gateway, "policy-timeout")
    assert response.status_code == 408, response.text
    assert rig.upstream.count("slow-policy") == 1


def test_an_error_class_the_policy_does_not_name_keeps_num_retries(rig: Rig) -> None:
    """The policy has no `ServiceUnavailableErrorRetries` and no `DefaultRetries`; the upstream answers 503.

    The router falls back to `num_retries: 1`, so the upstream must receive exactly two requests, and the
    caller gets HTTP 503.
    """
    response: Final = chat(rig.gateway, "policy-503")
    assert response.status_code == 503, response.text
    assert rig.upstream.count("status-503-policy") == 2


def test_model_group_retry_policy_overrides_the_global_policy_for_its_group(rig: Rig) -> None:
    """`model_group_retry_policy` gives group `policy-group-500` `InternalServerErrorRetries: 0`; the global
    policy says 2.

    The upstream, which answers 500, must receive exactly one request, and the caller gets HTTP 500.
    """
    response: Final = chat(rig.gateway, "policy-group-500")
    assert response.status_code == 500, response.text
    assert rig.upstream.count("status-500-grouppolicy") == 1


def test_a_group_retry_policy_replaces_the_whole_global_policy_not_just_its_own_fields(rig: Rig) -> None:
    """Group `policy-group-429` has a group policy that only sets `InternalServerErrorRetries`; the global
    policy says `RateLimitErrorRetries: 3`.

    The group's upstream answers 429. The group policy has no rate limit field, so the count comes from
    `num_retries: 1` and not from the global policy: the upstream must receive exactly two requests, not four.
    The caller gets HTTP 429.
    """
    response: Final = chat(rig.gateway, "policy-group-429")
    assert response.status_code == 429, response.text
    assert rig.upstream.count("status-429-grouppolicy") == 2


def test_a_request_with_zero_retries_ignores_the_retry_policy(rig: Rig) -> None:
    """The policy sets `RateLimitErrorRetries: 3` and the caller sends `x-litellm-num-retries: 0`.

    A request that asks for zero retries is not handed to the policy at all: the upstream, which answers 429,
    must receive exactly one request, and the caller gets HTTP 429.
    """
    response: Final = chat(rig.gateway, "policy-header-zero", headers={"x-litellm-num-retries": "0"})
    assert response.status_code == 429, response.text
    assert rig.upstream.count("status-429-headerzero") == 1


def test_retry_after_is_the_minimum_wait_before_a_retry_on_a_group_with_one_deployment(rig: Rig) -> None:
    """`router_settings.retry_after: 3`; the group has one deployment whose upstream answers 500 once, then 200.

    The caller gets 200 after two upstream requests (the policy allows two retries for a 500). The second
    request must arrive at least 3 seconds after the first: the 0.5s backoff is raised to `retry_after`. At
    most 0.75s of jitter is added, with 3s of slack for a slow runner.
    """
    rig.upstream.script("script-retryafter-setting", 500)
    response: Final = chat(rig.gateway, "policy-retry-after")
    assert response.status_code == 200, response.text
    first, second = (seen.at for seen in rig.upstream.calls("script-retryafter-setting"))
    assert RETRY_AFTER_SECONDS <= second - first < RETRY_AFTER_SECONDS + 0.75 + 3
