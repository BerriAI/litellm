"""`router_settings.allowed_fails_policy`: a failure allowance per error class.

One owned proxy runs with `num_retries: 0`, `cooldown_time: 60` and the policy
`AuthenticationErrorAllowedFails: 2`, `RateLimitErrorAllowedFails: 0`. A deployment is cooled down when its
failure count inside one `cooldown_time` window is greater than the allowance of the failing error class. An
error class the policy does not name uses the process default allowance of 3, so it takes four failures. A
policy entry also lifts the "never cool down a lone deployment" protection for its error class.

Each two-deployment group pairs a failing deployment with `weight: 1` and a healthy sibling without a weight,
so the failing deployment is always picked until it is cooled down.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    NO_DEPLOYMENTS,
    Rig,
    chat,
    deployment,
    error_message,
    routing_rig,
    statuses,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "cooldown_time": 60,
    "allowed_fails_policy": {"AuthenticationErrorAllowedFails": 2, "RateLimitErrorAllowedFails": 0},
}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("afp-401", "status-401-afp", api_base, weight=1),
        deployment("afp-401", "ok-afp-401", api_base),
        deployment("afp-429", "status-429-afp", api_base, weight=1),
        deployment("afp-429", "ok-afp-429", api_base),
        deployment("afp-503", "status-503-afp", api_base, weight=1),
        deployment("afp-503", "ok-afp-503", api_base),
        deployment("afp-lone-429", "status-429-afp-lone", api_base),
        deployment("afp-lone-503", "status-503-afp-lone", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-allowed-fails-policy"), _models, ROUTER_SETTINGS) as value:
        yield value


def _burst(rig: Rig, group: str, requests: int) -> tuple[int, ...]:
    return statuses(tuple(chat(rig.gateway, group) for _ in range(requests)))


def test_authentication_error_allowed_fails_two_cools_a_deployment_on_its_third_401(rig: Rig) -> None:
    """The policy allows two authentication failures; a weighted deployment answers 401, its sibling is healthy.

    Requests one to three must return 401 and requests four and five must return 200 from the sibling. The
    failing upstream receives exactly three requests. Without the policy the first 401 would start the
    cooldown.
    """
    assert _burst(rig, "afp-401", 5) == (401, 401, 401, 200, 200)
    assert rig.upstream.count("status-401-afp") == 3


def test_rate_limit_error_allowed_fails_zero_cools_a_deployment_on_its_first_429(rig: Rig) -> None:
    """The policy allows zero rate limit failures; a weighted deployment answers 429, its sibling is healthy.

    Request one must return 429 and requests two to four must return 200 from the sibling. The failing
    upstream receives exactly one request.
    """
    assert _burst(rig, "afp-429", 4) == (429, 200, 200, 200)
    assert rig.upstream.count("status-429-afp") == 1


def test_an_error_class_the_policy_does_not_name_is_allowed_three_failures(rig: Rig) -> None:
    """The policy has no entry for a 503; a weighted deployment answers 503 and its sibling is healthy.

    With a policy configured, an unnamed error class uses the default allowance of 3: requests one to four
    must return 503 and requests five and six must return 200 from the sibling. The failing upstream receives
    exactly four requests, not the five the default failure rate rule would need.
    """
    assert _burst(rig, "afp-503", 6) == (503, 503, 503, 503, 200, 200)
    assert rig.upstream.count("status-503-afp") == 4


def test_a_policy_entry_cools_down_even_a_group_with_one_deployment(rig: Rig) -> None:
    """The policy allows zero rate limit failures; the group has a single deployment that answers 429.

    The first request returns the upstream's 429. The second must return HTTP 429 with a message containing
    `No deployments available for selected model`, and the upstream must have received exactly one request.
    Without a policy entry a lone deployment is never cooled down for a 429.
    """
    first: Final = chat(rig.gateway, "afp-lone-429")
    assert first.status_code == 429, first.text
    assert NO_DEPLOYMENTS not in first.text
    second: Final = chat(rig.gateway, "afp-lone-429")
    assert second.status_code == 429, second.text
    assert NO_DEPLOYMENTS in error_message(second), second.text
    assert rig.upstream.count("status-429-afp-lone") == 1


def test_a_lone_deployment_is_not_cooled_down_for_an_error_class_the_policy_does_not_name(rig: Rig) -> None:
    """The policy has no entry for a 503; the group has a single deployment that answers 503 and no fallback.

    Six requests, past the default allowance of three, must all return 503 from the upstream and all six must
    reach it.
    """
    assert _burst(rig, "afp-lone-503", 6) == (503,) * 6
    assert rig.upstream.count("status-503-afp-lone") == 6
