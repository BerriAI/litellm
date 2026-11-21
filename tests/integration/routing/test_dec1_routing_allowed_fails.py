"""`router_settings.allowed_fails`: how many failures a deployment may have before it is cooled down.

One owned proxy runs with `num_retries: 0`, `cooldown_time: 60` and `allowed_fails: 1`. Setting
`allowed_fails` explicitly replaces the default cooldown rules with a plain counter per deployment: the
deployment is cooled down when its failure count inside one `cooldown_time` window is greater than
`allowed_fails`. That changes two things callers can see. A 429 no longer cools a deployment down on the
first failure, and a 5xx no longer needs five requests. It also means a group with a single deployment and no
fallback is never cooled down, not even for a 401.

Each group pairs a failing deployment with `weight: 1` and a healthy sibling without a weight, so the failing
deployment is always picked until it is cooled down.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import Rig, chat, deployment, routing_rig, statuses
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {"num_retries": 0, "cooldown_time": 60, "allowed_fails": 1}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("fails-500", "status-500-fails", api_base, weight=1),
        deployment("fails-500", "ok-fails-500", api_base),
        deployment("fails-429", "status-429-fails", api_base, weight=1),
        deployment("fails-429", "ok-fails-429", api_base),
        deployment("fails-400", "status-400-fails", api_base, weight=1),
        deployment("fails-400", "ok-fails-400", api_base),
        deployment("fails-lone-401", "status-401-fails-lone", api_base),
        deployment("fails-lone-500", "status-500-fails-lone", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-allowed-fails"), _models, ROUTER_SETTINGS) as value:
        yield value


def _burst(rig: Rig, group: str, requests: int) -> tuple[int, ...]:
    return statuses(tuple(chat(rig.gateway, group) for _ in range(requests)))


def test_allowed_fails_one_cools_a_deployment_on_its_second_500(rig: Rig) -> None:
    """`allowed_fails: 1`; a weighted deployment answers 500 and its sibling is healthy.

    Requests one and two must return 500 and requests three to five must return 200 from the sibling. The
    failing upstream receives exactly two requests: the second failure is the first one above the allowance.
    Without the setting the cooldown would only start at the fifth failure.
    """
    assert _burst(rig, "fails-500", 5) == (500, 500, 200, 200, 200)
    assert rig.upstream.count("status-500-fails") == 2
    assert rig.upstream.count("ok-fails-500") == 3


def test_allowed_fails_one_lets_a_first_429_through_without_a_cooldown(rig: Rig) -> None:
    """`allowed_fails: 1`; a weighted deployment answers 429 and its sibling is healthy.

    Without the setting a single 429 cools the deployment down. With it, requests one and two must both
    return 429 from the failing upstream, and only request three returns 200 from the sibling. The failing
    upstream receives exactly two requests.
    """
    assert _burst(rig, "fails-429", 4) == (429, 429, 200, 200)
    assert rig.upstream.count("status-429-fails") == 2


def test_a_400_does_not_count_towards_allowed_fails(rig: Rig) -> None:
    """`allowed_fails: 1`; a weighted deployment answers 400 and its sibling is healthy.

    Five requests must all return 400 from the failing upstream and the sibling must receive nothing: a bad
    request is not a failure the counter sees.
    """
    assert _burst(rig, "fails-400", 5) == (400,) * 5
    assert rig.upstream.count("status-400-fails") == 5
    assert rig.upstream.count("ok-fails-400") == 0


def test_with_allowed_fails_set_a_group_with_one_deployment_is_not_cooled_down_by_401s(rig: Rig) -> None:
    """`allowed_fails: 1`; the group has a single deployment that answers 401 and no fallback.

    Four requests must all return 401 from the upstream and all four must reach it. Under the default rules
    the first 401 would cool this deployment down and the second request would get a 429. With `allowed_fails`
    set, a lone deployment without a fallback is never cooled down.
    """
    assert _burst(rig, "fails-lone-401", 4) == (401,) * 4
    assert rig.upstream.count("status-401-fails-lone") == 4


def test_with_allowed_fails_set_a_group_with_one_deployment_is_not_cooled_down_by_500s(rig: Rig) -> None:
    """`allowed_fails: 1`; the group has a single deployment that answers 500 and no fallback.

    Four requests, well past the allowance, must all return 500 from the upstream and all four must reach it.
    """
    assert _burst(rig, "fails-lone-500", 4) == (500,) * 4
    assert rig.upstream.count("status-500-fails-lone") == 4
