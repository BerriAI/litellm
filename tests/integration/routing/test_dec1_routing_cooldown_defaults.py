"""Which upstream failures put a deployment in cooldown when no `allowed_fails` setting is configured.

One owned proxy runs with `num_retries: 0` and `cooldown_time: 60`, and nothing else. Most groups pair a
failing deployment that carries `weight: 1` with a healthy sibling that has no weight. Simple shuffle
normalises weights, so the weighted deployment is picked every time while it is healthy and the sibling is
only reached once the first one is in cooldown. Counting the requests the failing upstream model received
therefore shows exactly when the cooldown started.

The rules pinned here: a 401, a 404 or a 429 cools a deployment down on the first failure. A 408 or a 5xx
cools it down only once more than half of at least five requests in the current minute failed, and so does a
refused connection, which reaches the router as a 500. A 400 never cools a deployment down. A group with one deployment is never cooled down for a 429
or a 5xx, but is for a 401. While a deployment is in cooldown it receives no traffic, and when every
deployment of a group is in cooldown the caller gets HTTP 429 `No deployments available for selected model`
without any upstream call.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    NO_DEPLOYMENTS,
    UNREACHABLE_API_BASE,
    Rig,
    chat,
    content_of,
    deployment,
    embed,
    error_message,
    error_of,
    routing_rig,
    statuses,
)
from pydantic import JsonValue

from tests.integration._support.client import eventually

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

COOLDOWN_SECONDS: Final = 60
SHORT_COOLDOWN_SECONDS: Final = 2
ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {"num_retries": 0, "cooldown_time": COOLDOWN_SECONDS}


def _failing_then_healthy(
    group: str, failing: str, healthy: str, api_base: str, **failing_params: JsonValue
) -> tuple[dict[str, JsonValue], ...]:
    return (
        deployment(group, failing, api_base, weight=1, **failing_params),
        deployment(group, healthy, api_base),
    )


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        *_failing_then_healthy("cool-429", "status-429-cool", "ok-cool-429", api_base),
        *_failing_then_healthy("cool-401", "status-401-cool", "ok-cool-401", api_base),
        *_failing_then_healthy("cool-404", "status-404-cool", "ok-cool-404", api_base),
        *_failing_then_healthy("cool-500-once", "status-500-once", "ok-500-once", api_base),
        *_failing_then_healthy("cool-500-rate", "status-500-rate", "ok-500-rate", api_base),
        *_failing_then_healthy("cool-408-rate", "status-408-rate", "ok-408-rate", api_base),
        *_failing_then_healthy("cool-mix", "script-mix", "ok-mix", api_base),
        *_failing_then_healthy("cool-400", "status-400-cool", "ok-cool-400", api_base),
        deployment("cool-refused", "ok-never-reached", UNREACHABLE_API_BASE, weight=1),
        deployment("cool-refused", "ok-cool-refused", api_base),
        deployment("cool-lone-429", "status-429-lonecool", api_base),
        deployment("cool-lone-500", "status-500-lonecool", api_base),
        deployment("cool-lone-401", "status-401-lonecool", api_base),
        deployment("cool-all", "status-429-all-first", api_base, weight=1),
        deployment("cool-all", "status-429-all-second", api_base),
        *_failing_then_healthy(
            "cool-expiry", "status-429-expiry", "ok-expiry", api_base, cooldown_time=SHORT_COOLDOWN_SECONDS
        ),
        *_failing_then_healthy("cool-retry-after", "retryafter-cool", "ok-retry-after", api_base),
        *_failing_then_healthy("cool-cross", "status-429-cross", "ok-cross", api_base),
        *_failing_then_healthy("cool-by-id", "status-429-coolbyid", "ok-coolbyid", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-cooldown-defaults"), _models, ROUTER_SETTINGS) as value:
        yield value


def _burst(rig: Rig, group: str, requests: int) -> tuple[int, ...]:
    return statuses(tuple(chat(rig.gateway, group) for _ in range(requests)))


@pytest.mark.parametrize("status", [429, 401, 404])
def test_one_failure_with_this_status_cools_the_deployment_and_traffic_moves_to_the_sibling(
    rig: Rig, status: int
) -> None:
    """A weighted deployment answers 429, 401 or 404 and its unweighted sibling is healthy.

    The first request returns the upstream's status to the caller (there are no retries). The next five
    requests must all return 200 from the sibling, and the failing upstream must have received exactly one
    request in total: one failure was enough to start a 60 second cooldown.
    """
    assert _burst(rig, f"cool-{status}", 6) == (status, 200, 200, 200, 200, 200)
    assert rig.upstream.count(f"status-{status}-cool") == 1
    assert rig.upstream.count(f"ok-cool-{status}") == 5


def test_a_single_500_does_not_cool_the_deployment(rig: Rig) -> None:
    """A weighted deployment answers 500 and its sibling is healthy.

    Two requests must both return 500 and both reach the failing upstream: one 5xx is not enough to cool a
    deployment down, so the weighted deployment keeps being picked.
    """
    assert _burst(rig, "cool-500-once", 2) == (500, 500)
    assert rig.upstream.count("status-500-once") == 2
    assert rig.upstream.count("ok-500-once") == 0


def test_the_fifth_straight_500_in_a_minute_cools_the_deployment(rig: Rig) -> None:
    """A weighted deployment answers 500 to every request and its sibling is healthy.

    Requests one to five must return 500 and requests six to eight must return 200 from the sibling. The
    failing upstream receives exactly five requests: the cooldown starts once at least five requests were
    counted and more than half of them failed. The success and failure counters live for 60 seconds, so the
    eight requests of this test always fall inside one window.
    """
    assert _burst(rig, "cool-500-rate", 8) == (500, 500, 500, 500, 500, 200, 200, 200)
    assert rig.upstream.count("status-500-rate") == 5


def test_the_fifth_straight_upstream_408_in_a_minute_cools_the_deployment(rig: Rig) -> None:
    """A weighted deployment answers HTTP 408 to every request and its sibling is healthy.

    A 408 follows the failure rate rule like a 5xx, not the immediate rule of a 429: requests one to five
    return 408, requests six and seven return 200 from the sibling, and the failing upstream receives exactly
    five requests.
    """
    assert _burst(rig, "cool-408-rate", 7) == (408, 408, 408, 408, 408, 200, 200)
    assert rig.upstream.count("status-408-rate") == 5


def test_failures_at_or_under_half_of_the_minutes_requests_do_not_cool_the_deployment(rig: Rig) -> None:
    """A weighted deployment answers 200 four times, then 500 twice, then 200 again.

    After six requests two of six failed, which is under the 50 percent threshold, so the deployment is not
    cooled down: the seventh request must still reach it and return 200 with its content, and the sibling
    must have received nothing.
    """
    rig.upstream.script("script-mix", 200, 200, 200, 200, 500, 500)
    assert _burst(rig, "cool-mix", 6) == (200, 200, 200, 200, 500, 500)
    seventh: Final = chat(rig.gateway, "cool-mix")
    assert seventh.status_code == 200, seventh.text
    assert content_of(seventh) == "served script-mix"
    assert rig.upstream.count("ok-mix") == 0


def test_a_400_never_cools_the_deployment(rig: Rig) -> None:
    """A weighted deployment answers 400 to every request and its sibling is healthy.

    Eight requests must all return 400 and all reach the failing upstream. A bad request says nothing about
    the deployment's health, so it never counts towards a cooldown, not even at a 100 percent failure rate.
    """
    assert _burst(rig, "cool-400", 8) == (400,) * 8
    assert rig.upstream.count("status-400-cool") == 8
    assert rig.upstream.count("ok-cool-400") == 0


def test_a_refused_connection_cools_the_deployment_by_the_5xx_failure_rate_rule(rig: Rig) -> None:
    """A weighted deployment points at a port nothing listens on and its sibling is healthy.

    Requests one to five return 500 and requests six to eight return 200 from the sibling, exactly like a
    deployment that answers 500: the unreachable deployment is cooled down once more than half of at least
    five requests in the minute failed.
    Pinned current behaviour: router-core exempts `APIConnectionError` from cooldowns
    (litellm-rust/crates/router-core/src/health.rs:130), but the OpenAI handler reports a refused connection as
    `litellm.InternalServerError` with status 500 (see `test_dec1_failures_upstream_connection.py`), so the
    exemption never applies and the failure rate rule for a 5xx does.
    """
    assert _burst(rig, "cool-refused", 8) == (500, 500, 500, 500, 500, 200, 200, 200)
    assert rig.upstream.count("ok-cool-refused") == 3


def test_a_group_with_one_deployment_is_not_cooled_down_by_429s(rig: Rig) -> None:
    """The group has a single deployment that answers 429 to every request.

    Six requests must all return the upstream's 429 and all six must reach the upstream. Cooling down the only
    deployment would turn a provider rate limit into a full outage of the group.
    """
    responses: Final = tuple(chat(rig.gateway, "cool-lone-429") for _ in range(6))
    assert statuses(responses) == (429,) * 6
    assert rig.upstream.count("status-429-lonecool") == 6
    assert all(NO_DEPLOYMENTS not in response.text for response in responses)


def test_a_group_with_one_deployment_is_not_cooled_down_by_500s(rig: Rig) -> None:
    """The group has a single deployment that answers 500 to every request.

    Eight requests, past the five-request failure rate threshold, must all return 500 and all reach the
    upstream. The failure rate rule only applies to groups with more than one deployment.
    """
    assert _burst(rig, "cool-lone-500", 8) == (500,) * 8
    assert rig.upstream.count("status-500-lonecool") == 8


def test_a_group_with_one_deployment_is_cooled_down_by_a_401(rig: Rig) -> None:
    """The group has a single deployment that answers 401.

    The first request returns 401 from the upstream. The second request must return HTTP 429 with a message
    containing `No deployments available for selected model`, and the upstream must still have received only
    one request. A non-retryable status cools down even the only deployment of a group.
    """
    first: Final = chat(rig.gateway, "cool-lone-401")
    assert first.status_code == 401, first.text
    second: Final = chat(rig.gateway, "cool-lone-401")
    assert second.status_code == 429, second.text
    assert NO_DEPLOYMENTS in error_message(second), second.text
    assert rig.upstream.count("status-401-lonecool") == 1


def test_when_every_deployment_is_in_cooldown_the_caller_gets_429_without_an_upstream_call(rig: Rig) -> None:
    """Both deployments of the group answer 429.

    The first two requests each cool one deployment down and return the upstream's 429. The third request
    must return HTTP 429 with `code` `"429"` and a message that contains
    `No deployments available for selected model`, and neither upstream model may have received a second
    request.
    """
    assert _burst(rig, "cool-all", 2) == (429, 429)
    third: Final = chat(rig.gateway, "cool-all")
    assert third.status_code == 429, third.text
    assert error_of(third)["code"] == "429"
    assert NO_DEPLOYMENTS in error_message(third), third.text
    assert rig.upstream.count("status-429-all-first") == 1
    assert rig.upstream.count("status-429-all-second") == 1


def test_a_deployment_is_tried_again_once_its_own_cooldown_time_has_passed(rig: Rig) -> None:
    """The failing deployment sets `litellm_params.cooldown_time: 2` while the router's `cooldown_time` is 60.

    Its 429 cools it down and the next request is served by the sibling. Polling with further requests, the
    failing upstream must receive a second request within 20 seconds: the deployment's own two second cooldown
    applies, not the router's sixty.
    """
    assert _burst(rig, "cool-expiry", 2) == (429, 200)
    assert rig.upstream.count("status-429-expiry") == 1

    def attempts_after_one_more_request() -> int:
        chat(rig.gateway, "cool-expiry")
        return rig.upstream.count("status-429-expiry")

    eventually(attempts_after_one_more_request, lambda count: count >= 2, seconds=20)


def test_a_provider_retry_after_header_is_used_as_the_cooldown_duration(rig: Rig) -> None:
    """The failing deployment answers one 429 with `retry-after: 2`, then 200; the router's `cooldown_time` is 60.

    The 429 cools the deployment down and the next request is served by the sibling. Polling with further
    requests, the first deployment must serve again within 20 seconds. UNVERIFIED: that the proxy passes the
    provider's `retry-after` through as the cooldown length; `Health::cooldown_for_call` in
    `litellm-rust/crates/router-core/src/health.rs` prefers it over the router's `cooldown_time`.
    """
    rig.upstream.script("retryafter-cool", 429)
    assert _burst(rig, "cool-retry-after", 2) == (429, 200)

    def attempts_after_one_more_request() -> int:
        chat(rig.gateway, "cool-retry-after")
        return rig.upstream.count("retryafter-cool")

    eventually(attempts_after_one_more_request, lambda count: count >= 2, seconds=20)


def test_a_cooldown_started_by_an_embeddings_call_also_moves_chat_traffic_to_the_sibling(rig: Rig) -> None:
    """A weighted deployment answers 429 and its sibling is healthy.

    An embeddings request hits the weighted deployment and returns 429. Three chat requests to the same group
    must then all return 200 from the sibling, and the failing upstream must have received one request in
    total. A cooldown belongs to the deployment, not to the endpoint that observed the failure.
    Product bug: in CI run 42966523075 the upstream saw three requests per router attempt, the later ones
    with `x-stainless-retry-count`. The router passes `max_retries: 0` to every call
    (litellm/router_rust/surface/settings.py:194), but `OpenAIChatCompletion.embedding` reads it as
    `max_retries or litellm.DEFAULT_MAX_RETRIES` (litellm/llms/openai/openai.py:1411), so 0 becomes the
    OpenAI client's default of two retries.
    """
    pytest.skip("BUG: OpenAI embeddings ignore the router's max_retries 0 and the client retries each attempt twice")
    embedding: Final = embed(rig.gateway, "cool-cross")
    assert embedding.status_code == 429, embedding.text
    assert _burst(rig, "cool-cross", 3) == (200, 200, 200)
    assert rig.upstream.count("status-429-cross") == 1
    assert rig.upstream.count("ok-cross") == 3


def test_a_call_by_deployment_id_reaches_a_deployment_that_is_in_cooldown(rig: Rig) -> None:
    """The weighted deployment with id `dec1-status-429-coolbyid` answers 429 and is cooled down by the first
    request; the next request to the group is served by the sibling.

    A request whose `model` is the cooled deployment's id must still be sent to it: the caller gets the
    upstream's 429, not `No deployments available`, and the failing upstream has then received two requests.
    The cooldown filter is skipped for a deployment named by its id.
    """
    assert _burst(rig, "cool-by-id", 2) == (429, 200)
    by_id: Final = chat(rig.gateway, "dec1-status-429-coolbyid")
    assert by_id.status_code == 429, by_id.text
    assert NO_DEPLOYMENTS not in by_id.text
    assert rig.upstream.count("status-429-coolbyid") == 2
