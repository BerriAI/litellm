"""Which upstream failures the router retries inside one model group, and how many times.

One owned proxy runs with `router_settings.num_retries: 2` and cooldowns disabled, so the only thing that
changes between tests is the upstream's answer. Every group is made of deployments that all call one scripted
upstream model, and the tests count how many requests that upstream model received. A retry budget of 2 means
at most three upstream requests. Statuses 408, 409, 429 and every 5xx are retryable. A 400 and a 404 are
never retried. A 401 is retried only when the group has another deployment to try.

Groups of two or more deployments retry immediately. A group with a single deployment waits between attempts
(0.5s doubling up to 8s, plus up to 0.75s of jitter), and a provider `retry-after` header of up to 60 seconds
replaces that wait. A rewrite that changes any of these counts changes provider bills and latency for every
caller, so each count is pinned as a literal.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    MESSAGE_PREFIX,
    Rig,
    chat,
    content_of,
    deployment,
    error_message,
    error_of,
    pair,
    routing_rig,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

NUM_RETRIES: Final = 2
ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {"num_retries": NUM_RETRIES, "disable_cooldowns": True}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        *pair("retry-500", "status-500-retry", api_base),
        *pair("retry-recover", "script-recover", api_base),
        *pair("retry-429", "status-429-retry", api_base),
        *pair("retry-503", "status-503-retry", api_base),
        *pair("retry-408", "status-408-retry", api_base),
        *pair("retry-400", "status-400-retry", api_base),
        *pair("retry-404", "status-404-retry", api_base),
        deployment("retry-401-lone", "status-401-lone", api_base),
        deployment("retry-401-siblings", "status-401-first", api_base),
        deployment("retry-401-siblings", "status-401-second", api_base),
        deployment("retry-401-siblings", "status-401-third", api_base),
        *pair("retry-drop", "drop-retry", api_base),
        *pair("retry-header-zero", "status-500-headerzero", api_base),
        deployment("retry-by-id", "status-500-byid", api_base),
        deployment("retry-backoff", "script-backoff", api_base),
        deployment("retry-after-header", "retryafter-lone", api_base),
        *pair("retry-timeout", "slow-retry", api_base, timeout=1),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-retries"), _models, ROUTER_SETTINGS) as value:
        yield value


def test_router_num_retries_two_gives_a_500_three_upstream_attempts_and_returns_the_500(rig: Rig) -> None:
    """The group's two deployments both call an upstream model that always answers 500, with `num_retries: 2`.

    One chat request must reach the upstream exactly three times (one attempt plus two retries). The caller
    gets HTTP 500 with the OpenAI error envelope: `type` `internal_server_error`, `code` `"500"`, `param` null,
    and a message that starts with `litellm.InternalServerError: InternalServerError: OpenAIException - ` and
    carries the upstream's text.
    """
    response: Final = chat(rig.gateway, "retry-500")
    assert response.status_code == 500, response.text
    assert rig.upstream.count("status-500-retry") == NUM_RETRIES + 1
    error: Final = error_of(response)
    assert {key: error[key] for key in ("type", "param", "code")} == {
        "type": "internal_server_error",
        "param": None,
        "code": "500",
    }, response.text
    assert error_message(response).startswith(MESSAGE_PREFIX[500] + "scripted 500 from status-500-retry"), response.text


def test_a_retry_that_succeeds_returns_200_with_attempted_and_max_retries_headers(rig: Rig) -> None:
    """The upstream answers 500 once and 200 after that.

    The caller gets 200 with the second attempt's content, the upstream saw exactly two requests, and the
    response carries `x-litellm-attempted-retries: 1` and `x-litellm-max-retries: 2`. Clients and dashboards
    read these two headers to see that a retry happened.
    """
    rig.upstream.script("script-recover", 500)
    response: Final = chat(rig.gateway, "retry-recover")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served script-recover"
    assert rig.upstream.count("script-recover") == 2
    assert response.headers["x-litellm-attempted-retries"] == "1"
    assert response.headers["x-litellm-max-retries"] == str(NUM_RETRIES)


def test_a_429_is_retried_until_the_budget_is_spent_and_returns_429(rig: Rig) -> None:
    """The upstream always answers 429.

    The upstream must receive three requests and the caller gets HTTP 429 with `code` `"429"`, `type`
    `throttling_error` and a message starting `litellm.RateLimitError: RateLimitError: OpenAIException - `.
    """
    response: Final = chat(rig.gateway, "retry-429")
    assert response.status_code == 429, response.text
    assert rig.upstream.count("status-429-retry") == NUM_RETRIES + 1
    assert error_of(response)["code"] == "429"
    assert error_of(response)["type"] == "throttling_error"
    assert error_message(response).startswith(MESSAGE_PREFIX[429]), response.text


def test_a_503_is_retried_until_the_budget_is_spent_and_returns_503(rig: Rig) -> None:
    """The upstream always answers 503.

    The upstream must receive three requests and the caller gets HTTP 503 with `code` `"503"` and a message
    starting `litellm.ServiceUnavailableError: ServiceUnavailableError: OpenAIException - `.
    """
    response: Final = chat(rig.gateway, "retry-503")
    assert response.status_code == 503, response.text
    assert rig.upstream.count("status-503-retry") == NUM_RETRIES + 1
    assert error_of(response)["code"] == "503"
    assert error_message(response).startswith(MESSAGE_PREFIX[503]), response.text


def test_an_upstream_408_is_retried_until_the_budget_is_spent_and_returns_408(rig: Rig) -> None:
    """The upstream always answers HTTP 408.

    408 is a retryable status, so the upstream must receive three requests. The caller gets HTTP 408 with
    `code` `"408"` and a message starting `litellm.Timeout: Timeout Error: OpenAIException - `.
    """
    response: Final = chat(rig.gateway, "retry-408")
    assert response.status_code == 408, response.text
    assert rig.upstream.count("status-408-retry") == NUM_RETRIES + 1
    assert error_of(response)["code"] == "408"
    assert error_message(response).startswith(MESSAGE_PREFIX[408]), response.text


def test_a_400_is_never_retried_and_returns_400(rig: Rig) -> None:
    """The upstream always answers 400 and the group has two deployments and a retry budget of 2.

    The upstream must receive exactly one request: a bad request would fail the same way on a retry. The
    caller gets HTTP 400 with `code` `"400"` and a message starting `litellm.BadRequestError: OpenAIException - `.
    """
    response: Final = chat(rig.gateway, "retry-400")
    assert response.status_code == 400, response.text
    assert rig.upstream.count("status-400-retry") == 1
    assert error_of(response)["code"] == "400"
    assert error_message(response).startswith(MESSAGE_PREFIX[400] + "scripted 400 from status-400-retry"), response.text


def test_a_404_is_never_retried_and_returns_404(rig: Rig) -> None:
    """The upstream always answers 404 and the group has two deployments and a retry budget of 2.

    The upstream must receive exactly one request and the caller gets HTTP 404 with `code` `"404"` and a
    message starting `litellm.NotFoundError: NotFoundError: OpenAIException - `.
    """
    response: Final = chat(rig.gateway, "retry-404")
    assert response.status_code == 404, response.text
    assert rig.upstream.count("status-404-retry") == 1
    assert error_of(response)["code"] == "404"
    assert error_message(response).startswith(MESSAGE_PREFIX[404]), response.text


def test_a_401_on_a_group_with_one_deployment_is_not_retried(rig: Rig) -> None:
    """The group has a single deployment whose upstream always answers 401.

    The same credential would fail again, so the upstream must receive exactly one request. The caller gets
    HTTP 401 with `code` `"401"` and a message starting
    `litellm.AuthenticationError: AuthenticationError: OpenAIException - `.
    """
    response: Final = chat(rig.gateway, "retry-401-lone")
    assert response.status_code == 401, response.text
    assert rig.upstream.count("status-401-lone") == 1
    assert error_of(response)["code"] == "401"
    assert error_message(response).startswith(MESSAGE_PREFIX[401]), response.text


def test_a_401_is_retried_once_on_each_sibling_deployment(rig: Rig) -> None:
    """The group has three deployments, each calling its own upstream model, and all three answer 401.

    A sibling may hold a working credential, so the router retries, and it never goes back to a deployment
    that already answered with a non-retryable status. With a budget of two retries each of the three upstream
    models must receive exactly one request, and the caller gets HTTP 401.
    """
    response: Final = chat(rig.gateway, "retry-401-siblings")
    assert response.status_code == 401, response.text
    counts: Final = {
        model: rig.upstream.count(model) for model in ("status-401-first", "status-401-second", "status-401-third")
    }
    assert counts == {"status-401-first": 1, "status-401-second": 1, "status-401-third": 1}


def test_a_connection_the_upstream_drops_is_retried_and_returns_500(rig: Rig) -> None:
    """The upstream accepts the request and closes the connection without answering.

    A connection error is retryable, so the upstream must receive three requests. The caller gets HTTP 500
    with `code` `"500"`. UNVERIFIED: that a dropped connection is classified as a connection error with status
    500 rather than a timeout. See the OpenAI branch of `litellm/litellm_core_utils/exception_mapping_utils.py`.
    """
    response: Final = chat(rig.gateway, "retry-drop")
    assert response.status_code == 500, response.text
    assert rig.upstream.count("drop-retry") == NUM_RETRIES + 1
    assert error_of(response)["code"] == "500"


def test_the_x_litellm_num_retries_header_zero_turns_retries_off_for_one_request(rig: Rig) -> None:
    """The router's budget is 2 and the caller sends the request header `x-litellm-num-retries: 0`.

    The upstream, which always answers 500, must receive exactly one request and the caller gets HTTP 500. A
    request's own retry count wins over the router's.
    """
    response: Final = chat(rig.gateway, "retry-header-zero", headers={"x-litellm-num-retries": "0"})
    assert response.status_code == 500, response.text
    assert rig.upstream.count("status-500-headerzero") == 1


def test_a_call_addressed_by_deployment_id_is_attempted_once_while_its_group_name_is_retried(rig: Rig) -> None:
    """One deployment with id `dec1-status-500-byid` in group `retry-by-id` always answers 500.

    Calling the group by name spends the whole budget: three upstream requests. Calling the same deployment
    with `model` set to its id sends exactly one more request, because a call by id counts no healthy
    deployments to retry on. Both calls return HTTP 500. The by-name call waits for backoff, which is why this
    test takes a few seconds.
    """
    by_name: Final = chat(rig.gateway, "retry-by-id")
    assert by_name.status_code == 500, by_name.text
    assert rig.upstream.count("status-500-byid") == NUM_RETRIES + 1
    by_id: Final = chat(rig.gateway, "dec1-status-500-byid")
    assert by_id.status_code == 500, by_id.text
    assert rig.upstream.count("status-500-byid") == NUM_RETRIES + 2


def test_retries_on_a_group_with_one_deployment_back_off_half_a_second_then_one_second(rig: Rig) -> None:
    """The group has a single deployment whose upstream answers 500 twice and then 200.

    The caller gets 200 after three upstream requests. The wait before the first retry must be at least 0.5s
    and before the second at least 1.0s, each with at most 0.75s of jitter on top (a generous 3s of slack is
    allowed for a slow runner). A group with several deployments does not wait at all.
    """
    rig.upstream.script("script-backoff", 500, 500)
    response: Final = chat(rig.gateway, "retry-backoff")
    assert response.status_code == 200, response.text
    first, second, third = (seen.at for seen in rig.upstream.calls("script-backoff"))
    assert 0.5 <= second - first < 0.5 + 0.75 + 3
    assert 1.0 <= third - second < 1.0 + 0.75 + 3
    assert response.headers["x-litellm-attempted-retries"] == "2"


def test_a_provider_retry_after_header_sets_the_wait_before_the_retry(rig: Rig) -> None:
    """The group has a single deployment whose upstream answers 429 with `retry-after: 2` once, then 200.

    The caller gets 200 after two upstream requests, and the second request must arrive at least 2 seconds
    after the first (the backoff alone would be 0.5s to 1.25s). A provider's retry-after of up to 60 seconds
    replaces the computed backoff.
    """
    rig.upstream.script("retryafter-lone", 429)
    response: Final = chat(rig.gateway, "retry-after-header")
    assert response.status_code == 200, response.text
    first, second = (seen.at for seen in rig.upstream.calls("retryafter-lone"))
    assert 2.0 <= second - first < 2.0 + 0.75 + 3


def test_a_deployment_timeout_is_retried_and_returns_408(rig: Rig) -> None:
    """Both deployments set `litellm_params.timeout: 1` and the upstream takes 4 seconds to answer.

    Each attempt times out after a second, a timeout is retryable, so the upstream must receive three requests.
    The caller gets HTTP 408 with `code` `"408"` and a message starting `litellm.Timeout: `.
    """
    response: Final = chat(rig.gateway, "retry-timeout")
    assert response.status_code == 408, response.text
    assert rig.upstream.count("slow-retry") == NUM_RETRIES + 1
    assert error_of(response)["code"] == "408"
    assert error_message(response).startswith("litellm.Timeout: "), response.text
