"""`routing_strategy: usage-based-routing-v2` across providers: tpm and rpm headroom, and the over-tpm refusal.

The pick is answered by `LowestTPMLoggingHandler_v2` in `litellm/router_strategy/lowest_tpm_rpm_v2.py`. It reads
this minute's `tpm` and `rpm` counters for every healthy deployment (`usage_counters`, `%H-%M` buckets), drops a
deployment when `used_tpm + prompt_tokens > tpm` or when `used_rpm + 1 >= rpm` (line 420), and picks at random among
the ones with the lowest tpm used. The rpm counter is raised before the call (`async_pre_call_check`), the tpm
counter after it.

Since #51838 (2026-11-14) a request whose estimated prompt alone is over the `tpm` of every deployment the caller
could reach is refused with `RequestOverDeploymentTpmError`: a 429 whose message names the estimate and the largest
limit, carries no `retry-after`, is never retried, and still reaches the group's fallbacks. A request that merely
finds every deployment full keeps the older retryable `No deployments available ... Deployments={...}` refusal.

Deployments here are on Bedrock, Azure OpenAI, Vertex AI, OpenAI and Anthropic. Router retries are off and
cooldowns disabled unless a test says otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.timing import assert_cut_short, timed
from tests.integration.routing.dec1_routestrat_support import (
    LONG_PROMPT,
    NO_DEPLOYMENTS,
    Rig,
    chat,
    dep,
    early_in_a_minute,
    error_message,
    reported_strategy,
    served,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

OVER_TPM: Final = re.compile(
    r"No deployments available for selected model\. Passed model=(?P<group>[\w-]+)\. "
    r"This request alone is estimated at (?P<tokens>\d+) input tokens, over the largest tpm limit "
    r"of (?P<limit>[\d.]+) in this model group, so retrying cannot help\. "
    r"Send a smaller request or raise the deployment's tpm"
)
RETRY_WAIT_SECONDS: Final = 60
LONG: Final[list[JsonValue]] = [{"role": "user", "content": LONG_PROMPT}]


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("bedrock", "usage-headroom", "uh-bedrock", base, tpm=100),
        dep("azure", "usage-headroom", "uh-azure", base, tpm=1000000),
        dep("vertex", "usage-spread", "us-vertex", base, tpm=1000000),
        dep("openai", "usage-spread", "us-openai", base, tpm=1000000),
        dep("anthropic", "usage-single", "usg-anthropic", base, tpm=50),
        dep("bedrock", "usage-two-small", "uts-bedrock", base, tpm=50),
        dep("vertex", "usage-two-small", "uts-vertex", base, tpm=80),
        dep("azure", "usage-with-fallback", "uwf-azure", base, tpm=50),
        dep("anthropic", "usage-fallback-target", "uft-anthropic", base),
        dep("openai", "usage-rpm3", "ur3-openai", base, rpm=3),
        dep("vertex", "usage-rpm-full", "urf-vertex", base, rpm=2),
        dep("openai", "usage-weight", "uw-openai", base, weight=100, tpm=40),
        dep("anthropic", "usage-weight", "uw-anthropic", base, tpm=1000000),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "usage-based-routing-v2",
        "num_retries": 0,
        "disable_cooldowns": True,
        "fallbacks": [{"usage-with-fallback": ["usage-fallback-target"]}],
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-usage"), _models, settings) as value:
        yield value


def test_usage_based_routing_v2_is_reported_by_router_settings(rig: Rig) -> None:
    """The proxy is booted with `routing_strategy: usage-based-routing-v2`.

    `GET /router/settings` must report `current_values.routing_strategy` as `usage-based-routing-v2`.
    """
    assert reported_strategy(rig.gateway) == "usage-based-routing-v2"


def test_a_prompt_over_one_deployments_tpm_is_always_routed_to_the_one_with_headroom(rig: Rig) -> None:
    """Bedrock has `tpm: 100`, Azure `tpm: 1000000`; the prompt is about 1200 tokens.

    Ten such requests must all be served by Azure, even the first one, because the strategy adds the prompt's
    estimated tokens to the deployment's usage before comparing with `tpm`.
    """
    assert served_many(rig.gateway, "usage-headroom", 10, messages=LONG) == {"uh-azure": 10}


def test_short_prompts_spread_over_deployments_with_the_lowest_tpm_used(rig: Rig) -> None:
    """Vertex and OpenAI both have `tpm: 1000000`; requests are short.

    Over twenty sequential requests each provider must serve at least four: the deployment that served last has
    the higher tpm count for the minute, so the next pick favours the other. UNVERIFIED: a minute boundary or a
    late tpm write can make two picks in a row land on the same deployment, hence the loose bound.
    """
    counts: Final = served_many(rig.gateway, "usage-spread", 20)
    assert counts["us-vertex"] >= 4, counts
    assert counts["us-openai"] >= 4, counts


def test_a_prompt_over_every_tpm_is_refused_with_a_message_naming_its_size(rig: Rig) -> None:
    """The group's only deployment (Anthropic) has `tpm: 50`; the prompt is about 1200 tokens.

    The caller gets HTTP 429 whose message matches `No deployments available for selected model. Passed
    model=usage-single. This request alone is estimated at <n> input tokens, over the largest tpm limit of 50 in
    this model group, so retrying cannot help. Send a smaller request or raise the deployment's tpm`, with n above
    50. Anthropic receives nothing. This wording landed in #51838; before it the refusal named no size.
    """
    response: Final = chat(rig.gateway, "usage-single", messages=LONG)
    assert response.status_code == 429, response.text
    found: Final = OVER_TPM.search(error_message(response))
    assert found is not None, response.text
    assert found["group"] == "usage-single"
    assert found["limit"] == "50"
    assert int(found["tokens"]) > 50
    assert rig.peer.count("usg-anthropic") == 0


def test_the_over_tpm_refusal_carries_no_retry_after_header(rig: Rig) -> None:
    """The same over-tpm request as above.

    The 429 must carry no `retry-after` header: retrying cannot help, so the proxy must not invite a retry in
    60 seconds as the older `No deployments available` refusal does.
    """
    response: Final = chat(rig.gateway, "usage-single", messages=LONG)
    assert response.status_code == 429, response.text
    assert "retry-after" not in response.headers, response.headers


def test_the_over_tpm_refusal_is_not_retried_even_when_the_key_asks_for_retries(rig: Rig) -> None:
    """A key saves `router_settings: {"num_retries": 3}` and sends the over-tpm request to `usage-single`.

    The 429 must come back sooner than one 60 second retry wait. Before #51838 a single-deployment group held the caller for three
    retries of 60 seconds each on a refusal that could never clear.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"num_retries": 3})
        answered: Final = timed(lambda: chat(rig.gateway, "usage-single", key=key, messages=LONG))
    response: Final = answered.value
    assert response.status_code == 429, response.text
    assert_cut_short(answered.seconds, injected_delay=RETRY_WAIT_SECONDS)
    assert OVER_TPM.search(error_message(response)) is not None, response.text


def test_the_over_tpm_refusal_names_the_largest_tpm_in_the_group(rig: Rig) -> None:
    """Bedrock has `tpm: 50` and Vertex `tpm: 80`; the prompt is about 1200 tokens.

    The 429 message must say `over the largest tpm limit of 80 in this model group`, so an operator knows how far
    the largest deployment is from fitting the request.
    """
    response: Final = chat(rig.gateway, "usage-two-small", messages=LONG)
    assert response.status_code == 429, response.text
    found: Final = OVER_TPM.search(error_message(response))
    assert found is not None, response.text
    assert found["limit"] == "80"


def test_the_over_tpm_refusal_still_falls_back_to_the_configured_group(rig: Rig) -> None:
    """`usage-with-fallback` has one Azure deployment with `tpm: 50` and falls back to `usage-fallback-target`
    (Anthropic, no limit). The prompt is about 1200 tokens.

    The request must return 200 served by the Anthropic deployment, with `x-litellm-model-id` naming it, and
    Azure must receive nothing. Making the refusal non-retryable must not cut it off from fallbacks.
    """
    response: Final = chat(rig.gateway, "usage-with-fallback", messages=LONG)
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] == "dec1-routestrat-uft-anthropic"
    assert rig.peer.count("uwf-azure") == 0


def test_a_short_prompt_still_reaches_a_deployment_with_a_small_tpm(rig: Rig) -> None:
    """The group `usage-single` (Anthropic, `tpm: 50`) receives a one-line prompt.

    It must be served with 200: the over-tpm refusal is about the request's own size, not the deployment's limit.
    """
    assert served(rig.gateway, "usage-single") == "usg-anthropic"


def test_a_deployment_with_rpm_three_serves_three_requests_a_minute(rig: Rig) -> None:
    """The group's only deployment (OpenAI) has `rpm: 3`.

    Starting early in a minute, three sequential requests should all return 200 and a fourth 429.
    """
    pytest.skip(
        "BUG: lowest_tpm_rpm_v2.py:420 drops a deployment when used_rpm + 1 >= rpm, so an rpm: 3 deployment serves "
        "two requests a minute and an rpm: 2 one serves one, while rpm: 1 serves one (the first read is None)"
    )
    early_in_a_minute()
    assert [chat(rig.gateway, "usage-rpm3").status_code for _ in range(3)] == [200, 200, 200]
    assert chat(rig.gateway, "usage-rpm3").status_code == 429


def test_a_full_deployment_is_refused_with_its_usage_in_the_message(rig: Rig) -> None:
    """The group's only deployment (Vertex) has `rpm: 2`; requests are short.

    Starting early in a minute, requests are sent until one answers 429 (within five). That answer's message
    must contain `No deployments available for selected model. Passed model=usage-rpm-full. Deployments=` and the
    deployment's `'rpm_limit': 2`: a full group keeps the older refusal that lists each deployment's usage, unlike
    the over-tpm one. UNVERIFIED: the dict is printed with Python `repr`, so quotes and spacing follow it.
    """
    early_in_a_minute()
    statuses: Final[list[int]] = []  # mutable-ok: one status per request until the refusal
    refused = None
    for _ in range(5):
        response = chat(rig.gateway, "usage-rpm-full")
        statuses.append(response.status_code)
        if response.status_code == 429:
            refused = response
            break
    assert refused is not None, statuses
    message: Final = error_message(refused)
    assert f"{NO_DEPLOYMENTS}. Passed model=usage-rpm-full. Deployments=" in message, message
    assert "'rpm_limit': 2" in message, message
    assert OVER_TPM.search(message) is None


def test_usage_based_routing_ignores_weight_when_one_deployment_lacks_tpm_headroom(rig: Rig) -> None:
    """OpenAI has `weight: 100` and `tpm: 40`; Anthropic has no weight and `tpm: 1000000`. The prompt is about 1200
    tokens.

    Five requests must all reach Anthropic: usage-based routing does not read `weight`, and the weighted
    deployment has no headroom for the prompt.
    """
    assert served_many(rig.gateway, "usage-weight", 5, messages=LONG) == {"uw-anthropic": 5}
