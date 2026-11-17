"""Prometheus /metrics after successful, streamed, cached and embedding calls must not change when the
Prometheus logger moves from the Python CustomLogger to a native CustomLoggerV2 callback.

Contract. With `callbacks: [prometheus]` the proxy exposes on `/metrics` one counter increment per successful
call on `litellm_requests_metric`, the response cost on `litellm_spend_metric`, the provider usage on the three
token counters, one observation on the request and provider latency histograms, `litellm_proxy_total_requests_metric`
with status_code "200", deployment health gauges and counters, the provider `x-ratelimit-remaining-*` headers, the
configured deployment rpm, the virtual key remaining gauges and cache hit and miss counters. Label names, their
values for a virtual key, and the text exposition format (whole numbers written as `1.0`, `_created` samples) are
what dashboards key on, and the native port (step1/prometheus.md) rewrites every one of them.

Every test makes its own virtual key with a unique alias, so its series are its own even though the rig is
shared. Expected values come from the scripted provider (11 prompt and 4 completion tokens) and the custom prices
(0.001 and 0.002 per token), so a chat costs 0.019.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Scenario, eventually, held
from integration._support.prometheus_series import scrape
from integration.observability.dec1_cblv2_support import (
    CHAT,
    CHAT_COST,
    CHAT_RPM_LIMITED,
    EMBEDDING,
    EMBEDDING_TOKENS,
    PROVIDER_REMAINING_REQUESTS,
    PROVIDER_REMAINING_TOKENS,
    UPSTREAM_MODEL,
    Rig,
    cblv2_rig,
    chat,
    deployment_id,
    key_hash,
    label_names,
    marker,
    series,
    stream_chat,
    until_series,
    until_value,
    values,
)
from pydantic import JsonValue

REQUESTS: Final = "litellm_requests_metric_total"
SPEND: Final = "litellm_spend_metric_total"
TOTAL_TOKENS: Final = "litellm_total_tokens_metric_total"
INPUT_TOKENS: Final = "litellm_input_tokens_metric_total"
OUTPUT_TOKENS: Final = "litellm_output_tokens_metric_total"
PROXY_TOTAL: Final = "litellm_proxy_total_requests_metric_total"
TOTAL_LATENCY: Final = "litellm_request_total_latency_metric"
API_LATENCY: Final = "litellm_llm_api_latency_metric"
TTFT: Final = "litellm_llm_api_time_to_first_token_metric"
STATE: Final = "litellm_deployment_state"
DEPLOYMENT_SUCCESS: Final = "litellm_deployment_success_responses_total"
DEPLOYMENT_TOTAL: Final = "litellm_deployment_total_requests_total"
PER_OUTPUT_TOKEN: Final = "litellm_deployment_latency_per_output_token"
REMAINING_REQUESTS: Final = "litellm_remaining_requests_metric"
REMAINING_TOKENS: Final = "litellm_remaining_tokens_metric"
RPM_LIMIT: Final = "litellm_deployment_rpm_limit"
KEY_REMAINING_REQUESTS: Final = "litellm_remaining_api_key_requests_for_model"
KEY_REMAINING_TOKENS: Final = "litellm_remaining_api_key_tokens_for_model"
CACHE_HITS: Final = "litellm_cache_hits_metric_total"
CACHE_MISSES: Final = "litellm_cache_misses_metric_total"
CACHED_TOKENS: Final = "litellm_cached_tokens_metric_total"
KEY_REMAINING_BUDGET: Final = "litellm_remaining_api_key_budget_metric"
KEY_MAX_BUDGET: Final = "litellm_api_key_max_budget_metric"


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("cblv2-prometheus-success")
    with cblv2_rig(
        directory,
        {"callbacks": ["prometheus"], "cache": True, "cache_params": {"type": "local"}},
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def scenario(rig: Rig) -> Iterator[Scenario]:
    """One scenario for the module, so every key, team and user the tests make is cleaned up once at the end."""
    with rig.proxy.scenario() as opened:
        yield opened


def _key(scenario: Scenario, **fields: JsonValue) -> tuple[str, str, str]:
    """A fresh virtual key with a unique alias: (key, alias, sha256 hash)."""
    alias: Final = f"cblv2-prom-{uuid.uuid4().hex[:10]}"
    key: Final = scenario.key(key_alias=alias, **fields)
    return key, alias, key_hash(key)


def _ok(rig: Rig, key: str, **extra: JsonValue) -> None:
    response: Final = chat(rig, marker(), key=key, **extra)
    assert response.status_code == 200, response.text


def test_successful_chat_counts_one_request_with_the_key_labels(rig: Rig, scenario: Scenario) -> None:
    """One successful chat on a fresh key must add exactly 1 to `litellm_requests_metric_total` on the series whose
    labels are hashed_api_key=<sha256 of the key>, api_key_alias=<alias>, requested_model="cblv2-chat",
    model="gpt-4o-mini", model_id="cblv2-chat-id", api_provider="openai", team="None" and user="None"."""
    key, alias, hashed = _key(scenario)
    _ok(rig, key)
    found: Final = until_value(
        rig,
        REQUESTS,
        1.0,
        hashed_api_key=hashed,
        api_key_alias=alias,
        requested_model=CHAT,
        model=UPSTREAM_MODEL,
        model_id=deployment_id(CHAT),
        api_provider="openai",
        team="None",
        user="None",
    )
    assert len(found) == 1


def test_requests_metric_label_set_is_the_documented_one(rig: Rig, scenario: Scenario) -> None:
    """The series of `litellm_requests_metric_total` must carry exactly the label names of
    PrometheusMetricLabels.litellm_requests_metric plus org_id and org_alias, which get_labels appends to every
    metric in `_org_label_metrics`. A native port that adds, drops or renames a label breaks every dashboard query."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    sample: Final = until_series(rig, REQUESTS, hashed_api_key=hashed)[0]
    assert label_names(sample) == {
        "end_user",
        "hashed_api_key",
        "api_key_alias",
        "model",
        "team",
        "team_alias",
        "user",
        "user_email",
        "client_ip",
        "user_agent",
        "requested_model",
        "model_id",
        "api_provider",
        "org_id",
        "org_alias",
    }


def test_spend_metric_adds_the_response_cost(rig: Rig, scenario: Scenario) -> None:
    """One chat at 11 prompt and 4 completion tokens and custom prices 0.001 and 0.002 must add 0.019 to
    `litellm_spend_metric_total` for the key. The spend counter is the cost the logging payload reports, not a
    value recomputed by the metrics layer."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    found: Final = until_series(rig, SPEND, hashed_api_key=hashed)
    assert [sample.value for sample in found] == [pytest.approx(CHAT_COST)]


def test_spend_metric_label_set_adds_service_tier(rig: Rig, scenario: Scenario) -> None:
    """`litellm_spend_metric` has the request labels plus `service_tier`, which `litellm_requests_metric` does not
    have. The two metrics are joined in dashboards, so their label sets must stay exactly as declared."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    sample: Final = until_series(rig, SPEND, hashed_api_key=hashed)[0]
    requests: Final = until_series(rig, REQUESTS, hashed_api_key=hashed)[0]
    assert label_names(sample) - label_names(requests) == {"service_tier"}
    assert label_names(requests) - label_names(sample) == set()


def test_token_counters_add_prompt_completion_and_total_tokens(rig: Rig, scenario: Scenario) -> None:
    """One chat must add 11 to `litellm_input_tokens_metric_total`, 4 to `litellm_output_tokens_metric_total` and
    15 to `litellm_total_tokens_metric_total`, all on the key's series. The counters come from the provider usage,
    not from a local token count."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_value(rig, TOTAL_TOKENS, 15.0, hashed_api_key=hashed)
    samples: Final = scrape(rig.proxy)
    assert values(samples, INPUT_TOKENS, hashed_api_key=hashed) == [11.0]
    assert values(samples, OUTPUT_TOKENS, hashed_api_key=hashed) == [4.0]


def test_proxy_total_requests_counts_status_200_on_the_route(rig: Rig, scenario: Scenario) -> None:
    """A successful chat must add 1 to `litellm_proxy_total_requests_metric_total` on the series with
    status_code="200", route="/v1/chat/completions" and requested_model="cblv2-chat" for the key."""
    key, alias, hashed = _key(scenario)
    _ok(rig, key)
    until_value(
        rig,
        PROXY_TOTAL,
        1.0,
        hashed_api_key=hashed,
        api_key_alias=alias,
        status_code="200",
        route="/v1/chat/completions",
        requested_model=CHAT,
    )


def test_two_calls_on_one_key_add_up_on_one_series(rig: Rig, scenario: Scenario) -> None:
    """Two successful chats on the same key must leave one `litellm_requests_metric_total` series at 2.0 and one
    spend series at 0.038, not two series: the label values for the same key and model are stable across calls."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    _ok(rig, key)
    until_value(rig, REQUESTS, 2.0, hashed_api_key=hashed)
    assert values(scrape(rig.proxy), SPEND, hashed_api_key=hashed) == [pytest.approx(2 * CHAT_COST)]


def test_latency_histograms_count_one_observation_per_call(rig: Rig, scenario: Scenario) -> None:
    """One chat must add exactly one observation to `litellm_request_total_latency_metric` and to
    `litellm_llm_api_latency_metric` for the key: each `_count` is 1.0 and each `_sum` is positive."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    for name in (TOTAL_LATENCY, API_LATENCY):
        until_value(rig, f"{name}_count", 1.0, hashed_api_key=hashed)
        assert values(scrape(rig.proxy), f"{name}_sum", hashed_api_key=hashed)[0] > 0, name


def test_request_total_latency_label_set_is_the_documented_one(rig: Rig, scenario: Scenario) -> None:
    """`litellm_request_total_latency_metric` series must carry exactly model_group, end_user, hashed_api_key,
    api_key_alias, requested_model, team, team_alias, user, model, model_id, api_provider, service_tier, org_id and
    org_alias (plus `le` on buckets). The input_sequence_length label is off by default."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    sample: Final = until_series(rig, f"{TOTAL_LATENCY}_count", hashed_api_key=hashed)[0]
    assert label_names(sample) == {
        "model_group",
        "end_user",
        "hashed_api_key",
        "api_key_alias",
        "requested_model",
        "team",
        "team_alias",
        "user",
        "model",
        "model_id",
        "api_provider",
        "service_tier",
        "org_id",
        "org_alias",
    }


def test_histogram_buckets_are_written_with_a_decimal_point(rig: Rig, scenario: Scenario) -> None:
    """The raw `/metrics` text must write whole-number bucket bounds the way prometheus_client does, for example
    `le="1.0"` and `le="+Inf"`, never `le="1"`. The native port writes its own exposition text (`number()` in the
    gateway prometheus module), and a different spelling silently splits every histogram query."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_series(rig, f"{TOTAL_LATENCY}_count", hashed_api_key=hashed)
    text: Final = rig.proxy.client.get("/metrics", headers={"Authorization": f"Bearer {rig.proxy.key}"}).text
    bucket_lines: Final = [
        line for line in text.splitlines() if line.startswith(f"{TOTAL_LATENCY}_bucket") and hashed in line
    ]
    assert any('le="1.0"' in line for line in bucket_lines), bucket_lines[:3]
    assert any('le="+Inf"' in line for line in bucket_lines), bucket_lines[:3]
    assert not any('le="1"' in line for line in bucket_lines), bucket_lines[:3]


def test_counters_expose_created_samples(rig: Rig, scenario: Scenario) -> None:
    """With prometheus_client 0.20 every counter exposes a `<name>_created` sample next to `<name>_total`, so
    `litellm_requests_metric_created` exists for the key. The native gateway module writes no `_created` samples,
    so this pins today's output and makes the change visible; an owner may decide to drop it deliberately."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_series(rig, REQUESTS, hashed_api_key=hashed)
    assert series(scrape(rig.proxy), "litellm_requests_metric_created", hashed_api_key=hashed)


def test_non_streamed_call_has_no_time_to_first_token_series(rig: Rig, scenario: Scenario) -> None:
    """A non-streamed chat must not create a `litellm_llm_api_time_to_first_token_metric` series for the key: the
    histogram is observed only for streamed calls."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_series(rig, f"{API_LATENCY}_count", hashed_api_key=hashed)
    assert series(scrape(rig.proxy), f"{TTFT}_count", hashed_api_key=hashed) == ()


def test_streamed_call_counts_once_and_observes_time_to_first_token(rig: Rig, scenario: Scenario) -> None:
    """A streamed chat read to the end must add exactly 1 to `litellm_requests_metric_total` and exactly one
    observation to `litellm_llm_api_time_to_first_token_metric` for the key. A port that logs per chunk, or that
    drops the stream-only histogram, changes both."""
    key, _, hashed = _key(scenario)
    status, _, _ = stream_chat(rig, marker(), key=key)
    assert status == 200
    until_value(rig, REQUESTS, 1.0, hashed_api_key=hashed)
    until_value(rig, f"{TTFT}_count", 1.0, hashed_api_key=hashed)
    held(
        lambda: values(scrape(rig.proxy), REQUESTS, hashed_api_key=hashed),
        lambda found: found == [1.0],
        holding=3,
    )


def test_streamed_call_tokens_and_spend_match_the_non_streamed_call(rig: Rig, scenario: Scenario) -> None:
    """A streamed chat whose final chunk carries usage 11/4 must add 15 total tokens and 0.019 spend for the key,
    the same as the non-streamed call: usage is taken from the assembled stream."""
    key, _, hashed = _key(scenario)
    assert stream_chat(rig, marker(), key=key)[0] == 200
    until_value(rig, TOTAL_TOKENS, 15.0, hashed_api_key=hashed)
    assert values(scrape(rig.proxy), SPEND, hashed_api_key=hashed) == [pytest.approx(CHAT_COST)]


def test_streamed_call_counts_a_200_on_proxy_total_requests(rig: Rig, scenario: Scenario) -> None:
    """A streamed chat must add 1 to `litellm_proxy_total_requests_metric_total` with status_code="200" for the key,
    once the stream has been read to the end."""
    key, _, hashed = _key(scenario)
    assert stream_chat(rig, marker(), key=key)[0] == 200
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="200")


def test_deployment_state_is_zero_after_a_success(rig: Rig, scenario: Scenario) -> None:
    """After a successful chat the `litellm_deployment_state` gauge for model_id="cblv2-chat-id" must be 0
    (healthy), labelled litellm_model_name="gpt-4o-mini" and api_provider="openai"."""
    key, _, _ = _key(scenario)
    _ok(rig, key)
    until_value(
        rig,
        STATE,
        0.0,
        model_id=deployment_id(CHAT),
        litellm_model_name=UPSTREAM_MODEL,
        api_provider="openai",
    )


def test_deployment_state_label_set_is_model_id_base_and_provider(rig: Rig, scenario: Scenario) -> None:
    """`litellm_deployment_state` must carry exactly litellm_model_name, model_id, api_base and api_provider, and
    its api_base is the deployment base without a trailing slash."""
    key, _, _ = _key(scenario)
    _ok(rig, key)
    sample: Final = until_series(rig, STATE, model_id=deployment_id(CHAT))[0]
    assert label_names(sample) == {"litellm_model_name", "model_id", "api_base", "api_provider"}
    assert sample.labels["api_base"].endswith("/v1"), sample.labels


def test_deployment_success_and_total_requests_count_the_call(rig: Rig, scenario: Scenario) -> None:
    """One chat on a fresh key must add 1 to `litellm_deployment_success_responses_total` and 1 to
    `litellm_deployment_total_requests_total` on the series with model_id="cblv2-chat-id", model_group and
    requested_model "cblv2-chat" and the key's hash."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    labels: Final = {
        "model_id": deployment_id(CHAT),
        "model_group": CHAT,
        "requested_model": CHAT,
        "hashed_api_key": hashed,
    }
    until_value(rig, DEPLOYMENT_SUCCESS, 1.0, **labels)
    until_value(rig, DEPLOYMENT_TOTAL, 1.0, **labels)


def test_deployment_latency_per_output_token_is_observed_once(rig: Rig, scenario: Scenario) -> None:
    """One chat with 4 completion tokens must add one observation to `litellm_deployment_latency_per_output_token`
    for the key's series (`_count` 1.0, `_sum` positive)."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_value(rig, f"{PER_OUTPUT_TOKEN}_count", 1.0, hashed_api_key=hashed)
    assert values(scrape(rig.proxy), f"{PER_OUTPUT_TOKEN}_sum", hashed_api_key=hashed)[0] > 0


def test_provider_remaining_gauges_take_the_provider_rate_limit_headers(rig: Rig, scenario: Scenario) -> None:
    """The scripted provider answers with `x-ratelimit-remaining-requests: 97` and
    `x-ratelimit-remaining-tokens: 9000`. After a chat, `litellm_remaining_requests_metric` and
    `litellm_remaining_tokens_metric` for the key and model_id="cblv2-chat-id" must be 97 and 9000.
    UNVERIFIED: read from set_llm_deployment_success_metrics (litellm/integrations/prometheus.py:3735); the
    header reaches hidden_params additional_headers as `x_ratelimit_remaining_requests`."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_value(
        rig, REMAINING_REQUESTS, float(PROVIDER_REMAINING_REQUESTS), hashed_api_key=hashed, model_id=deployment_id(CHAT)
    )
    until_value(
        rig, REMAINING_TOKENS, float(PROVIDER_REMAINING_TOKENS), hashed_api_key=hashed, model_id=deployment_id(CHAT)
    )


def test_deployment_rpm_limit_gauge_reports_the_configured_rpm(rig: Rig, scenario: Scenario) -> None:
    """The deployment "cblv2-chat-rpm-limited" is configured with rpm 1. After one call through it,
    `litellm_deployment_rpm_limit` for model_id="cblv2-chat-rpm-limited-id" and model_group
    "cblv2-chat-rpm-limited" must be 1.0."""
    key, _, _ = _key(scenario)
    _ok(rig, key, model=CHAT_RPM_LIMITED)
    until_value(rig, RPM_LIMIT, 1.0, model_id=deployment_id(CHAT_RPM_LIMITED), model_group=CHAT_RPM_LIMITED)


def test_key_without_limits_reports_maxsize_remaining_requests_and_tokens(rig: Rig, scenario: Scenario) -> None:
    """A key with no rpm or tpm limit must export `litellm_remaining_api_key_requests_for_model` and
    `..._tokens_for_model` at sys.maxsize (9223372036854775807) with model="cblv2-chat" (the model group, not the
    provider model). Read from `_set_virtual_key_rate_limit_metrics` (litellm/integrations/prometheus.py:2740)."""
    key, alias, hashed = _key(scenario)
    _ok(rig, key)
    labels: Final = {"hashed_api_key": hashed, "api_key_alias": alias, "model": CHAT}
    until_value(rig, KEY_REMAINING_REQUESTS, float(sys.maxsize), **labels)
    until_value(rig, KEY_REMAINING_TOKENS, float(sys.maxsize), **labels)


def test_key_remaining_requests_gauge_has_no_model_id_label(rig: Rig, scenario: Scenario) -> None:
    """`litellm_remaining_api_key_requests_for_model` is declared twice in PrometheusMetricLabels
    (litellm/types/integrations/prometheus.py:803 and :854) and the second declaration, without model_id, wins.
    Its series must carry exactly hashed_api_key, api_key_alias and model. A port that reads the first
    declaration adds a label."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    sample: Final = until_series(rig, KEY_REMAINING_REQUESTS, hashed_api_key=hashed)[0]
    assert label_names(sample) == {"hashed_api_key", "api_key_alias", "model"}


def test_key_with_rpm_limit_reports_its_remaining_requests_after_a_call(rig: Rig, scenario: Scenario) -> None:
    """A key with `rpm_limit: 5` must, after one chat, export `litellm_remaining_api_key_requests_for_model` below
    sys.maxsize and at most 4 for model="cblv2-chat". UNVERIFIED: the value comes from the v3 limiter's
    `x-ratelimit-remaining-requests` in hidden_params when metadata has no per-model variable
    (`_get_remaining_from_v3_rate_limit_headers`); the exact number depends on which descriptor the limiter
    reports first."""
    key, _, hashed = _key(scenario, rpm_limit=5)
    _ok(rig, key)
    found: Final = eventually(
        lambda: values(scrape(rig.proxy), KEY_REMAINING_REQUESTS, hashed_api_key=hashed, model=CHAT),
        lambda found: len(found) == 1 and found[0] < float(sys.maxsize),
        seconds=30,
    )
    assert found[0] <= 4.0


def test_cache_miss_then_hit_counts_one_miss_and_one_hit(rig: Rig, scenario: Scenario) -> None:
    """With a local response cache, two identical chats on one key must add 1 to `litellm_cache_misses_metric_total`
    and 1 to `litellm_cache_hits_metric_total` for the key, and the hit must add 15 (the cached response's total
    tokens) to `litellm_cached_tokens_metric_total`."""
    key, _, hashed = _key(scenario)
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 200
    assert chat(rig, text, key=key).status_code == 200
    until_value(rig, CACHE_HITS, 1.0, hashed_api_key=hashed)
    samples: Final = scrape(rig.proxy)
    assert values(samples, CACHE_MISSES, hashed_api_key=hashed) == [1.0]
    assert values(samples, CACHED_TOKENS, hashed_api_key=hashed) == [15.0]


def test_cache_hit_counts_a_request_but_adds_no_spend(rig: Rig, scenario: Scenario) -> None:
    """A cache miss followed by a hit must leave `litellm_requests_metric_total` at 2 for the key but
    `litellm_spend_metric_total` at 0.019, because the hit's response_cost is 0."""
    key, _, hashed = _key(scenario)
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 200
    assert chat(rig, text, key=key).status_code == 200
    until_value(rig, REQUESTS, 2.0, hashed_api_key=hashed)
    assert values(scrape(rig.proxy), SPEND, hashed_api_key=hashed) == [pytest.approx(CHAT_COST)]


def test_cache_hit_does_not_count_a_provider_deployment_success(rig: Rig, scenario: Scenario) -> None:
    """A cache hit never reaches the provider, so after a miss and a hit the key's
    `litellm_deployment_success_responses_total` series must stay at 1.0. UNVERIFIED: a cache hit still runs
    async_log_success_event; whether set_llm_deployment_success_metrics returns early for it was not traced."""
    key, _, hashed = _key(scenario)
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 200
    assert chat(rig, text, key=key).status_code == 200
    until_value(rig, CACHE_HITS, 1.0, hashed_api_key=hashed)
    held(
        lambda: values(scrape(rig.proxy), DEPLOYMENT_SUCCESS, hashed_api_key=hashed),
        lambda found: found == [1.0],
        holding=3,
    )


def test_end_user_label_carries_the_request_user_field(rig: Rig, scenario: Scenario) -> None:
    """A chat with body `user: <id>` must put that id in the `end_user` label of `litellm_requests_metric_total`
    for the key, since end-user tracking for Prometheus is on by default."""
    key, _, hashed = _key(scenario)
    end_user: Final = f"cblv2-end-user-{uuid.uuid4().hex[:8]}"
    _ok(rig, key, user=end_user)
    until_value(rig, REQUESTS, 1.0, hashed_api_key=hashed, end_user=end_user)


def test_team_key_labels_its_team_id_and_alias(rig: Rig, scenario: Scenario) -> None:
    """A key that belongs to a team with alias A must label its `litellm_requests_metric_total` series with
    team=<team_id> and team_alias=A."""
    alias: Final = f"cblv2-team-{uuid.uuid4().hex[:8]}"
    team: Final = scenario.team(team_alias=alias)
    key: Final = scenario.key(team_id=team)
    _ok(rig, key)
    until_value(rig, REQUESTS, 1.0, hashed_api_key=key_hash(key), team=team, team_alias=alias)


def test_internal_user_key_labels_the_user_id(rig: Rig, scenario: Scenario) -> None:
    """A key owned by user U must label its `litellm_requests_metric_total` series with user=U."""
    user: Final = f"cblv2-user-{uuid.uuid4().hex[:8]}"
    scenario.user(user_id=user)
    key: Final = scenario.key(user_id=user)
    _ok(rig, key)
    until_value(rig, REQUESTS, 1.0, hashed_api_key=key_hash(key), user=user)


def test_key_budget_gauges_report_max_and_remaining_after_a_call(rig: Rig, scenario: Scenario) -> None:
    """A key with max_budget 1.0 must, after one 0.019 chat, export `litellm_api_key_max_budget_metric` 1.0 and
    `litellm_remaining_api_key_budget_metric` 0.981 for its hash. The remaining value reads committed spend, which
    now comes from the Rust budget engine (#51535)."""
    key, _, hashed = _key(scenario, max_budget=1.0)
    _ok(rig, key)
    eventually(
        lambda: values(scrape(rig.proxy), KEY_REMAINING_BUDGET, hashed_api_key=hashed),
        lambda found: found == [pytest.approx(1.0 - CHAT_COST, abs=1e-6)],
        seconds=60,
    )
    assert values(scrape(rig.proxy), KEY_MAX_BUDGET, hashed_api_key=hashed) == [1.0]


def test_embedding_counts_a_request_with_its_input_tokens(rig: Rig, scenario: Scenario) -> None:
    """One embedding call with 5 prompt tokens must add 1 to `litellm_requests_metric_total` with
    requested_model="cblv2-embedding" and 5 to `litellm_input_tokens_metric_total` for the key."""
    key, _, hashed = _key(scenario)
    response: Final = rig.proxy.request("POST", "/v1/embeddings", {"model": EMBEDDING, "input": marker()}, key=key)
    assert response.status_code == 200, response.text
    until_value(rig, REQUESTS, 1.0, hashed_api_key=hashed, requested_model=EMBEDDING)
    assert values(scrape(rig.proxy), INPUT_TOKENS, hashed_api_key=hashed) == [float(EMBEDDING_TOKENS)]


def test_one_success_exposes_every_core_metric_family(rig: Rig, scenario: Scenario) -> None:
    """After one successful chat the scrape must contain each of these families: litellm_requests_metric,
    litellm_spend_metric, litellm_total_tokens_metric, litellm_input_tokens_metric, litellm_output_tokens_metric,
    litellm_proxy_total_requests_metric, litellm_request_total_latency_metric, litellm_llm_api_latency_metric,
    litellm_overhead_latency_metric, litellm_deployment_state, litellm_deployment_success_responses,
    litellm_deployment_total_requests and litellm_deployment_latency_per_output_token. The native module on the
    customer branch emits only 18 of 89 families, so a missing family is the most likely port regression."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_series(rig, REQUESTS, hashed_api_key=hashed)
    families: Final = {sample.family for sample in scrape(rig.proxy)}
    assert {
        "litellm_requests_metric",
        "litellm_spend_metric",
        "litellm_total_tokens_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_proxy_total_requests_metric",
        "litellm_request_total_latency_metric",
        "litellm_llm_api_latency_metric",
        "litellm_overhead_latency_metric",
        "litellm_deployment_state",
        "litellm_deployment_success_responses",
        "litellm_deployment_total_requests",
        "litellm_deployment_latency_per_output_token",
    } <= families


def test_metrics_endpoint_requires_a_key_by_default(rig: Rig, scenario: Scenario) -> None:
    """`require_auth_for_metrics_endpoint` defaults to true, so `GET /metrics` without an Authorization header must
    not return the exposition: the status is 401 and the body has no `litellm_requests_metric` text."""
    response: Final = rig.proxy.client.get("/metrics", follow_redirects=True)
    assert response.status_code == 401, response.status_code
    assert "litellm_requests_metric" not in response.text


def test_metrics_name_filter_returns_only_the_named_family(rig: Rig, scenario: Scenario) -> None:
    """`GET /metrics?name[]=litellm_spend_metric_total` must return the spend family and not the requests family:
    the endpoint applies prometheus_client's `name[]` filter (litellm/integrations/prometheus_metrics_endpoint.py)."""
    key, _, hashed = _key(scenario)
    _ok(rig, key)
    until_series(rig, SPEND, hashed_api_key=hashed)
    response: Final = rig.proxy.client.get(
        "/metrics",
        params={"name[]": "litellm_spend_metric_total"},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "litellm_spend_metric_total{" in response.text
    assert "litellm_requests_metric_total{" not in response.text
