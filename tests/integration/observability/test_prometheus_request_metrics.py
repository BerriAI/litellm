from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml

from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.types.integrations.prometheus import LATENCY_BUCKETS
from tests.integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.prometheus_series import (
    OVERFLOW,
    PROXY_FAILURES,
    PROXY_REQUESTS,
    Sample,
    label_values,
    scrape,
)
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_GOOD: Final = "prometheus-good-endpoint"
_LIMITED: Final = "prometheus-rate-limited-endpoint"
_FAILING: Final = "prometheus-failing-endpoint"
_LATENCY: Final = "prometheus-latency-endpoint"
_DEPLOYMENT: Final = "openai/prometheus-scripted-chat"
_DEPLOYMENT_FAILURE_PROMPT: Final = "return a provider error for this deployment"
_IMAGE_DEPLOYMENT: Final = "openai/gpt-image-1"
_IMAGE_PROMPT: Final = "scripted multipart provider failure"
_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jf6sAAAAASUVORK5CYII="
)
_BILLED_IMAGE: Final = "gemini-3.1-flash-image"
_BILLED_IMAGE_GROUP: Final = "prometheus-billed-image"
_BILLED_IMAGE_PATH: Final = f"/v1/models/{_BILLED_IMAGE}:generateContent"
_BILLED_IMAGE_PROMPT: Final = "scripted image request that returns only text"
_BILLED_INPUT_RATE: Final = 0.001
_BILLED_OUTPUT_RATE: Final = 0.002
_END_USER: Final = f"prometheus-end-user-{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    upstream: Wire


def _respond(request: Request) -> Reply:
    if request.target == "/v1/images/edits":
        return Reply(status=429, body=b'{"error":{"message":"rate limited","type":"rate_limit_error"}}')
    if request.target == _BILLED_IMAGE_PATH:
        return Reply(
            body=json.dumps(
                {
                    "candidates": [
                        {"finishReason": "NO_IMAGE", "content": {"role": "model", "parts": [{"text": "No image"}]}}
                    ],
                    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5, "totalTokenCount": 17},
                }
            ).encode()
        )
    body: Final = json.loads(request.body)
    if (
        body["model"] == _DEPLOYMENT.removeprefix("openai/")
        and body["messages"][0]["content"] == _DEPLOYMENT_FAILURE_PROMPT
    ):
        return Reply(status=503, body=b'{"error":{"message":"provider unavailable","type":"server_error"}}')
    if body["model"] == "429":
        return Reply(
            status=429,
            body=json.dumps({"error": {"message": "rate limited", "type": "rate_limit_error", "code": "429"}}).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "metered"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    params: Final = {"api_key": "sk-fixture", "api_base": f"{upstream.url}/v1"}
    configuration["model_list"] = [
        *configuration["model_list"],
        *(
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                    **params,
                },
            }
            for name in (_GOOD, _LATENCY)
        ),
        {"model_name": _LIMITED, "litellm_params": {"model": "openai/429", **params}},
        {"model_name": _FAILING, "litellm_params": {"model": "openai/429", **params}},
        {"model_name": "prometheus-chat-alias", "litellm_params": {"model": _DEPLOYMENT, **params}},
        {"model_name": "prometheus-image-alias", "litellm_params": {"model": _IMAGE_DEPLOYMENT, **params}},
        {
            "model_name": _BILLED_IMAGE_GROUP,
            "litellm_params": {
                "model": f"gemini/{_BILLED_IMAGE}",
                "input_cost_per_token": _BILLED_INPUT_RATE,
                "output_cost_per_token": _BILLED_OUTPUT_RATE,
                **params,
            },
        },
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration["litellm_settings"]["disable_end_user_cost_tracking_prometheus_only"] = True
    configuration.setdefault("router_settings", {})["num_retries"] = 0
    path: Final = directory / "prometheus-request-metrics.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-request-metrics")
    with (
        wire_server(_respond) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=_config(directory, upstream)) as owned,
    ):
        yield _Rig(owned, upstream)


def _ask(rig: _Rig, model: str, key: str | None = None, **extra: list[str] | str) -> httpx.Response:
    return rig.gateway.request(
        "POST",
        "/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"metrics {uuid.uuid4().hex}"}], **extra},
        key=key,
    )


def _series(samples: Sequence[Sample], name: str, **labels: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.name == name and all(sample.labels.get(label) == value for label, value in labels.items())
    )


def _until(rig: _Rig, name: str, **labels: str) -> tuple[Sample, ...]:
    return eventually(lambda: _series(scrape(rig.gateway), name, **labels), bool, seconds=30)


def test_a_deployment_name_labels_both_success_and_provider_failure(rig: _Rig) -> None:
    total: Final = "litellm_proxy_total_requests_metric_total"
    success: Final = _ask(rig, _DEPLOYMENT)
    assert success.status_code == 200, success.text
    assert [sample.value for sample in _until(rig, total, requested_model=_DEPLOYMENT, status_code="200")] == [1.0]

    failure: Final = rig.gateway.request(
        "POST",
        "/chat/completions",
        {"model": _DEPLOYMENT, "messages": [{"role": "user", "content": _DEPLOYMENT_FAILURE_PROMPT}]},
    )
    assert failure.status_code == 503, failure.text
    assert [(request.target, json.loads(request.body)["model"]) for request in rig.upstream.drain()] == [
        ("/v1/chat/completions", _DEPLOYMENT.removeprefix("openai/"))
    ] * 2

    failed: Final = _until(rig, "litellm_proxy_failed_requests_metric_total", exception_status="503")
    totals: Final = _until(rig, total, status_code="503")
    assert [(sample.labels["requested_model"], sample.value) for sample in failed] == [(_DEPLOYMENT, 1.0)]
    assert [(sample.labels["requested_model"], sample.value) for sample in totals] == [(_DEPLOYMENT, 1.0)]


def test_unmatched_models_share_one_failure_series_without_reaching_a_provider(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        for prefix in ("unmatched-one", "unmatched-two"):
            response: Final = _ask(rig, f"{prefix}-{uuid.uuid4().hex}", key)
            assert response.status_code == 400, response.text
        assert rig.upstream.drain() == ()
        hashed_key: Final = sha256(key.encode()).hexdigest()
        for metric, status_label in ((PROXY_FAILURES, "exception_status"), (PROXY_REQUESTS, "status_code")):
            samples: Final = _until(rig, metric, hashed_api_key=hashed_key, **{status_label: "400"})
            assert [(sample.labels["requested_model"], sample.value) for sample in samples] == [(OVERFLOW, 2.0)]


def test_multipart_image_edit_failure_keeps_the_raw_deployment_label(rig: _Rig) -> None:
    response: Final = rig.gateway.request_multipart(
        "/images/edits",
        {"model": _IMAGE_DEPLOYMENT, "prompt": _IMAGE_PROMPT},
        {"image": ("input.png", _PNG, "image/png")},
    )
    assert response.status_code == 429, response.text
    (forwarded,) = rig.upstream.drain()
    assert (forwarded.method, forwarded.target) == ("POST", "/v1/images/edits")
    assert forwarded.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert b"gpt-image-1" in forwarded.body and _IMAGE_PROMPT.encode() in forwarded.body and _PNG in forwarded.body

    for metric, status_label in ((PROXY_FAILURES, "exception_status"), (PROXY_REQUESTS, "status_code")):
        samples: Final = _until(rig, metric, route="/images/edits", **{status_label: "429"})
        assert [(sample.labels["requested_model"], sample.value) for sample in samples] == [(_IMAGE_DEPLOYMENT, 1.0)]


def test_billed_image_failure_exports_cost_and_tokens_under_the_routed_group(rig: _Rig) -> None:
    response: Final = rig.gateway.request(
        "POST", "/v1/images/generations", {"model": _BILLED_IMAGE_GROUP, "prompt": _BILLED_IMAGE_PROMPT}
    )
    assert response.status_code == 422, response.text
    (forwarded,) = rig.upstream.drain()
    assert (forwarded.method, forwarded.target) == ("POST", _BILLED_IMAGE_PATH)
    assert json.loads(forwarded.body)["contents"][0]["parts"][0]["text"] == _BILLED_IMAGE_PROMPT

    for metric, expected in (
        ("litellm_spend_metric_total", 12 * _BILLED_INPUT_RATE + 5 * _BILLED_OUTPUT_RATE),
        ("litellm_total_tokens_metric_total", 17.0),
        ("litellm_input_tokens_metric_total", 12.0),
        ("litellm_output_tokens_metric_total", 5.0),
    ):
        samples: Final = _until(rig, metric, model=_BILLED_IMAGE)
        assert [(sample.labels["requested_model"], sample.value) for sample in samples] == [
            (_BILLED_IMAGE_GROUP, pytest.approx(expected))
        ]


def test_a_rate_limited_call_counts_as_a_failed_and_a_429_total_request(rig: _Rig) -> None:
    response: Final = _ask(rig, _FAILING)
    assert response.status_code == 429, response.text
    assert len(rig.upstream.drain()) == 1
    failed: Final = _until(
        rig,
        "litellm_proxy_failed_requests_metric_total",
        api_key_alias="None",
        exception_class="Openai.RateLimitError",
        exception_status="429",
        hashed_api_key=LITELLM_PROXY_MASTER_KEY_ALIAS,
        requested_model=_FAILING,
        route="/chat/completions",
    )
    assert [sample.value for sample in failed] == [1.0]
    totals: Final = _until(
        rig,
        "litellm_proxy_total_requests_metric_total",
        hashed_api_key=LITELLM_PROXY_MASTER_KEY_ALIAS,
        requested_model=_FAILING,
        status_code="429",
    )
    assert [sample.value for sample in totals] == [1.0]


def test_a_good_call_exports_latency_histograms_on_the_shared_buckets_without_the_end_user(rig: _Rig) -> None:
    response: Final = _ask(rig, _LATENCY, user=_END_USER, tags=["teamB"])
    assert response.status_code == 200, response.text
    assert len(rig.upstream.drain()) == 1
    master: Final = {
        "api_key_alias": "None",
        "hashed_api_key": LITELLM_PROXY_MASTER_KEY_ALIAS,
        "requested_model": _LATENCY,
    }
    _until(rig, "litellm_request_total_latency_metric_bucket", le="0.005", **master)
    _until(rig, "litellm_llm_api_latency_metric_bucket", le="0.005", **master)
    for name in ("litellm_request_total_latency_metric_count", "litellm_llm_api_latency_metric_count"):
        assert [sample.value for sample in _until(rig, name, **master)] == [1.0], name
    samples: Final = scrape(rig.gateway)
    assert _END_USER not in label_values(samples)
    expected: Final = {str(bucket).replace("inf", "+Inf") for bucket in LATENCY_BUCKETS}
    for name in (
        "litellm_request_total_latency_metric_bucket",
        "litellm_llm_api_latency_metric_bucket",
        "litellm_overhead_latency_metric_bucket",
    ):
        assert {sample.labels["le"] for sample in _series(samples, name)} == expected, name


def test_client_side_fallbacks_count_one_success_and_one_failure(rig: _Rig) -> None:
    recovered: Final = _ask(rig, _LIMITED, fallbacks=[_GOOD])
    assert recovered.status_code == 200, recovered.text
    missing: Final = f"unknown-model-{uuid.uuid4().hex[:8]}"
    failed: Final = _ask(rig, _LIMITED, fallbacks=[missing])
    assert failed.status_code >= 400, failed.text
    rig.upstream.drain()
    shared: Final = {
        "api_key_alias": "None",
        "exception_class": "Openai.RateLimitError",
        "exception_status": "429",
        "hashed_api_key": LITELLM_PROXY_MASTER_KEY_ALIAS,
        "requested_model": _LIMITED,
    }
    succeeded: Final = _until(rig, "litellm_deployment_successful_fallbacks_total", fallback_model=_GOOD, **shared)
    assert [sample.value for sample in succeeded] == [1.0]
    lost: Final = _until(rig, "litellm_deployment_failed_fallbacks_total", fallback_model=missing, **shared)
    assert [sample.value for sample in lost] == [1.0]


@dataclass(frozen=True, slots=True)
class _Budget:
    remaining: float
    total: float
    hours: float


def _budget(samples: Sequence[Sample], scope: str, label: str, identity: str) -> _Budget | None:
    remaining: Final = _series(samples, f"litellm_remaining_{scope}_budget_metric", **{label: identity})
    total: Final = _series(samples, f"litellm_{scope}_max_budget_metric", **{label: identity})
    hours: Final = _series(samples, f"litellm_{scope}_budget_remaining_hours_metric", **{label: identity})
    if len(remaining) != 1 or len(total) != 1 or len(hours) != 1:
        return None
    return _Budget(remaining[0].value, total[0].value, hours[0].value)


def _reconciled(rig: _Rig, scope: str, label: str, identity: str, info: str, field: str, lookup: str) -> _Budget:
    def read() -> tuple[_Budget | None, float]:
        record: Final = object_value(rig.gateway.get(info, {field: lookup})[_INFO_FIELDS[info]])
        return _budget(scrape(rig.gateway), scope, label, identity), float(str(record["max_budget"])) - float(
            str(record["spend"])
        )

    budget, remaining = eventually(
        read,
        lambda state: (
            state[0] is not None and state[0].remaining < 10.0 and abs(state[1] - state[0].remaining) <= 0.001
        ),
        seconds=60,
    )
    assert budget is not None
    assert abs(remaining - budget.remaining) <= 0.001
    return budget


_INFO_FIELDS: Final = {"/team/info": "team_info", "/key/info": "info", "/user/info": "user_info"}


def test_a_team_call_exports_remaining_max_and_hours_gauges_matching_team_info(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(max_budget=10, budget_duration="7d")
        key: Final = scenario.key(team_id=team)
        assert _ask(rig, _GOOD, key).status_code == 200
        assert len(rig.upstream.drain()) == 1
        budget: Final = _reconciled(rig, "team", "team", team, "/team/info", "team_id", team)
        assert budget.total == 10.0
        assert 0 < budget.hours <= 168


def test_a_key_call_exports_remaining_max_and_hours_gauges_matching_key_info(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(max_budget=10, budget_duration="7d")
        assert _ask(rig, _GOOD, key).status_code == 200
        assert len(rig.upstream.drain()) == 1
        hashed: Final = sha256(key.encode()).hexdigest()
        budget: Final = _reconciled(rig, "api_key", "hashed_api_key", hashed, "/key/info", "key", key)
        assert budget.total == 10.0
        assert 0 <= budget.hours <= 168


def test_a_user_call_exports_remaining_max_and_hours_gauges_matching_user_info(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        user: Final = f"prometheus-user-{uuid.uuid4().hex}"
        scenario.user(user_id=user, max_budget=10, budget_duration="7d")
        key: Final = scenario.key(user_id=user)
        assert _ask(rig, _GOOD, key).status_code == 200
        assert len(rig.upstream.drain()) == 1
        budget: Final = _reconciled(rig, "user", "user", user, "/user/info", "user_id", user)
        assert budget.total == 10.0
        assert 0 <= budget.hours <= 168


def test_a_user_email_labels_the_spend_and_failed_request_series_of_its_keys(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        email: Final = f"prometheus-{uuid.uuid4().hex}@example.com"
        user: Final = f"prometheus-email-{uuid.uuid4().hex}"
        scenario.user(user_id=user, user_email=email)
        key: Final = scenario.key(user_id=user)
        assert _ask(rig, _GOOD, key).status_code == 200
        assert len(rig.upstream.drain()) == 1
        spend: Final = _until(rig, "litellm_spend_metric_total", user_email=email)
        assert [(sample.labels["user"], sample.value) for sample in spend] == [(user, pytest.approx(0.005))]
        assert email in label_values(scrape(rig.gateway))
        assert _ask(rig, _FAILING, key).status_code == 429
        assert len(rig.upstream.drain()) == 1
        failed: Final = _until(
            rig, "litellm_proxy_failed_requests_metric_total", user_email=email, requested_model=_FAILING
        )
        assert [(sample.labels["user"], sample.value) for sample in failed] == [(user, 1.0)]
