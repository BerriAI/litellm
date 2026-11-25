from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, gateway_from_environment, until_asserted
from integration._support.process import owned_proxy
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_UP: Final = "routing-decision-up"
_DOWN: Final = "routing-decision-down"
_ROUTER_UP: Final = "routing-decision-router-up"
_ROUTER_DOWN: Final = "routing-decision-router-down"
_UPSTREAM_MODEL: Final = "gpt-5.4-mini"
_DOWN_UPSTREAM_MODEL: Final = "gpt-5.4-mini-down"
_TIERS: Final = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
_CUSTOM_LABELS: Final = (
    "routing_decision.tier",
    "routing_decision.cause",
    "routing_decision.routed_model",
    "routing_decision.classifier_crux",
    "nested.caller",
)
_CALLER_METADATA: Final = {
    "nested": {"caller": "from-header"},
    "routing_decision": {"tier": "CALLER", "cause": "caller", "routed_model": "caller", "classifier_crux": "caller"},
}
_UNROUTED: Final = ("None", "None", "None", "None", "from-header")
_ROUTED_UP: Final = ("SIMPLE", "heuristic_scorer", _UP, "None", "from-header")
_ROUTED_DOWN: Final = ("SIMPLE", "heuristic_scorer", _DOWN, "None", "from-header")
_SUCCESS_METRICS: Final = (
    "litellm_proxy_total_requests_metric_total",
    "litellm_request_total_latency_metric_count",
    "litellm_total_tokens_metric_total",
    "litellm_deployment_success_responses_total",
)
_FAILURE_METRICS: Final = (
    "litellm_proxy_failed_requests_metric_total",
    "litellm_proxy_total_requests_metric_total",
    "litellm_llm_api_failed_requests_metric_total",
    "litellm_deployment_failure_responses_total",
)
_ROUTES: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
_PASS_THROUGH: Final = "/routing-decision-pass-through"
_PROXY_TOTAL: Final = ("litellm_proxy_total_requests_metric_total",)
_REFUSAL_METRICS: Final = ("litellm_proxy_failed_requests_metric_total", "litellm_proxy_total_requests_metric_total")


def _respond_with(vendor: rv.ResponsesVendor, request: Request) -> Reply:
    if request.target == _PASS_THROUGH:
        return Reply()
    if request.method == "POST" and json.loads(request.body).get("model") == _DOWN_UPSTREAM_MODEL:
        return rv.error(503, "synthetic upstream outage", "server_error")
    return vendor.respond(request)


def _deployment(name: str, upstream_model: str, upstream: Wire) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"openai/{upstream_model}",
            "api_key": "sk-fixture",
            "api_base": f"{upstream.url}/v1",
        },
    }


def _complexity_router(name: str, target: str) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": {tier: target for tier in _TIERS}},
        },
    }


def _config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        _deployment(_UP, _UPSTREAM_MODEL, upstream),
        _deployment(_DOWN, _DOWN_UPSTREAM_MODEL, upstream),
        _complexity_router(_ROUTER_UP, _UP),
        _complexity_router(_ROUTER_DOWN, _DOWN),
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration["litellm_settings"]["custom_prometheus_metadata_labels"] = [*_CUSTOM_LABELS, "user_api_key_hash"]
    configuration["general_settings"]["pass_through_endpoints"] = [
        {"path": _PASS_THROUGH, "target": f"{upstream.url}{_PASS_THROUGH}", "auth": True}
    ]
    configuration.setdefault("router_settings", {}).update({"num_retries": 0, "disable_cooldowns": True})
    path: Final = directory / "prometheus-routing-decision-labels.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("prometheus-routing-decision-labels")
    samples: Final = directory / "prometheus-multiproc"
    samples.mkdir()
    vendor: Final = rv.ResponsesVendor()
    with (
        wire_server(lambda request: _respond_with(vendor, request)) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"PROMETHEUS_MULTIPROC_DIR": str(samples)},
            config=_config(directory, upstream),
            workers=2,
        ) as owned,
    ):
        yield owned


def _body(route: str, model: str, stream: bool) -> dict[str, JsonValue]:
    text: Final = f"hi marker-{uuid.uuid4().hex}"
    if route == "/v1/responses":
        return {"model": model, "input": text, "stream": stream}
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}], "stream": stream}


def _send(gateway: Gateway, route: str, model: str, key: str, stream: bool) -> int:
    with gateway.client.stream(
        "POST",
        route,
        json=_body(route, model, stream),
        headers={
            "Authorization": f"Bearer {key}",
            "x-litellm-spend-logs-metadata": json.dumps(_CALLER_METADATA),
        },
    ) as response:
        response.read()
    return response.status_code


def _label_values(sample: Sample) -> tuple[str | None, ...]:
    return tuple(sample.labels.get(label.replace(".", "_")) for label in _CUSTOM_LABELS)


def _labels_by_metric(
    samples: Sequence[Sample], metrics: Sequence[str], hashed_key: str
) -> dict[str, list[tuple[str | None, ...]]]:
    return {
        metric: [
            _label_values(sample)
            for sample in samples
            if sample.name == metric and sample.labels.get("hashed_api_key") == hashed_key
        ]
        for metric in metrics
    }


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    ("model", "metrics", "succeeds", "expected"),
    [
        pytest.param(_ROUTER_UP, _SUCCESS_METRICS, True, _ROUTED_UP, id="auto-routed-success"),
        pytest.param(_ROUTER_DOWN, _FAILURE_METRICS, False, _ROUTED_DOWN, id="auto-routed-failure"),
        pytest.param(_UP, _SUCCESS_METRICS, True, _UNROUTED, id="direct-success"),
        pytest.param(_DOWN, _FAILURE_METRICS, False, _UNROUTED, id="direct-failure"),
    ],
)
def test_request_metrics_are_labelled_with_the_routers_decision_and_never_the_callers(
    gateway: Gateway,
    route: str,
    stream: bool,
    model: str,
    metrics: tuple[str, ...],
    succeeds: bool,
    expected: tuple[str, ...],
) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        status: Final = _send(gateway, route, model, key, stream)
        assert (status == 200) is succeeds, status

        def landed() -> None:
            found: Final = _labels_by_metric(scrape(gateway), metrics, sha256(key.encode()).hexdigest())
            assert found == {metric: [expected] for metric in metrics}

        until_asserted(landed, seconds=30)


@pytest.mark.parametrize("metadata_key", ("litellm_metadata", "metadata"))
def test_a_pass_through_caller_cannot_set_the_routing_decision_labels(gateway: Gateway, metadata_key: str) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(allowed_passthrough_routes=[_PASS_THROUGH])
        response: Final = gateway.client.post(
            _PASS_THROUGH, json={metadata_key: _CALLER_METADATA}, headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == 200, response.text

        def landed() -> None:
            found: Final = _labels_by_metric(scrape(gateway), _PROXY_TOTAL, sha256(key.encode()).hexdigest())
            assert found == {metric: [("None",) * len(_CUSTOM_LABELS)] for metric in _PROXY_TOTAL}

        until_asserted(landed, seconds=30)


def test_a_call_refused_before_routing_takes_no_label_from_the_caller(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[_UP])
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            json={
                **_body("/v1/chat/completions", _ROUTER_UP, stream=False),
                "metadata": _CALLER_METADATA,
                "standard_logging_object": {"metadata": _CALLER_METADATA},
            },
            headers={
                "Authorization": f"Bearer {key}",
                "x-litellm-spend-logs-metadata": json.dumps(_CALLER_METADATA),
            },
        )
        assert response.status_code == 403, response.text

        def landed() -> None:
            found: Final = _labels_by_metric(scrape(gateway), _REFUSAL_METRICS, sha256(key.encode()).hexdigest())
            assert found == {metric: [("None",) * len(_CUSTOM_LABELS)] for metric in _REFUSAL_METRICS}

        until_asserted(landed, seconds=30)


def test_unknown_keys_share_one_series_even_when_the_key_hash_is_a_custom_label(gateway: Gateway) -> None:
    statuses: Final = tuple(
        gateway.client.post(
            "/v1/chat/completions",
            json=_body("/v1/chat/completions", _ROUTER_UP, stream=False),
            headers={"Authorization": f"Bearer sk-{uuid.uuid4().hex}"},
        ).status_code
        for _ in range(2)
    )
    assert statuses == (401, 401)

    def landed() -> None:
        refused: Final = tuple(
            sample
            for sample in scrape(gateway)
            if sample.name == "litellm_proxy_failed_requests_metric_total" and sample.labels["exception_status"] == "401"
        )
        assert [(sample.labels["user_api_key_hash"], sample.value) for sample in refused] == [("None", 2.0)]

    until_asserted(landed, seconds=30)
