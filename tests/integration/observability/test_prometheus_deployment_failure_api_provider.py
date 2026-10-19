from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.prometheus_series import Sample, scrape
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_FAILURES: Final = "litellm_deployment_failure_responses_total"
_TOTALS: Final = "litellm_deployment_total_requests_total"
_SERVED: Final = "limited"


@dataclass(frozen=True, slots=True)
class _Deployment:
    group: str
    model: str
    provider: str
    extra: Mapping[str, str | int]
    model_id: str = ""


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    upstream: Wire


def _deployment(group: str, model: str, provider: str, **extra: str | int) -> _Deployment:
    return _Deployment(group, model, provider, extra, f"{group}-{uuid.uuid4().hex[:8]}")


_CHAT: Final = _deployment("deployment-failure-chat", "openai/refused-chat", "openai")
_EMBEDDING: Final = _deployment("deployment-failure-embedding", "openai/refused-embedding", "openai")
_RESPONSES: Final = _deployment("deployment-failure-responses", "openai/refused-responses", "openai")
_MESSAGES: Final = _deployment("deployment-failure-messages", "anthropic/claude-haiku-4-5", "anthropic")
_UNSUPPORTED: Final = _deployment("deployment-failure-unsupported", "anthropic/claude-haiku-4-5", "anthropic")
_LIMITED: Final = _deployment("deployment-failure-limited", f"openai/{_SERVED}-openai", "openai", rpm=1)
_LIMITED_AZURE: Final = _deployment("gpt-4o-mini", f"azure/{_SERVED}-azure", "azure", rpm=1, api_version="2024-10-21")
_DEPLOYMENTS: Final = (_CHAT, _EMBEDDING, _RESPONSES, _MESSAGES, _UNSUPPORTED, _LIMITED, _LIMITED_AZURE)


def _respond(request: Request) -> Reply:
    if _SERVED in request.target or json.loads(request.body).get("model", "").startswith(_SERVED):
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "served"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ).encode()
        )
    if request.target.endswith("/v1/messages"):
        return Reply(
            status=400,
            body=json.dumps(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "refused upstream"}}
            ).encode(),
        )
    return Reply(
        status=400,
        body=json.dumps(
            {"error": {"message": "refused upstream", "type": "invalid_request_error", "code": "refused"}}
        ).encode(),
    )


def _config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        *(
            {
                "model_name": deployment.group,
                "litellm_params": {
                    "model": deployment.model,
                    "api_key": "sk-fixture",
                    "api_base": upstream.url if deployment.provider != "openai" else f"{upstream.url}/v1",
                    **deployment.extra,
                },
                "model_info": {"id": deployment.model_id},
            }
            for deployment in _DEPLOYMENTS
        ),
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    router_settings: Final = configuration.setdefault("router_settings", {})
    router_settings["num_retries"] = 0
    router_settings["optional_pre_call_checks"] = ["enforce_model_rate_limits"]
    path: Final = directory / "prometheus-deployment-failure-api-provider.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-deployment-failure-api-provider")
    samples: Final = directory / "prometheus-multiproc"
    samples.mkdir()
    with (
        wire_server(_respond) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"PROMETHEUS_MULTIPROC_DIR": str(samples)},
            config=_config(directory, upstream),
            workers=2,
        ) as owned,
    ):
        yield _Rig(owned, upstream)


def _series(samples: Sequence[Sample], name: str, model_id: str) -> tuple[Sample, ...]:
    return tuple(sample for sample in samples if sample.name == name and sample.labels.get("model_id") == model_id)


def _providers(rig: _Rig, name: str, model_id: str) -> frozenset[str]:
    found: Final = eventually(lambda: _series(scrape(rig.gateway), name, model_id), bool, seconds=30)
    return frozenset(sample.labels["api_provider"] for sample in found)


def _assert_labeled(rig: _Rig, deployment: _Deployment) -> None:
    assert _providers(rig, _FAILURES, deployment.model_id) == {deployment.provider}
    assert _providers(rig, _TOTALS, deployment.model_id) == {deployment.provider}


def _post(rig: _Rig, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return rig.gateway.request("POST", path, body)


def _prompt() -> str:
    return f"deployment failure {uuid.uuid4().hex}"


def test_a_refused_chat_completion_labels_the_deployment_failure_with_its_provider(rig: _Rig) -> None:
    response: Final = _post(
        rig, "/v1/chat/completions", {"model": _CHAT.group, "messages": [{"role": "user", "content": _prompt()}]}
    )
    assert response.status_code == 400, response.text
    _assert_labeled(rig, _CHAT)


def test_a_refused_embedding_labels_the_deployment_failure_with_its_provider(rig: _Rig) -> None:
    response: Final = _post(rig, "/v1/embeddings", {"model": _EMBEDDING.group, "input": _prompt()})
    assert response.status_code == 400, response.text
    assert any(request.target.endswith("/v1/embeddings") for request in rig.upstream.drain())
    _assert_labeled(rig, _EMBEDDING)


def test_a_refused_responses_call_labels_the_deployment_failure_with_its_provider(rig: _Rig) -> None:
    response: Final = _post(rig, "/v1/responses", {"model": _RESPONSES.group, "input": _prompt()})
    assert response.status_code == 400, response.text
    assert any(request.target.endswith("/v1/responses") for request in rig.upstream.drain())
    _assert_labeled(rig, _RESPONSES)


def test_a_refused_anthropic_messages_call_labels_the_deployment_failure_with_its_provider(rig: _Rig) -> None:
    response: Final = _post(
        rig,
        "/v1/messages",
        {"model": _MESSAGES.group, "max_tokens": 16, "messages": [{"role": "user", "content": _prompt()}]},
    )
    assert response.status_code == 400, response.text
    assert any(request.target.endswith("/v1/messages") for request in rig.upstream.drain())
    _assert_labeled(rig, _MESSAGES)


def test_a_param_the_deployment_does_not_support_labels_the_deployment_failure_with_its_provider(rig: _Rig) -> None:
    rig.upstream.drain()
    response: Final = _post(
        rig,
        "/v1/chat/completions",
        {
            "model": _UNSUPPORTED.group,
            "messages": [{"role": "user", "content": _prompt()}],
            "logit_bias": {"50256": -100},
        },
    )
    assert response.status_code == 400, response.text
    assert "logit_bias" in response.text
    assert rig.upstream.drain() == ()
    _assert_labeled(rig, _UNSUPPORTED)


def _statuses(rig: _Rig, deployment: _Deployment) -> Iterator[int]:
    for _ in range(8):
        yield _post(
            rig,
            "/v1/chat/completions",
            {"model": deployment.group, "messages": [{"role": "user", "content": _prompt()}]},
        ).status_code


def _until_rate_limited(rig: _Rig, deployment: _Deployment) -> None:
    first_refusal: Final = next((status for status in _statuses(rig, deployment) if status != 200), None)
    assert first_refusal == 429, f"expected the rpm limit to refuse a request, first non-200 was {first_refusal}"


@pytest.mark.parametrize(
    "deployment", [_LIMITED, _LIMITED_AZURE], ids=["alias-group", "group-named-like-another-provider"]
)
def test_a_deployment_over_its_rpm_limit_labels_the_failure_with_the_deployment_provider(
    rig: _Rig, deployment: _Deployment
) -> None:
    _until_rate_limited(rig, deployment)
    _assert_labeled(rig, deployment)
