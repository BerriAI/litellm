from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
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
_UNREACHABLE: Final = "unreachable"
_STALLED: Final = "stalled"
_NEVER_RELEASED: Final = threading.Event()
_IN_HOUSE_HANDLER: Final = """
from litellm import CustomLLM


class InHouse(CustomLLM):
    pass


handler = InHouse()
"""


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
_STREAMED_CHAT: Final = _deployment("deployment-failure-streamed-chat", "openai/refused-streamed-chat", "openai")
_STREAMED_RESPONSES: Final = _deployment(
    "deployment-failure-streamed-responses", "openai/refused-streamed-responses", "openai"
)
_STREAMED_MESSAGES: Final = _deployment(
    "deployment-failure-streamed-messages", "anthropic/claude-haiku-4-5", "anthropic"
)
_UNREACHABLE_EMBEDDING: Final = _deployment(
    "deployment-failure-unreachable", f"openai/{_UNREACHABLE}-embedding", "openai"
)
_STALLED_EMBEDDING: Final = _deployment(
    "deployment-failure-stalled", f"openai/{_STALLED}-embedding", "openai", timeout=1
)
_JSON_PROVIDER: Final = _deployment("deployment-failure-json-provider", "abliteration/blocked", "abliteration", rpm=0)
_CUSTOM_PROVIDER: Final = _deployment("deployment-failure-custom-provider", "in-house/blocked", "in-house", rpm=0)
_PROMPT_MANAGEMENT: Final = _deployment(
    "deployment-failure-prompt-management", "litellm_agent/anthropic/claude-haiku-4-5", "anthropic"
)
_UNPREFIXED: Final = _deployment("deployment-failure-unprefixed", "gpt-4o-mini", "", rpm=0)
_FORGED_EMBEDDING: Final = _deployment("deployment-failure-forged-embedding", "openai/refused-forged", "openai")
_FORGED_BLOCKED: Final = _deployment("deployment-failure-forged-blocked", "openai/blocked-forged", "openai", rpm=0)
_UNAUTHENTICATED: Final = _deployment("deployment-failure-unauthenticated", "openai/refused-unauthenticated", "openai")
_SDK: Final = {
    client: _deployment(f"deployment-failure-{client}", model, provider)
    for client, model, provider in (
        ("openai-sync", "openai/refused-sdk-sync", "openai"),
        ("openai-async", "openai/refused-sdk-async", "openai"),
        ("anthropic-sync", "anthropic/claude-haiku-4-5", "anthropic"),
        ("anthropic-async", "anthropic/claude-haiku-4-5", "anthropic"),
    )
}
_BURST_EMBEDDING: Final = _deployment("burst-embedding", "openai/refused-burst-embedding", "openai")
_BURST_RESPONSES: Final = _deployment("burst-responses", "openai/refused-burst-responses", "openai")
_BURST_MESSAGES: Final = _deployment("burst-messages", "anthropic/claude-haiku-4-5", "anthropic")
_BURST_UNSUPPORTED: Final = _deployment("burst-unsupported", "anthropic/claude-haiku-4-5", "anthropic")
_BURST_BLOCKED: Final = _deployment("burst-blocked", "gemini/blocked-burst", "gemini", rpm=0)
_BURST: Final = (_BURST_EMBEDDING, _BURST_RESPONSES, _BURST_MESSAGES, _BURST_UNSUPPORTED, _BURST_BLOCKED)
_DEPLOYMENTS: Final = (
    _CHAT,
    _EMBEDDING,
    _RESPONSES,
    _MESSAGES,
    _UNSUPPORTED,
    _LIMITED,
    _LIMITED_AZURE,
    _STREAMED_CHAT,
    _STREAMED_RESPONSES,
    _STREAMED_MESSAGES,
    _UNREACHABLE_EMBEDDING,
    _STALLED_EMBEDDING,
    _JSON_PROVIDER,
    _CUSTOM_PROVIDER,
    _PROMPT_MANAGEMENT,
    _UNPREFIXED,
    _FORGED_EMBEDDING,
    _FORGED_BLOCKED,
    _UNAUTHENTICATED,
    *_SDK.values(),
    *_BURST,
)


def _respond(request: Request) -> Reply:
    model: Final = json.loads(request.body).get("model", "") if request.body else ""
    if model.startswith(_UNREACHABLE):
        return Reply(drop_connection=True)
    if model.startswith(_STALLED):
        return Reply(chunks=(b"{}",), gate_before_first=_NEVER_RELEASED)
    if _SERVED in request.target or model.startswith(_SERVED):
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
    (directory / "in_house_handler.py").write_text(_IN_HOUSE_HANDLER)
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration["litellm_settings"]["custom_provider_map"] = [
        {"provider": "in-house", "custom_handler": "in_house_handler.handler"}
    ]
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


def _counted(samples: Sequence[Sample], name: str, model_id: str) -> Mapping[str, float]:
    counts: Final[Counter[str]] = Counter()
    for sample in _series(samples, name, model_id):
        counts[sample.labels["api_provider"]] += sample.value
    return counts


def _counts(rig: _Rig, name: str, deployment: _Deployment, expected: int) -> Mapping[str, float]:
    return eventually(
        lambda: _counted(scrape(rig.gateway), name, deployment.model_id),
        lambda counts: sum(counts.values()) >= expected,
        seconds=60,
    )


def _post(rig: _Rig, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return rig.gateway.request("POST", path, body)


def _prompt() -> str:
    return f"deployment failure {uuid.uuid4().hex}"


def _chat(deployment: _Deployment, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": deployment.group, "messages": [{"role": "user", "content": _prompt()}], **extra}


def _messages(deployment: _Deployment, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": deployment.group, "max_tokens": 16, "messages": [{"role": "user", "content": _prompt()}], **extra}


def test_a_refused_chat_completion_labels_the_deployment_failure_with_its_provider(rig: _Rig) -> None:
    response: Final = _post(rig, "/v1/chat/completions", _chat(_CHAT))
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
    response: Final = _post(rig, "/v1/messages", _messages(_MESSAGES))
    assert response.status_code == 400, response.text
    assert any(request.target.endswith("/v1/messages") for request in rig.upstream.drain())
    _assert_labeled(rig, _MESSAGES)


def test_a_param_the_deployment_does_not_support_labels_the_deployment_failure_with_its_provider(rig: _Rig) -> None:
    rig.upstream.drain()
    response: Final = _post(rig, "/v1/chat/completions", _chat(_UNSUPPORTED, logit_bias={"50256": -100}))
    assert response.status_code == 400, response.text
    assert "logit_bias" in response.text
    assert rig.upstream.drain() == ()
    _assert_labeled(rig, _UNSUPPORTED)


def _statuses(rig: _Rig, deployment: _Deployment) -> Iterator[int]:
    for _ in range(8):
        yield _post(rig, "/v1/chat/completions", _chat(deployment)).status_code


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


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", _chat(_STREAMED_CHAT, stream=True)),
        ("/v1/responses", {"model": _STREAMED_RESPONSES.group, "input": _prompt(), "stream": True}),
        ("/v1/messages", _messages(_STREAMED_MESSAGES, stream=True)),
    ],
    ids=["chat", "responses", "messages"],
)
def test_a_refused_streaming_call_labels_the_deployment_failure_with_its_provider(
    rig: _Rig, path: str, body: Mapping[str, JsonValue]
) -> None:
    deployment: Final = next(deployment for deployment in _DEPLOYMENTS if deployment.group == body["model"])
    response: Final = _post(rig, path, body)
    assert response.status_code == 400, response.text
    _assert_labeled(rig, deployment)


@pytest.mark.parametrize(
    ("deployment", "status"),
    [(_UNREACHABLE_EMBEDDING, 500), (_STALLED_EMBEDDING, 408)],
    ids=["connection-dropped", "read-timeout"],
)
def test_an_upstream_that_never_answers_labels_the_deployment_failure_with_its_provider(
    rig: _Rig, deployment: _Deployment, status: int
) -> None:
    response: Final = _post(rig, "/v1/embeddings", {"model": deployment.group, "input": _prompt()})
    assert response.status_code == status, response.text
    _assert_labeled(rig, deployment)


@pytest.mark.parametrize(
    "deployment",
    [_JSON_PROVIDER, _CUSTOM_PROVIDER, _UNPREFIXED],
    ids=["json-configured-provider", "custom-provider-map", "unprefixed-model"],
)
def test_a_call_blocked_before_it_leaves_labels_the_failure_by_the_deployment_model_prefix(
    rig: _Rig, deployment: _Deployment
) -> None:
    rig.upstream.drain()
    response: Final = _post(rig, "/v1/chat/completions", _chat(deployment))
    assert response.status_code == 429, response.text
    assert rig.upstream.drain() == ()
    _assert_labeled(rig, deployment)


def test_a_prompt_management_deployment_labels_the_failure_with_the_provider_it_resolved_to(rig: _Rig) -> None:
    response: Final = _post(rig, "/v1/chat/completions", _chat(_PROMPT_MANAGEMENT, logit_bias={"50256": -100}))
    assert response.status_code == 400, response.text
    found: Final = eventually(
        lambda: tuple(
            sample
            for sample in scrape(rig.gateway)
            if sample.name == _FAILURES and sample.labels.get("model_group") == _PROMPT_MANAGEMENT.group
        ),
        bool,
        seconds=30,
    )
    assert {sample.labels["api_provider"] for sample in found} == {_PROMPT_MANAGEMENT.provider}


@pytest.mark.parametrize(
    "forged",
    ["bedrock/forged", 5, ["bedrock/forged"], "", f"bedrock/{'x' * 5000}"],
    ids=["provider-string", "int", "list", "empty", "five-kb"],
)
@pytest.mark.parametrize(
    ("path", "deployment", "status"),
    [("/v1/embeddings", _FORGED_EMBEDDING, 400), ("/v1/chat/completions", _FORGED_BLOCKED, 429)],
    ids=["refused-upstream", "blocked-before-the-call"],
)
def test_a_caller_cannot_pick_the_label_through_request_metadata(
    rig: _Rig, path: str, deployment: _Deployment, status: int, forged: JsonValue
) -> None:
    bucket: Final[dict[str, JsonValue]] = {"deployment": forged, "model_info": {"id": deployment.model_id}}
    body: Final = (
        _chat(deployment, metadata=bucket, litellm_metadata=bucket)
        if path == "/v1/chat/completions"
        else {"model": deployment.group, "input": _prompt(), "metadata": bucket, "litellm_metadata": bucket}
    )
    response: Final = _post(rig, path, body)
    assert response.status_code == status, response.text
    _assert_labeled(rig, deployment)
    every_provider: Final = {
        sample.labels.get("api_provider") for sample in scrape(rig.gateway) if sample.name in (_FAILURES, _TOTALS)
    }
    assert not any(str(provider).startswith("bedrock") for provider in every_provider), every_provider


def test_an_unauthenticated_request_leaves_the_deployment_series_alone(rig: _Rig) -> None:
    body: Final[dict[str, JsonValue]] = {"model": _UNAUTHENTICATED.group, "input": _prompt()}
    rejected: Final = rig.gateway.request("POST", "/v1/embeddings", body, key="sk-not-a-litellm-key")
    assert rejected.status_code == 401, rejected.text
    refused: Final = _post(rig, "/v1/embeddings", body)
    assert refused.status_code == 400, refused.text
    assert _counts(rig, _FAILURES, _UNAUTHENTICATED, 1) == {"openai": 1}
    assert _counts(rig, _TOTALS, _UNAUTHENTICATED, 1) == {"openai": 1}


def _sdk_v1(gateway: Gateway) -> str:
    return f"{str(gateway.client.base_url).rstrip('/')}/v1"


def _sdk_origin(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _openai_sync(gateway: Gateway, deployment: _Deployment) -> int:
    with (
        httpx.Client(timeout=15, trust_env=False) as transport,
        openai.OpenAI(api_key=gateway.key, base_url=_sdk_v1(gateway), max_retries=0, http_client=transport) as client,
    ):
        with pytest.raises(openai.APIStatusError) as refusal:
            client.embeddings.create(model=deployment.group, input=_prompt())
        return refusal.value.status_code


def _openai_async(gateway: Gateway, deployment: _Deployment) -> int:
    async def run() -> int:
        async with (
            httpx.AsyncClient(timeout=15, trust_env=False) as transport,
            openai.AsyncOpenAI(
                api_key=gateway.key, base_url=_sdk_v1(gateway), max_retries=0, http_client=transport
            ) as client,
        ):
            with pytest.raises(openai.APIStatusError) as refusal:
                await client.embeddings.create(model=deployment.group, input=_prompt())
            return refusal.value.status_code

    return asyncio.run(run())


def _anthropic_sync(gateway: Gateway, deployment: _Deployment) -> int:
    with (
        httpx.Client(timeout=15, trust_env=False) as transport,
        anthropic.Anthropic(
            api_key=gateway.key, base_url=_sdk_origin(gateway), max_retries=0, http_client=transport
        ) as client,
    ):
        with pytest.raises(anthropic.APIStatusError) as refusal:
            client.messages.create(
                model=deployment.group, max_tokens=16, messages=[{"role": "user", "content": _prompt()}]
            )
        return refusal.value.status_code


def _anthropic_async(gateway: Gateway, deployment: _Deployment) -> int:
    async def run() -> int:
        async with (
            httpx.AsyncClient(timeout=15, trust_env=False) as transport,
            anthropic.AsyncAnthropic(
                api_key=gateway.key, base_url=_sdk_origin(gateway), max_retries=0, http_client=transport
            ) as client,
        ):
            with pytest.raises(anthropic.APIStatusError) as refusal:
                await client.messages.create(
                    model=deployment.group, max_tokens=16, messages=[{"role": "user", "content": _prompt()}]
                )
            return refusal.value.status_code

    return asyncio.run(run())


_SDK_CALLS: Final[Mapping[str, Callable[[Gateway, _Deployment], int]]] = {
    "openai-sync": _openai_sync,
    "openai-async": _openai_async,
    "anthropic-sync": _anthropic_sync,
    "anthropic-async": _anthropic_async,
}


@pytest.mark.parametrize("client", list(_SDK_CALLS))
def test_a_refusal_through_the_provider_sdk_labels_the_deployment_failure_with_its_provider(
    rig: _Rig, client: str
) -> None:
    assert _SDK_CALLS[client](rig.gateway, _SDK[client]) == 400
    _assert_labeled(rig, _SDK[client])


_BURST_SIZE: Final = 10
_BURST_CALLS: Final[Mapping[str, tuple[str, Callable[[], Mapping[str, JsonValue]], int]]] = {
    _BURST_EMBEDDING.model_id: ("/v1/embeddings", lambda: {"model": _BURST_EMBEDDING.group, "input": _prompt()}, 400),
    _BURST_RESPONSES.model_id: ("/v1/responses", lambda: {"model": _BURST_RESPONSES.group, "input": _prompt()}, 400),
    _BURST_MESSAGES.model_id: ("/v1/messages", lambda: _messages(_BURST_MESSAGES), 400),
    _BURST_UNSUPPORTED.model_id: (
        "/v1/chat/completions",
        lambda: _chat(_BURST_UNSUPPORTED, logit_bias={"50256": -100}),
        400,
    ),
    _BURST_BLOCKED.model_id: ("/v1/chat/completions", lambda: _chat(_BURST_BLOCKED), 429),
}


def test_a_concurrent_burst_of_failures_counts_each_one_once_under_its_deployment_provider(rig: _Rig) -> None:
    calls: Final = [deployment.model_id for deployment in _BURST for _ in range(_BURST_SIZE)]
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        statuses: Final = Counter(
            pool.map(
                lambda model_id: (
                    model_id,
                    _post(rig, _BURST_CALLS[model_id][0], _BURST_CALLS[model_id][1]()).status_code,
                ),
                calls,
            )
        )
    assert statuses == Counter({(model_id, status): _BURST_SIZE for model_id, (_, _, status) in _BURST_CALLS.items()})
    for deployment in _BURST:
        assert _counts(rig, _FAILURES, deployment, _BURST_SIZE) == {deployment.provider: _BURST_SIZE}
        assert _counts(rig, _TOTALS, deployment, _BURST_SIZE) == {deployment.provider: _BURST_SIZE}
