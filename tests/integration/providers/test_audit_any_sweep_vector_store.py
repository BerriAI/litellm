from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration.authorization._guardrail_opt_out import upstream_observations
from pydantic import JsonValue

CONFIG_STORE_ID: Final = "vs_integration_config_store"
FAIL_STORE_ID: Final = "vs_audit_fail_store"
PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"


def _marker() -> str:
    return uuid.uuid4().hex


def _chat_body(model: str, marker: str, store_id: str, *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": marker}],
        "vector_store_ids": [store_id],
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }


def _sdk_chat_body(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        **{key: value for key, value in body.items() if key != "vector_store_ids"},
        "extra_body": {"vector_store_ids": body["vector_store_ids"]},
    }


def _search_observations(gateway: Gateway, marker: str, store_id: str) -> tuple[dict[str, JsonValue], ...]:
    path: Final = f"/vector_stores/{store_id}/search"
    return tuple(
        observation
        for observation in upstream_observations(gateway)
        if observation["path"] == path and marker in json.dumps(observation["body"])
    )


def _chat_observations(gateway: Gateway, marker: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        observation
        for observation in upstream_observations(gateway)
        if observation["path"] == "/v1/chat/completions" and marker in json.dumps(observation["body"])
    )


def _message_provider_fields(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    choices: Final = payload["choices"]
    assert isinstance(choices, list) and choices, payload
    message: Final = object_value(object_value(choices[0])["message"])
    fields: Final = message.get("provider_specific_fields")
    assert isinstance(fields, dict), payload
    return object_value(fields)


def _sse_data_lines(text: str) -> tuple[dict[str, JsonValue], ...]:
    parsed: list[dict[str, JsonValue]] = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        value = json.loads(line.removeprefix("data: "))
        if isinstance(value, dict):
            parsed.append(value)
    return tuple(parsed)


def _chunk_annotated(chunk: dict[str, JsonValue]) -> bool:
    choices: Final = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    delta: Final = object_value(choices[0]).get("delta")
    if not isinstance(delta, dict):
        return False
    fields: Final = object_value(delta).get("provider_specific_fields")
    return "search_results" in json.dumps(fields)


def _openai_client(gateway: Gateway, key: str) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=key,
        http_client=httpx.Client(trust_env=False),
    )


def _openai_async_client(gateway: Gateway, key: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=key,
        http_client=httpx.AsyncClient(trust_env=False),
    )


def _register_scenario(gateway: Gateway, scenario_id: str, response: dict[str, JsonValue]) -> None:
    with httpx.Client(timeout=5, trust_env=False) as client:
        result: Final = client.post(
            f"{gateway.upstream_url}/__scenarios",
            json={"scenario_id": scenario_id, "response": response},
        )
    assert result.status_code == 200, result.text


def test_vector_store_hook_chat_completions_raw_httpx(gateway: Gateway) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, CONFIG_STORE_ID), key=key
        )
        assert response.status_code == 200, response.text
        assert _message_provider_fields(response.json())["search_results"]
        assert len(_search_observations(gateway, marker, CONFIG_STORE_ID)) == 1


def test_vector_store_hook_chat_completions_raw_httpx_stream(gateway: Gateway) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, CONFIG_STORE_ID, stream=True), key=key
        )
        assert response.status_code == 200, response.text
        chunks: Final = _sse_data_lines(response.text)
        assert any(_chunk_annotated(chunk) for chunk in chunks), response.text
        assert len(_search_observations(gateway, marker, CONFIG_STORE_ID)) == 1


@pytest.mark.parametrize("stream", (False, True))
def test_vector_store_hook_chat_completions_openai_sdk(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        client: Final = _openai_client(gateway, key)
        body: Final = _chat_body(model, marker, CONFIG_STORE_ID, stream=stream)
        if stream:
            with client.chat.completions.stream(
                **{k: v for k, v in _sdk_chat_body(body).items() if k != "stream"}
            ) as event_stream:
                for _event in event_stream:
                    pass
            snapshot: Final = event_stream.get_final_completion().model_dump(mode="json")
            assert _message_provider_fields(snapshot)["search_results"], snapshot
        else:
            completion: Final = client.chat.completions.create(**_sdk_chat_body(body))
            assert _message_provider_fields(completion.model_dump(mode="json"))["search_results"]
        assert len(_search_observations(gateway, marker, CONFIG_STORE_ID)) == 1


@pytest.mark.parametrize("stream", (False, True))
async def test_vector_store_hook_chat_completions_openai_async_sdk(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        client: Final = _openai_async_client(gateway, key)
        body: Final = _chat_body(model, marker, CONFIG_STORE_ID, stream=stream)
        if stream:
            stream_response: Final = await client.chat.completions.create(**_sdk_chat_body(body))
            chunks: list[dict[str, JsonValue]] = []
            async for chunk in stream_response:
                chunks.append(chunk.model_dump(mode="json"))
            assert any(_chunk_annotated(chunk) for chunk in chunks), chunks
        else:
            completion: Final = await client.chat.completions.create(**_sdk_chat_body(body))
            assert _message_provider_fields(completion.model_dump(mode="json"))["search_results"]
        assert len(_search_observations(gateway, marker, CONFIG_STORE_ID)) == 1


@pytest.mark.parametrize("stream", (False, True))
def test_vector_store_hook_responses_api(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker()
    scenario_id: Final = "auditresp" if not stream else "auditrespstream"
    if stream:
        _register_scenario(
            gateway,
            scenario_id,
            {
                "content_type": "text/event-stream",
                "frames": (
                    'event: response.created\ndata: {"type":"response.created","response":{"id":"$UNIQUE_ID","object":"response","status":"in_progress"}}',
                    'event: response.completed\ndata: {"type":"response.completed","response":{"id":"$UNIQUE_ID","object":"response","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}}',
                ),
            },
        )
    else:
        _register_scenario(
            gateway,
            scenario_id,
            {
                "content_type": "application/x-routed",
                "routes": {
                    "POST /responses": {
                        "content_type": "application/json",
                        "body": {
                            "id": "resp_$UNIQUE_ID",
                            "object": "response",
                            "created_at": 1700000000,
                            "status": "completed",
                            "model": "openai/gpt-4o-mini",
                            "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
                            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                        },
                    }
                },
            },
        )
    with gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{gateway.upstream_url}/{scenario_id}")
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": marker,
                "vector_store_ids": [CONFIG_STORE_ID],
                "stream": stream,
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        if stream:
            assert "response.completed" in response.text
        else:
            payload: Final = response.json()
            assert payload["object"] == "response" and payload["id"], payload
        assert len(_search_observations(gateway, marker, CONFIG_STORE_ID)) == 1


def test_vector_store_annotation_survives_cache_hit(gateway: Gateway) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        body: Final = _chat_body(model, marker, CONFIG_STORE_ID)
        first: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert first.status_code == 200, first.text
        second: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert second.status_code == 200, second.text
        assert second.json()["id"] == first.json()["id"]
        assert _message_provider_fields(second.json())["search_results"]
        assert len(_chat_observations(gateway, marker)) == 1


@dataclass(frozen=True, slots=True)
class FailureModeGateways:
    annotate: Gateway
    error: Gateway
    upstream: Gateway


def _failure_mode_config(directory: Path, mode: str, upstream_url: str) -> Path:
    config: Final = yaml.safe_load(PROXY_CONFIG.read_text())
    config["litellm_settings"]["vector_store_search_failure_mode"] = mode
    config["vector_store_registry"] = [
        {
            "vector_store_name": "audit-fail-store",
            "litellm_params": {
                "vector_store_id": FAIL_STORE_ID,
                "custom_llm_provider": "openai",
                "api_base": f"{upstream_url}/auditnostore",
                "api_key": "integration-provider-key",
            },
        }
    ]
    path: Final = directory / f"proxy_vector_store_{mode}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def failure_gateways(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FailureModeGateways]:
    with gateway_from_environment() as upstream_gateway, ExitStack() as stack:
        gateways: dict[str, Gateway] = {}
        for mode in ("annotate", "error"):
            directory: Final = tmp_path_factory.mktemp(f"audit_vs_{mode}")
            config: Final = _failure_mode_config(directory, mode, upstream_gateway.upstream_url)
            owned: Final = stack.enter_context(owned_proxy(upstream_gateway, directory, {}, config=config, workers=2))
            gateways[mode] = owned
        yield FailureModeGateways(gateways["annotate"], gateways["error"], upstream_gateway)


def _register_failing_store(gateway: Gateway) -> None:
    _register_scenario(
        gateway,
        "auditnostore",
        {
            "content_type": "application/x-routed",
            "routes": {
                f"POST /vector_stores/{FAIL_STORE_ID}/search": {
                    "content_type": "application/json",
                    "body": {"error": {"message": "scripted store failure"}},
                    "status": 500,
                }
            },
        },
    )


def test_vector_store_search_failure_annotate_mode(failure_gateways: FailureModeGateways) -> None:
    _register_failing_store(failure_gateways.upstream)
    marker: Final = _marker()
    with failure_gateways.annotate.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = failure_gateways.annotate.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, FAIL_STORE_ID), key=key
        )
        assert response.status_code == 200, response.text
        fields: Final = _message_provider_fields(response.json())
        failures: Final = fields.get("vector_store_search_failures")
        assert isinstance(failures, list) and failures, response.text
        assert FAIL_STORE_ID in json.dumps(failures) and "scripted store failure" in json.dumps(failures), failures


def test_vector_store_search_failure_error_mode(failure_gateways: FailureModeGateways) -> None:
    _register_failing_store(failure_gateways.upstream)
    marker: Final = _marker()
    with failure_gateways.error.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = failure_gateways.error.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, FAIL_STORE_ID), key=key
        )
        assert response.status_code != 200, response.text
        assert "error" in response.json(), response.text
        assert "vector store" in response.text.lower() or FAIL_STORE_ID in response.text, response.text
