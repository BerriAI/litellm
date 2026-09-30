from __future__ import annotations

import base64
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
import yaml
from integration._support.client import Gateway, JsonValue, Scenario, eventually
from integration._support.database import read_rows, write_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import (
    _aws_event_frame,  # pyright: ignore[reportPrivateUsage]  # project Bedrock event-stream encoder
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

import litellm

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
EndpointScope: TypeAlias = Literal["streaming", "non_streaming"]
ModelKind: TypeAlias = Literal["router", "direct"]
StreamingAction: TypeAlias = Literal["converse-stream", "invoke-with-response-stream"]
NonStreamingAction: TypeAlias = Literal["converse", "invoke"]
BedrockAction: TypeAlias = StreamingAction | NonStreamingAction
BEDROCK_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
BEDROCK_FALSE_POSITIVE_MODEL_ID: Final = "anthropic.claude-converse-stream-test-v1:0"
BEDROCK_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
GUARDRAIL_PATH: Final = "/beta/litellm_basic_guardrail_api"
SERVER_STREAMING_CLASSIFICATION_KEY: Final = "litellm_server_streaming_classification"
STREAMING_ACTIONS: Final[tuple[StreamingAction, ...]] = (
    "converse-stream",
    "invoke-with-response-stream",
)
NON_STREAMING_ACTIONS: Final[tuple[NonStreamingAction, ...]] = ("converse", "invoke")
SCOPES: Final[tuple[EndpointScope, ...]] = ("streaming", "non_streaming")
HOSTILE_CLASSIFICATION_CASES: Final = (
    pytest.param("is_streaming_request", True, id="boolean-marker"),
    pytest.param("is_streaming_request", "litellm-server-streaming", id="server-marker-string"),
    pytest.param("litellm_server_streaming_classification", True, id="classification-field"),
)
WORKTREE: Final = Path(__file__).resolve().parents[3]
LITELLM_PATH: Final = Path(litellm.__file__).resolve()
assert LITELLM_PATH.is_relative_to(WORKTREE), (LITELLM_PATH, WORKTREE)
print(f"stream_scope repro litellm import: {LITELLM_PATH}")  # noqa: T201  # required worktree evidence


def _json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _strings(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(chain.from_iterable(_strings(item) for item in value))
    if isinstance(value, dict):
        return tuple(chain.from_iterable(_strings(item) for item in value.values()))
    return ()


def _key_names(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(chain.from_iterable(_key_names(item) for item in value.values()))
    if isinstance(value, list):
        return tuple(chain.from_iterable(_key_names(item) for item in value))
    return ()


def _marker(body: JsonValue) -> str:
    return next(value for value in _strings(body) if value.startswith("scope-"))


def _bedrock_converse_stream(marker: str) -> bytes:
    return b"".join(
        _aws_event_frame(event_type, payload, marker, marker)
        for event_type, payload in (
            ("messageStart", {"role": "assistant"}),
            (
                "contentBlockDelta",
                {"delta": {"text": f"scripted Bedrock reply {marker}"}, "contentBlockIndex": 0},
            ),
            ("messageStop", {"stopReason": "end_turn"}),
            ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
        )
    )


def _invoke_chunk(payload: Mapping[str, JsonValue], marker: str) -> bytes:
    encoded: Final = base64.b64encode(_json(payload)).decode()
    return _aws_event_frame("chunk", {"bytes": encoded}, marker, marker)


def _bedrock_invoke_stream(marker: str) -> bytes:
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": f"msg-{marker}",
                "type": "message",
                "role": "assistant",
                "model": BEDROCK_MODEL_ID,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"scripted Bedrock reply {marker}"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 11, "output_tokens": 4},
        },
        {"type": "message_stop"},
    )
    return b"".join(_invoke_chunk(event, marker) for event in events)


def _provider(request: Request) -> Reply:
    if not request.body:
        return Reply(status=400, body=_json({"error": "empty request body"}))
    body: Final = JSON_OBJECT.validate_json(request.body)
    marker: Final = _marker(body)
    target: Final = request.target.split("?", 1)[0]
    if target.startswith("/passthrough"):
        if body.get("stream") is True:
            streamed_response: Final = _json({"received": body})
            return Reply(
                content_type="text/event-stream",
                chunks=(b"data: " + streamed_response + b"\n\n", b"data: [DONE]\n\n"),
            )
        return Reply(body=_json({"received": body}))
    if target.endswith("/converse-stream"):
        return Reply(body=_bedrock_converse_stream(marker), content_type=BEDROCK_EVENT_STREAM)
    if target.endswith("/invoke-with-response-stream"):
        return Reply(body=_bedrock_invoke_stream(marker), content_type=BEDROCK_EVENT_STREAM)
    if target.endswith("/converse") or target.endswith("/invoke"):
        return Reply(body=_json({"output": f"scripted Bedrock reply {marker}"}))
    if target == "/v1/chat/completions":
        if body.get("stream") is True:
            chunk: Final = {
                "id": f"chatcmpl-{marker}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": f"scripted chat reply {marker}"},
                        "finish_reason": None,
                    }
                ],
            }
            final_chunk: Final = {
                **chunk,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            return Reply(
                content_type="text/event-stream",
                chunks=(
                    b"data: " + _json(chunk) + b"\n\n",
                    b"data: " + _json(final_chunk) + b"\n\n",
                    b"data: [DONE]\n\n",
                ),
            )
        return Reply(
            body=_json(
                {
                    "id": f"chatcmpl-{marker}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"scripted chat reply {marker}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
        )
    return Reply(status=404, body=_json({"error": f"unexpected upstream path: {target}"}))


def _sink(request: Request) -> Reply:
    assert request.target.endswith(GUARDRAIL_PATH), request.target
    assert b"scope-" in request.body, request.body.decode()
    return Reply(body=_json({"action": "NONE"}))


def _rail(name: str, sink: Wire, scope: EndpointScope) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": False,
            "stream_scope": scope,
            "api_base": f"{sink.url}/{name}",
            "api_key": "synthetic-guardrail-key",
        },
    }


def _chat_proxy_config(provider_url: str, guardrails: list[dict[str, JsonValue]]) -> dict[str, object]:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    return {
        **config,
        "guardrails": guardrails,
        "model_list": [
            {
                "model_name": "scope-invalid-config-chat",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{provider_url}/v1",
                    "api_key": "synthetic-provider-key",
                },
            }
        ],
    }


@dataclass(frozen=True, slots=True)
class ReproRig:
    candidate: Gateway
    direct_candidate: Gateway
    scenario: Scenario
    models: Mapping[str, str]
    rails: Mapping[str, str]
    provider: Wire
    sink: Wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ReproRig]:
    with httpx.Client(
        base_url=os.environ["INTEGRATION_PROXY_URL"],
        timeout=30,
        trust_env=False,
    ) as root_client:
        root_gateway: Final = Gateway(
            root_client,
            os.environ.get("INTEGRATION_MASTER_KEY", "sk-integration-master"),
            os.environ["INTEGRATION_UPSTREAM_URL"],
        )
        directory: Final = tmp_path_factory.mktemp("guardrail-stream-scope-repro")
        with wire_server(_provider) as provider, wire_server(_sink) as sink:
            rails: Final = MappingProxyType(
                {
                    "bedrock_streaming": "bedrock_streaming",
                    "bedrock_non_streaming": "bedrock_non_streaming",
                    "chat_streaming": "chat_streaming",
                    "passthrough_streaming": "passthrough_streaming",
                    "passthrough_non_streaming": "passthrough_non_streaming",
                    "passthrough_spoof_streaming": "passthrough_spoof_streaming",
                }
            )
            models: Final = MappingProxyType(
                {
                    "chat": "scope-chat",
                    "bedrock_router": "scope-bedrock-router",
                    "bedrock_false_positive_router": "scope-bedrock-converse-stream-model",
                }
            )
            config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            config["guardrails"] = [
                _rail(rails["bedrock_streaming"], sink, "streaming"),
                _rail(rails["bedrock_non_streaming"], sink, "non_streaming"),
                _rail(rails["chat_streaming"], sink, "streaming"),
                _rail(rails["passthrough_streaming"], sink, "streaming"),
                _rail(rails["passthrough_non_streaming"], sink, "non_streaming"),
                _rail(rails["passthrough_spoof_streaming"], sink, "streaming"),
            ]
            config["model_list"] = [
                {
                    "model_name": models["chat"],
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": f"{provider.url}/v1",
                        "api_key": "synthetic-provider-key",
                    },
                },
                {
                    "model_name": models["bedrock_router"],
                    "litellm_params": {
                        "model": f"bedrock/{BEDROCK_MODEL_ID}",
                        "api_base": provider.url,
                        "aws_access_key_id": "AKIASYNTHETICSTREAMSCOPE",
                        "aws_secret_access_key": "synthetic-bedrock-secret",
                        "aws_region_name": "us-east-1",
                    },
                },
                {
                    "model_name": models["bedrock_false_positive_router"],
                    "litellm_params": {
                        "model": f"bedrock/{BEDROCK_FALSE_POSITIVE_MODEL_ID}",
                        "api_base": provider.url,
                        "aws_access_key_id": "AKIASYNTHETICSTREAMSCOPE",
                        "aws_secret_access_key": "synthetic-bedrock-secret",
                        "aws_region_name": "us-east-1",
                    },
                },
            ]
            config["environment_variables"] = {
                "AWS_BEDROCK_RUNTIME_ENDPOINT": provider.url,
                "AWS_ACCESS_KEY_ID": "AKIASYNTHETICSTREAMSCOPE",
                "AWS_SECRET_ACCESS_KEY": "synthetic-bedrock-secret",
                "AWS_REGION": "us-east-1",
                "AWS_REGION_NAME": "us-east-1",
            }
            config["general_settings"]["pass_through_endpoints"] = [
                {
                    "path": "/pt-forward",
                    "target": f"{provider.url}/passthrough",
                    "include_subpath": True,
                },
                {
                    "path": "/pt-spoof",
                    "target": f"{provider.url}/passthrough",
                    "include_subpath": True,
                    "guardrails": {rails["passthrough_spoof_streaming"]: None},
                },
                {
                    "path": "/pt-scope",
                    "target": f"{provider.url}/passthrough",
                    "include_subpath": True,
                    "guardrails": {
                        rails["passthrough_streaming"]: None,
                        rails["passthrough_non_streaming"]: None,
                    },
                },
            ]
            config_path: Final = directory / "stream-scope-repro.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with owned_proxy_process(root_gateway, directory, {}, config=config_path, workers=1) as owned:
                with owned.gateway.scenario() as scenario:
                    direct_config: Final = {
                        **config,
                        "model_list": [
                            *config["model_list"],
                            {
                                "model_name": f"scope-unused-{uuid.uuid4().hex}*",
                                "litellm_params": {
                                    "model": "openai/gpt-4o-mini",
                                    "api_base": f"{provider.url}/v1",
                                    "api_key": "synthetic-provider-key",
                                },
                            },
                        ],
                    }
                    direct_config_path: Final = directory / "stream-scope-direct.yaml"
                    direct_config_path.write_text(yaml.safe_dump(direct_config))
                    with owned_proxy_process(
                        root_gateway,
                        directory,
                        {},
                        config=direct_config_path,
                        workers=1,
                    ) as direct:
                        yield ReproRig(
                            owned.gateway,
                            direct.gateway,
                            scenario,
                            models,
                            rails,
                            provider,
                            sink,
                        )


def _bedrock_request(
    action: BedrockAction,
    model_path: str,
    marker: str,
) -> tuple[str, dict[str, JsonValue]]:
    if action in ("converse", "converse-stream"):
        body: Final = {
            "messages": [{"role": "user", "content": [{"text": marker}]}],
            "inferenceConfig": {"maxTokens": 16},
        }
    else:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [{"type": "text", "text": marker}]}],
        }
    return f"/bedrock/model/{model_path}/{action}", body


def _matching_requests(wire: Wire, marker: str) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if marker.encode() in request.body)


def _rail_scans(rows: Sequence[Request], rail_name: str, marker: str) -> tuple[Request, ...]:
    return tuple(
        request for request in rows if request.target.startswith(f"/{rail_name}/") and marker.encode() in request.body
    )


def _chat_request_with_scans(
    gateway: Gateway,
    sink: Wire,
    model: str,
    marker: str,
    streamed: bool,
    rail_name: str,
) -> tuple[httpx.Response, tuple[Request, ...]]:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": marker}],
            "stream": streamed,
            "guardrails": [rail_name],
        },
    )
    return response, _rail_scans(sink.drain(), rail_name, marker)


def _spend_row_for_call(call_id: str, content: bytes) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, litellm_call_id, spend, prompt_tokens, completion_tokens, metadata "
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s OR litellm_call_id=%s',
            (call_id, call_id),
        ),
        lambda values: len(values) >= 1,
        seconds=70,
    )
    assert len(rows) == 1, (call_id, rows, content)
    assert call_id in (rows[0]["request_id"], rows[0]["litellm_call_id"]), content
    return rows[0]


@pytest.mark.parametrize("model_kind", ("router", "direct"))
@pytest.mark.parametrize("action", STREAMING_ACTIONS)
@pytest.mark.parametrize("scope", SCOPES)
def test_bedrock_streaming_actions_run_streaming_scoped_rails(
    rig: ReproRig,
    model_kind: ModelKind,
    action: StreamingAction,
    scope: EndpointScope,
) -> None:
    marker: Final = f"scope-bedrock-stream-{uuid.uuid4().hex}"
    model_path: Final = rig.models["bedrock_router"] if model_kind == "router" else BEDROCK_MODEL_ID
    path, body = _bedrock_request(action, model_path, marker)
    expected_body: Final = (
        _bedrock_converse_stream(marker) if action == "converse-stream" else _bedrock_invoke_stream(marker)
    )
    call_id: Final = f"stream-scope-{uuid.uuid4().hex}"
    key: Final = rig.scenario.key(guardrails=[rig.rails[f"bedrock_{scope}"]])
    candidate: Final = rig.candidate if model_kind == "router" else rig.direct_candidate
    response: Final = candidate.request(
        "POST",
        path,
        body,
        key=key,
        headers={"x-litellm-call-id": call_id},
    )
    assert response.status_code == 200, response.content
    assert response.headers.get("content-type") == BEDROCK_EVENT_STREAM, dict(response.headers)
    assert response.content == expected_body, response.content
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows, response.content)
    provider_body: Final = JSON_OBJECT.validate_json(provider_rows[0].body)
    key_names: Final = _key_names(provider_body)
    assert "is_streaming_request" not in key_names, provider_body
    assert not tuple(name for name in key_names if name.startswith("litellm_")), provider_body
    _spend_row_for_call(call_id, response.content)
    sink_rows: Final = _rail_scans(rig.sink.drain(), rig.rails[f"bedrock_{scope}"], marker)
    expected_scans: Final = int(scope == "streaming")
    assert len(sink_rows) == expected_scans, (marker, model_kind, action, scope, sink_rows, response.content)


@pytest.mark.parametrize("model_kind", ("router", "direct"))
@pytest.mark.parametrize("action", NON_STREAMING_ACTIONS)
@pytest.mark.parametrize("scope", SCOPES)
def test_bedrock_non_streaming_actions_run_non_streaming_scoped_rails(
    rig: ReproRig,
    model_kind: ModelKind,
    action: NonStreamingAction,
    scope: EndpointScope,
) -> None:
    marker: Final = f"scope-bedrock-nonstream-{uuid.uuid4().hex}"
    model_path: Final = rig.models["bedrock_router"] if model_kind == "router" else BEDROCK_MODEL_ID
    path, body = _bedrock_request(action, model_path, marker)
    key: Final = rig.scenario.key(guardrails=[rig.rails[f"bedrock_{scope}"]])
    candidate: Final = rig.candidate if model_kind == "router" else rig.direct_candidate
    response: Final = candidate.request("POST", path, body, key=key)
    assert response.status_code == 200, response.text
    assert marker in response.text, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows, response.text)
    provider_body: Final = JSON_OBJECT.validate_json(provider_rows[0].body)
    key_names: Final = _key_names(provider_body)
    assert "is_streaming_request" not in key_names, provider_body
    assert not tuple(name for name in key_names if name.startswith("litellm_")), provider_body
    sink_rows: Final = _rail_scans(rig.sink.drain(), rig.rails[f"bedrock_{scope}"], marker)
    expected_scans: Final = int(scope == "non_streaming")
    assert len(sink_rows) == expected_scans, (marker, model_kind, action, scope, sink_rows, response.text)


@pytest.mark.parametrize("model_kind", ("router", "direct"))
def test_bedrock_model_id_streaming_action_text_on_converse_is_non_streaming(
    rig: ReproRig,
    model_kind: ModelKind,
) -> None:
    marker: Final = f"scope-bedrock-converse-model-{uuid.uuid4().hex}"
    model_path: Final = (
        rig.models["bedrock_false_positive_router"] if model_kind == "router" else BEDROCK_FALSE_POSITIVE_MODEL_ID
    )
    path, body = _bedrock_request("converse", model_path, marker)
    call_id: Final = f"stream-scope-bedrock-{uuid.uuid4().hex}"
    key: Final = rig.scenario.key(
        guardrails=[
            rig.rails["bedrock_streaming"],
            rig.rails["bedrock_non_streaming"],
        ]
    )
    candidate: Final = rig.candidate if model_kind == "router" else rig.direct_candidate
    response: Final = candidate.request(
        "POST",
        path,
        body,
        key=key,
        headers={"x-litellm-call-id": call_id},
    )
    assert response.status_code == 200, response.text
    assert marker in response.text, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, model_kind, provider_rows, response.text)
    sink_rows: Final = rig.sink.drain()
    streaming_rows: Final = _rail_scans(sink_rows, rig.rails["bedrock_streaming"], marker)
    non_streaming_rows: Final = _rail_scans(sink_rows, rig.rails["bedrock_non_streaming"], marker)
    assert streaming_rows == (), (marker, model_kind, streaming_rows, response.text)
    assert len(non_streaming_rows) == 1, (marker, model_kind, non_streaming_rows, response.text)


@pytest.mark.parametrize("streamed", (False, True), ids=("stream-absent", "stream-true"))
def test_configured_passthrough_forwards_caller_is_streaming_request_field(
    rig: ReproRig,
    streamed: bool,
) -> None:
    marker: Final = f"scope-passthrough-forward-{uuid.uuid4().hex}"
    caller_value: Final = f"caller-{uuid.uuid4().hex}"
    body: Final = {
        "marker": marker,
        "is_streaming_request": caller_value,
        **({"stream": True} if streamed else {}),
    }
    response: Final = rig.candidate.request("POST", "/pt-forward", body)
    assert response.status_code == 200, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows, response.text)
    upstream_body: Final = JSON_OBJECT.validate_json(provider_rows[0].body)
    assert upstream_body.get("is_streaming_request") == caller_value, (
        marker,
        caller_value,
        upstream_body,
        response.text,
    )
    assert upstream_body == body, (marker, body, upstream_body, response.text)
    if streamed:
        assert response.headers.get("content-type", "").lower().startswith("text/event-stream"), dict(response.headers)
        event_body: Final = response.text.removeprefix("data: ").split("\n", maxsplit=1)[0]
        response_body: Final = JSON_OBJECT.validate_json(event_body)
    else:
        response_body = JSON_OBJECT.validate_json(response.content)
    assert response_body == {"received": body}, response.text


@pytest.mark.parametrize(("hostile_field", "hostile_value"), HOSTILE_CLASSIFICATION_CASES)
def test_client_cannot_spoof_server_stream_classification(
    rig: ReproRig,
    hostile_field: str,
    hostile_value: JsonValue,
) -> None:
    chat_marker: Final = f"scope-chat-spoof-{uuid.uuid4().hex}"
    chat_body: Final = {
        "model": rig.models["chat"],
        "messages": [{"role": "user", "content": chat_marker}],
        "stream": False,
        hostile_field: hostile_value,
    }
    chat_key: Final = rig.scenario.key(guardrails=[rig.rails["chat_streaming"]])
    chat_response: Final = rig.candidate.request(
        "POST",
        "/v1/chat/completions",
        chat_body,
        key=chat_key,
    )
    chat_provider_rows: Final = _matching_requests(rig.provider, chat_marker)
    chat_upstream_body: Final = chat_provider_rows[0].body.decode() if chat_provider_rows else "<no upstream request>"
    print(  # noqa: T201  # required chat hostile-body observation
        f"chat hostile {hostile_field}={hostile_value!r}: "
        f"status={chat_response.status_code}, response={chat_response.text!r}, upstream={chat_upstream_body}"
    )
    assert chat_response.status_code == 200, chat_response.text
    assert len(chat_provider_rows) == 1, (chat_marker, chat_provider_rows, chat_response.text)
    assert JSON_OBJECT.validate_json(chat_provider_rows[0].body) == {
        "messages": [{"role": "user", "content": chat_marker}],
        "model": "gpt-4o-mini",
        hostile_field: hostile_value,
    }, (hostile_field, hostile_value, chat_upstream_body)
    assert _rail_scans(rig.sink.drain(), rig.rails["chat_streaming"], chat_marker) == (), (
        chat_marker,
        hostile_field,
        hostile_value,
        chat_response.text,
    )


@pytest.mark.parametrize(("hostile_field", "hostile_value"), HOSTILE_CLASSIFICATION_CASES)
def test_configured_passthrough_cannot_spoof_server_stream_classification(
    rig: ReproRig,
    hostile_field: str,
    hostile_value: JsonValue,
) -> None:
    passthrough_marker: Final = f"scope-passthrough-spoof-{uuid.uuid4().hex}"
    passthrough_body: Final = {
        "marker": passthrough_marker,
        "stream": False,
        hostile_field: hostile_value,
    }
    passthrough_response: Final = rig.candidate.request("POST", "/pt-spoof", passthrough_body)
    assert passthrough_response.status_code == 200, passthrough_response.text
    passthrough_provider_rows: Final = _matching_requests(rig.provider, passthrough_marker)
    assert len(passthrough_provider_rows) == 1, (
        passthrough_marker,
        passthrough_provider_rows,
        passthrough_response.text,
    )
    assert _rail_scans(rig.sink.drain(), rig.rails["passthrough_spoof_streaming"], passthrough_marker) == (), (
        passthrough_marker,
        hostile_field,
        hostile_value,
        passthrough_response.text,
    )


@pytest.mark.parametrize(
    ("streamed", "request_fields"),
    ((True, {"stream": True}), (False, {"stream": False}), (False, {})),
    ids=("stream-true", "stream-false", "stream-absent"),
)
def test_passthrough_scope_follows_proxy_stream_decision(
    rig: ReproRig,
    streamed: bool,
    request_fields: dict[str, JsonValue],
) -> None:
    marker: Final = f"scope-passthrough-scope-{uuid.uuid4().hex}"
    body: Final = {"marker": marker, **request_fields}
    response: Final = rig.candidate.request("POST", "/pt-scope", body)
    assert response.status_code == 200, response.text
    if streamed:
        expected_frame: Final = f"data: {_json({'received': body}).decode()}\n\ndata: [DONE]\n\n"
        assert response.headers.get("content-type", "").lower().startswith("text/event-stream"), dict(response.headers)
        assert response.text == expected_frame, response.text
        assert response.headers.get("transfer-encoding", "").lower() == "chunked", dict(response.headers)
    else:
        response_body: Final = JSON_OBJECT.validate_json(response.content)
        assert response_body == {"received": body}, response.text
        assert "content-length" in response.headers, dict(response.headers)
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows, response.text)
    upstream_body: Final = JSON_OBJECT.validate_json(provider_rows[0].body)
    assert upstream_body == body, (marker, body, upstream_body, response.text)
    assert SERVER_STREAMING_CLASSIFICATION_KEY not in upstream_body, upstream_body
    sink_rows: Final = rig.sink.drain()
    streaming_rows: Final = _rail_scans(sink_rows, rig.rails["passthrough_streaming"], marker)
    non_streaming_rows: Final = _rail_scans(sink_rows, rig.rails["passthrough_non_streaming"], marker)
    assert len(streaming_rows) == int(streamed), (marker, streamed, streaming_rows, response.text)
    assert len(non_streaming_rows) == int(not streamed), (marker, streamed, non_streaming_rows, response.text)


def test_invalid_yaml_stream_scope_keeps_rail_running_on_both_shapes(rig: ReproRig, tmp_path: Path) -> None:
    name: Final = f"scope-invalid-yaml-{uuid.uuid4().hex}"
    invalid_rail: Final = {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": False,
            "stream_scope": "sometimes",
            "api_base": f"{rig.sink.url}/{name}",
            "api_key": "synthetic-guardrail-key",
        },
    }
    config: Final = _chat_proxy_config(rig.provider.url, [invalid_rail])
    config_path: Final = tmp_path / "invalid-stream-scope.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with owned_proxy_process(rig.candidate, tmp_path, {}, config=config_path, workers=1) as owned:
        markers: Final = tuple(f"scope-invalid-yaml-{int(streamed)}-{uuid.uuid4().hex}" for streamed in (False, True))
        observations: Final = tuple(
            _chat_request_with_scans(
                owned.gateway,
                rig.sink,
                "scope-invalid-config-chat",
                marker,
                streamed,
                name,
            )
            for streamed, marker in zip((False, True), markers)
        )
        assert tuple(response.status_code for response, _ in observations) == (200, 200), tuple(
            response.text for response, _ in observations
        )
        assert tuple(len(sink_rows) for _, sink_rows in observations) == (1, 1), (markers, observations)


def test_persisted_invalid_stream_scope_row_stays_readable_and_enforced(
    rig: ReproRig,
    tmp_path: Path,
) -> None:
    guardrail_id: Final = str(uuid.uuid4())
    guardrail_name: Final = f"scope-invalid-persisted-{uuid.uuid4().hex}"
    params: Final = {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": False,
        "api_base": f"{rig.sink.url}/{guardrail_name}",
        "api_key": "synthetic-guardrail-key",
        "stream_scope": "sometimes",
    }
    database_url: Final = os.environ.get("INTEGRATION_PROXY_DATABASE_URL") or os.environ["DATABASE_URL"]
    write_rows(
        'INSERT INTO "LiteLLM_GuardrailsTable" '
        '("guardrail_id", "guardrail_name", "litellm_params", "created_at", "updated_at") '
        "VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (guardrail_id, guardrail_name, json.dumps(params)),
        database_url=database_url,
    )
    try:
        config: Final = _chat_proxy_config(rig.provider.url, [])
        config_path: Final = tmp_path / "persisted-invalid-stream-scope.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(rig.candidate, tmp_path, {}, config=config_path, workers=1) as owned:
            info: Final = eventually(
                lambda: owned.gateway.request("GET", f"/guardrails/{guardrail_id}/info"),
                lambda response: response.status_code != 404,
                seconds=20,
            )
            listed: Final = owned.gateway.request("GET", "/v2/guardrails/list")
            list_payload: Final = JSON_OBJECT.validate_json(listed.content) if listed.status_code == 200 else {}
            listed_guardrails: Final = list_payload.get("guardrails")
            includes_row: Final = isinstance(listed_guardrails, list) and any(
                isinstance(row, dict) and row.get("guardrail_id") == guardrail_id for row in listed_guardrails
            )
            markers: Final = (
                f"scope-invalid-persisted-0-{uuid.uuid4().hex}",
                f"scope-invalid-persisted-1-{uuid.uuid4().hex}",
            )
            observations: Final = tuple(
                _chat_request_with_scans(
                    owned.gateway,
                    rig.sink,
                    "scope-invalid-config-chat",
                    marker,
                    streamed,
                    guardrail_name,
                )
                for streamed, marker in zip((False, True), markers)
            )
            assert (
                info.status_code == 200
                and listed.status_code == 200
                and includes_row
                and tuple(response.status_code for response, _ in observations) == (200, 200)
                and tuple(len(sink_rows) for _, sink_rows in observations) == (1, 1)
            ), {
                "info": (info.status_code, info.text),
                "list": (listed.status_code, listed.text),
                "includes_row": includes_row,
                "responses": tuple((response.status_code, response.text) for response, _ in observations),
                "scan_counts": tuple(len(sink_rows) for _, sink_rows in observations),
            }
    finally:
        write_rows(
            'DELETE FROM "LiteLLM_GuardrailsTable" WHERE guardrail_id=%s',
            (guardrail_id,),
            database_url=database_url,
        )


def test_management_rejects_invalid_stream_scope(rig: ReproRig) -> None:
    name: Final = f"scope-invalid-management-{uuid.uuid4().hex}"
    invalid_params: Final = {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": False,
        "api_base": f"{rig.sink.url}/{name}",
        "api_key": "synthetic-guardrail-key",
        "stream_scope": "sometimes",
    }
    created_invalid: Final = rig.candidate.request(
        "POST",
        "/guardrails",
        {"guardrail": {"guardrail_name": name, "litellm_params": invalid_params}},
    )
    assert created_invalid.status_code == 422, created_invalid.text

    valid_params: Final = {**invalid_params, "stream_scope": "both"}
    created: Final = rig.candidate.request(
        "POST",
        "/guardrails",
        {"guardrail": {"guardrail_name": name, "litellm_params": valid_params}},
    )
    assert created.status_code == 200, created.text
    guardrail_id: Final = str(created.json()["guardrail_id"])
    try:
        put_response: Final = rig.candidate.request(
            "PUT",
            f"/guardrails/{guardrail_id}",
            {"guardrail": {"guardrail_name": name, "litellm_params": invalid_params}},
        )
        patch_response: Final = rig.candidate.request(
            "PATCH",
            f"/guardrails/{guardrail_id}",
            {"litellm_params": {"stream_scope": "sometimes"}},
        )
        assert put_response.status_code == 422, put_response.text
        assert patch_response.status_code == 422, patch_response.text
    finally:
        deleted: Final = rig.candidate.request("DELETE", f"/guardrails/{guardrail_id}")
        assert deleted.status_code == 200, deleted.text
