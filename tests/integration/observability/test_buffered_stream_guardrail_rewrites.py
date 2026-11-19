import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, JsonValue, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

_CANARY: Final = "jane.doe@canary-pii.example"
_MASK: Final = "[MASKED_EMAIL]"
_GUARDRAIL_PATH: Final = "/beta/litellm_basic_guardrail_api"
_BEDROCK_GUARDRAIL_ID: Final = "synthetic" + uuid.uuid4().hex[:8]
_TEXT_ONLY_PROMPT: Final = "answer in text only"


def _json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _sse(data: object, event: str | None = None) -> bytes:
    prefix: Final = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {json.dumps(data)}\n\n".encode()


def _masked(value: JsonValue) -> JsonValue:
    return JSON_OBJECT.validate_json(json.dumps({"value": value}).replace(_CANARY, _MASK))["value"]


def _masking_guardrail(request: Request) -> Reply:
    assert request.target == _GUARDRAIL_PATH, request.target
    body: Final = JSON_OBJECT.validate_json(request.body)
    if _CANARY not in request.body.decode():
        return Reply(body=_json({"action": "NONE"}))
    rewrite: Final = {key: _masked(body[key]) for key in ("texts", "tool_calls") if key in body}
    return Reply(body=_json({"action": "GUARDRAIL_INTERVENED", **rewrite}))


def _bedrock_masking_guardrail(request: Request) -> Reply:
    assert request.target == f"/guardrail/{_BEDROCK_GUARDRAIL_ID}/version/DRAFT/apply", request.target
    body: Final = JSON_OBJECT.validate_json(request.body)
    if _CANARY not in request.body.decode():
        return Reply(body=_json({"action": "NONE", "outputs": [], "assessments": []}))
    pii: Final = {"type": "EMAIL", "match": _CANARY, "action": "ANONYMIZED"}
    return Reply(
        body=_json(
            {
                "action": "GUARDRAIL_INTERVENED",
                "outputs": _masked([{"text": item["text"]["text"]} for item in body["content"]]),
                "assessments": [{"sensitiveInformationPolicy": {"piiEntities": [pii]}}],
            }
        )
    )


def _text_only_chat_stream() -> tuple[bytes, ...]:
    base: Final = {"id": "chatcmpl-canary", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return (
        *(
            _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": part}}]})
            for part in ("I will email ", _CANARY, " now.")
        ),
        _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    )


def _chat_stream() -> tuple[bytes, ...]:
    base: Final = {"id": "chatcmpl-canary", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    tool_call: Final = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "send_email"}}
    arguments: Final = ('{"to": "', _CANARY, '"}')
    return (
        _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "I will email "}}]}),
        _sse({**base, "choices": [{"index": 0, "delta": {"content": f"{_CANARY} now."}}]}),
        _sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{**tool_call, "function": {"name": "send_email", "arguments": ""}}]},
                    }
                ],
            }
        ),
        *(
            _sse(
                {
                    **base,
                    "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": part}}]}}],
                }
            )
            for part in arguments
        ),
        _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        b"data: [DONE]\n\n",
    )


def _anthropic_stream() -> tuple[bytes, ...]:
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_canary",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I will email "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"{_CANARY} now."}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "send_email", "input": {}},
        },
        *(
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": part}}
            for part in ('{"to": "', _CANARY, '"}')
        ),
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    )
    return tuple(_sse(event, event=str(event["type"])) for event in events)


def _provider(request: Request) -> Reply:
    target: Final = request.target.split("?", 1)[0]
    if target.endswith("/messages"):
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream())
    if target.endswith("/chat/completions") and _TEXT_ONLY_PROMPT in request.body.decode():
        return Reply(content_type="text/event-stream", chunks=_text_only_chat_stream())
    if target.endswith("/chat/completions"):
        return Reply(content_type="text/event-stream", chunks=_chat_stream())
    return Reply(status=404, body=_json({"error": f"unexpected target {target}"}))


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    guardrail: str
    bedrock_guardrail: str
    openai_model: str
    bridged_responses_model: str
    anthropic_model: str
    policy: Wire
    bedrock_policy: Wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("buffered-stream-rewrites")
    guardrail: Final = f"buffered-masker-{uuid.uuid4().hex}"
    bedrock_guardrail: Final = f"bedrock-masker-{uuid.uuid4().hex}"
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = directory / "buffered-stream-rewrites.yaml"
    with (
        wire_server(_masking_guardrail) as policy,
        wire_server(_bedrock_masking_guardrail) as bedrock_policy,
        wire_server(_provider) as upstream,
    ):
        config.write_text(
            yaml.safe_dump(
                {
                    **stock,
                    "guardrails": [
                        {
                            "guardrail_name": guardrail,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": "post_call",
                                "default_on": False,
                                "api_base": policy.url,
                                "api_key": "synthetic-guardrail-key",
                                "streaming_buffer_until_moderated": True,
                            },
                        },
                        {
                            "guardrail_name": bedrock_guardrail,
                            "litellm_params": {
                                "guardrail": "bedrock",
                                "mode": "post_call",
                                "default_on": False,
                                "mask_response_content": True,
                                "streaming_buffer_release_on_scan": True,
                                "streaming_sampling_rate": 1,
                                "guardrailIdentifier": _BEDROCK_GUARDRAIL_ID,
                                "guardrailVersion": "DRAFT",
                                "aws_region_name": "us-east-1",
                                "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
                                "aws_secret_access_key": "synthetic-secret",
                                "aws_bedrock_runtime_endpoint": bedrock_policy.url,
                            },
                        },
                    ],
                }
            )
        )
        with (
            gateway_from_environment() as shared,
            owned_proxy(shared, directory, {}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            yield _Rig(
                gateway=candidate,
                guardrail=guardrail,
                bedrock_guardrail=bedrock_guardrail,
                openai_model=scenario.model(api_base=upstream.url + "/v1", api_key="synthetic-openai-key"),
                bridged_responses_model=scenario.model(
                    api_base=upstream.url + "/v1", api_key="synthetic-openai-key", use_chat_completions_api=True
                ),
                anthropic_model=scenario.model(
                    model="anthropic/claude-sonnet-4-5", api_base=upstream.url, api_key="synthetic-anthropic-key"
                ),
                policy=policy,
                bedrock_policy=bedrock_policy,
            )


def _streamed_body(rig: _Rig, path: str, body: Mapping[str, JsonValue], guardrail: str) -> str:
    with rig.gateway.client.stream(
        "POST",
        path,
        json={**body, "stream": True, "guardrails": [guardrail]},
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
        timeout=httpx.Timeout(60),
    ) as response:
        text: Final = response.read().decode()
        assert response.status_code == 200, text
        return text


def _request(rig: _Rig, route: str) -> tuple[str, dict[str, JsonValue]]:
    prompt: Final[list[JsonValue]] = [{"role": "user", "content": "email jane"}]
    if route == "chat":
        return "/v1/chat/completions", {"model": rig.openai_model, "messages": prompt}
    if route == "messages":
        return "/v1/messages", {"model": rig.anthropic_model, "max_tokens": 64, "messages": prompt}
    return "/v1/responses", {"model": rig.bridged_responses_model, "input": "email jane"}


@pytest.mark.parametrize("route", ["chat", "messages", "responses"])
def test_buffered_stream_reaches_the_client_with_the_guardrail_masks_in_text_and_tool_arguments(
    rig: _Rig, route: str
) -> None:
    rig.policy.drain()
    path, body = _request(rig, route)

    streamed: Final = _streamed_body(rig, path, body, rig.guardrail)

    scans: Final = rig.policy.drain()
    assert len(scans) == 1, scans
    assert _CANARY in scans[0].body.decode()
    assert _CANARY not in streamed, streamed
    assert streamed.count(_MASK) >= 2, streamed


def test_masking_guardrail_that_releases_on_scan_holds_the_responses_stream_and_releases_it_masked(rig: _Rig) -> None:
    rig.bedrock_policy.drain()
    body: Final[dict[str, JsonValue]] = {"model": rig.bridged_responses_model, "input": _TEXT_ONLY_PROMPT}

    streamed: Final = _streamed_body(rig, "/v1/responses", body, rig.bedrock_guardrail)

    scans: Final = rig.bedrock_policy.drain()
    assert len(scans) == 1, [scan.body for scan in scans]
    assert _CANARY in scans[0].body.decode()
    assert _CANARY not in streamed, streamed
    assert _MASK in streamed, streamed
