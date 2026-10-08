import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.responses_stream import frame, response_object
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_SSN: Final = "123-45-6789"
_MASKED_SSN: Final = "{US_SOCIAL_SECURITY_NUMBER}"
_PIECES: Final = ("Customer ", "record: ", "SSN ", _SSN, " is on ", "file.")
_CLEAN_PIECES: Final = ("Customer ", "record: ", "nothing ", "sensitive ", "is on ", "file.")
_ARGUMENTS: Final = '{"city": "Paris"}'
_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]


def _bedrock_policy(guardrail_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.target == f"/guardrail/{guardrail_id}/version/DRAFT/apply", request.target
        body: Final = json.loads(request.body)
        assert body["source"] == "OUTPUT", body
        texts: Final = [item["text"]["text"] for item in body["content"]]
        if not any(_SSN in text for text in texts):
            return Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())
        return Reply(
            body=json.dumps(
                {
                    "action": "GUARDRAIL_INTERVENED",
                    "outputs": [{"text": text.replace(_SSN, _MASKED_SSN)} for text in texts],
                    "assessments": [
                        {
                            "sensitiveInformationPolicy": {
                                "piiEntities": [
                                    {"type": "US_SOCIAL_SECURITY_NUMBER", "match": _SSN, "action": "ANONYMIZED"}
                                ]
                            }
                        }
                    ],
                }
            ).encode()
        )

    return respond


def _responses_stream(identity: str, pieces: tuple[str, ...]) -> tuple[bytes, ...]:
    text: Final = "".join(pieces)
    message: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": "msg_scripted",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    call: Final[dict[str, JsonValue]] = {
        "type": "function_call",
        "id": "fc_scripted",
        "call_id": "call_scripted",
        "name": "get_weather",
        "arguments": _ARGUMENTS,
        "status": "completed",
    }
    part: Final = {"item_id": "msg_scripted", "output_index": 0, "content_index": 0}
    usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 11,
        "output_tokens": 9,
        "total_tokens": 20,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    events: Final[list[dict[str, JsonValue]]] = [
        {"type": "response.created", "response": response_object(identity, "in_progress")},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {"type": "response.content_part.added", **part, "part": {"type": "output_text", "text": "", "annotations": []}},
        *({"type": "response.output_text.delta", **part, "delta": piece} for piece in pieces),
        {"type": "response.output_text.done", **part, "text": text},
        {
            "type": "response.content_part.done",
            **part,
            "part": {"type": "output_text", "text": text, "annotations": []},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**call, "status": "in_progress", "arguments": ""},
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_scripted",
            "output_index": 1,
            "delta": _ARGUMENTS,
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_scripted",
            "output_index": 1,
            "arguments": _ARGUMENTS,
        },
        {"type": "response.output_item.done", "output_index": 1, "item": call},
        {
            "type": "response.completed",
            "response": response_object(identity, "completed", output=[message, call], usage=usage),
        },
    ]
    return tuple(frame({**event, "sequence_number": number}) for number, event in enumerate(events))


def _chat_stream(identity: str, pieces: tuple[str, ...]) -> tuple[bytes, ...]:
    def chunk(delta: dict[str, str], finish: str | None = None) -> bytes:
        payload: Final = {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    return (
        chunk({"role": "assistant", "content": ""}),
        *(chunk({"content": piece}) for piece in pieces),
        chunk({}, "stop"),
        b"data: [DONE]\n\n",
    )


def _provider(pieces: tuple[str, ...]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        body: Final = json.loads(request.body)
        assert body["stream"] is True, body
        if request.target == "/v1/responses":
            return Reply(content_type="text/event-stream", chunks=_responses_stream("resp_" + uuid.uuid4().hex, pieces))
        assert request.target == "/v1/chat/completions", request.target
        return Reply(content_type="text/event-stream", chunks=_chat_stream("chatcmpl-" + uuid.uuid4().hex, pieces))

    return respond


def _config(tmp_path: Path, guardrail_id: str, policy: Wire, **flags: JsonValue) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": "bedrock-output-" + uuid.uuid4().hex,
            "litellm_params": {
                "guardrail": "bedrock",
                "mode": "post_call",
                "default_on": True,
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "DRAFT",
                "aws_region_name": "us-east-1",
                "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
                "aws_secret_access_key": "synthetic-secret",
                "aws_bedrock_runtime_endpoint": policy.url,
                **flags,
            },
        }
    ]
    path: Final = tmp_path / "bedrock-streaming.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _events(body: str) -> list[dict[str, JsonValue]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _assert_responses_stream_delivers(body: str, expected_text: str) -> None:
    events: Final = _events(body)
    deltas: Final = "".join(str(event["delta"]) for event in events if event["type"] == "response.output_text.delta")
    assert deltas == expected_text, body
    (created,) = (event["response"] for event in events if event["type"] == "response.created")
    (completed,) = (event["response"] for event in events if event["type"] == "response.completed")
    assert isinstance(created, dict) and isinstance(completed, dict), body
    assert completed["id"] == created["id"], body
    output: Final = completed["output"]
    assert isinstance(output, list), body
    assert [item["type"] for item in output if isinstance(item, dict)] == ["message", "function_call"], body
    message, call = output
    assert isinstance(message, dict) and isinstance(call, dict), body
    assert message["content"] == [{"type": "output_text", "text": expected_text, "annotations": []}], body
    assert (call["name"], call["arguments"], call["call_id"]) == ("get_weather", _ARGUMENTS, "call_scripted"), body


def _stream_responses(candidate: Gateway, model: str) -> str:
    response: Final = candidate.request(
        "POST",
        "/v1/responses",
        {"model": model, "input": "Read back the customer record", "stream": True, "tools": _TOOLS},
    )
    assert response.status_code == 200, response.text
    return response.text


def test_bedrock_held_responses_stream_delivers_masked_text_with_tool_call_and_id(
    gateway: Gateway, tmp_path: Path
) -> None:
    guardrail_id: Final = "synthetic" + uuid.uuid4().hex[:8]
    with wire_server(_bedrock_policy(guardrail_id)) as policy, wire_server(_provider(_PIECES)) as upstream:
        path: Final = _config(tmp_path, guardrail_id, policy)
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            body: Final = _stream_responses(candidate, model)
            assert _SSN not in body, body
            _assert_responses_stream_delivers(body, "".join(_PIECES).replace(_SSN, _MASKED_SSN))
            assert len(policy.drain()) == 1


def test_bedrock_held_responses_stream_without_findings_is_released_unchanged(gateway: Gateway, tmp_path: Path) -> None:
    guardrail_id: Final = "synthetic" + uuid.uuid4().hex[:8]
    with wire_server(_bedrock_policy(guardrail_id)) as policy, wire_server(_provider(_CLEAN_PIECES)) as upstream:
        path: Final = _config(tmp_path, guardrail_id, policy)
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            _assert_responses_stream_delivers(_stream_responses(candidate, model), "".join(_CLEAN_PIECES))
            assert len(policy.drain()) == 1


def test_bedrock_mask_response_content_holds_responses_stream_and_delivers_masked_text(
    gateway: Gateway, tmp_path: Path
) -> None:
    guardrail_id: Final = "synthetic" + uuid.uuid4().hex[:8]
    with wire_server(_bedrock_policy(guardrail_id)) as policy, wire_server(_provider(_PIECES)) as upstream:
        path: Final = _config(tmp_path, guardrail_id, policy, mask_response_content=True)
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            body: Final = _stream_responses(candidate, model)
            assert _SSN not in body, body
            _assert_responses_stream_delivers(body, "".join(_PIECES).replace(_SSN, _MASKED_SSN))


def test_bedrock_windowed_stream_never_releases_text_bedrock_anonymized(gateway: Gateway, tmp_path: Path) -> None:
    guardrail_id: Final = "synthetic" + uuid.uuid4().hex[:8]
    with wire_server(_bedrock_policy(guardrail_id)) as policy, wire_server(_provider(_PIECES)) as upstream:
        path: Final = _config(
            tmp_path, guardrail_id, policy, streaming_buffer_release_on_scan=True, streaming_sampling_rate=2
        )
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            chat: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "stream": True, "messages": [{"role": "user", "content": "Read back the record"}]},
            )
            assert _SSN not in chat.text, chat.text
            assert chat.status_code == 200, chat.text
            assert "Violated guardrail policy" in chat.text, chat.text
            responses: Final = candidate.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": "Read back the customer record", "stream": True, "tools": _TOOLS},
            )
            assert _SSN not in responses.text, responses.text
            assert "Violated guardrail policy" in responses.text, responses.text
            assert policy.drain(), "the guardrail never scanned the stream"


def test_bedrock_held_responses_stream_served_from_cache_masks_every_event(gateway: Gateway, tmp_path: Path) -> None:
    guardrail_id: Final = "synthetic" + uuid.uuid4().hex[:8]
    with wire_server(_bedrock_policy(guardrail_id)) as policy, wire_server(_provider(_PIECES)) as upstream:
        path: Final = _config(tmp_path, guardrail_id, policy)
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )

            def stream_and_count_upstream_calls() -> tuple[str, int]:
                body: Final = _stream_responses(candidate, model)
                return body, len([seen for seen in upstream.drain() if seen.method == "POST"])

            cached, _ = eventually(stream_and_count_upstream_calls, lambda observed: observed[1] == 0, seconds=30)
            event_types: Final = {str(event["type"]) for event in _events(cached)}
            assert {"response.content_part.added", "response.output_item.added"} <= event_types, cached
            assert _SSN not in cached, cached
            _assert_responses_stream_delivers(cached, "".join(_PIECES).replace(_SSN, _MASKED_SSN))
