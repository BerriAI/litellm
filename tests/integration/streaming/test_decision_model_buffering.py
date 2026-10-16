import json
import uuid
from pathlib import Path
from typing import Final, Literal

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse
from pydantic import JsonValue


@pytest.mark.parametrize("endpoint", ["chat/completions", "messages", "responses"])
@pytest.mark.parametrize(
    ("buffering", "probability", "judge_status", "fallback"),
    [
        (True, 1.0, 200, "fail_closed"),
        (True, 0.0, 200, "fail_closed"),
        (False, 1.0, 200, "fail_closed"),
        (None, 1.0, 200, "fail_closed"),
        (True, 0.0, 500, "fail_closed"),
        (True, 0.0, 500, "fail_open"),
    ],
    ids=["buffered-block", "buffered-allow", "unbuffered-block", "default-block", "outage-closed", "outage-open"],
)
def test_decision_model_streaming_buffering(
    gateway: Gateway,
    tmp_path: Path,
    buffering: bool | None,
    probability: float,
    endpoint: str,
    judge_status: int,
    fallback: Literal["fail_closed", "fail_open"],
) -> None:
    identity: Final = uuid.uuid4().hex
    frames: Final = _frames(identity, endpoint)
    chat: Final = register_scenario(
        identity,
        SseResponse(content_type="text/event-stream", frames=frames),
        control_url=gateway.upstream_url,
    )
    try:
        decider: Final = register_scenario(
            identity + "-decider",
            JsonResponse(
                content_type="application/json",
                status=judge_status,
                body={
                    "model": "scripted-decider",
                    "answers": [{"type": "predicate", "name": "bananas", "probability": probability}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            ),
            control_url=gateway.upstream_url,
        )
        try:
            config: Final = {
                **JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())),
                "model_list": [
                    {
                        "model_name": "buffering-chat",
                        "litellm_params": {
                            "model": "anthropic/scripted-chat" if endpoint == "messages" else "openai/scripted-chat",
                            "api_key": "synthetic",
                            "api_base": chat.api_base(),
                        },
                    },
                    {
                        "model_name": "buffering-decider",
                        "litellm_params": {
                            "model": "openai/scripted-decider",
                            "api_key": "synthetic",
                            "api_base": decider.api_base(),
                        },
                    },
                ],
                "guardrails": [
                    {
                        "guardrail_name": "banana-policy",
                        "litellm_params": {
                            "guardrail": "decision_model",
                            "mode": "post_call",
                            "default_on": True,
                            "decision_model": "buffering-decider",
                            "unreachable_fallback": fallback,
                            **({"streaming_buffer_until_moderated": buffering} if buffering is not None else {}),
                            "checks": [
                                {
                                    "name": "bananas",
                                    "instructions": "Does the text mention bananas?",
                                    "action": "block",
                                    "threshold": 0.5,
                                }
                            ],
                        },
                    }
                ],
                "general_settings": {"master_key": gateway.key},
            }
            config_path: Final = tmp_path / "decision-buffering.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with owned_proxy(gateway, tmp_path, {}, config=config_path, workers=2) as candidate:
                response: Final = candidate.request(
                    "POST",
                    f"/v1/{endpoint}",
                    {
                        "model": "buffering-chat",
                        **(
                            {"input": "Tell me about bananas"}
                            if endpoint == "responses"
                            else {"messages": [{"role": "user", "content": "Tell me about bananas"}]}
                        ),
                        **({"max_tokens": 100} if endpoint == "messages" else {}),
                        "stream": True,
                        "cache": {"no-cache": True, "no-store": True},
                    },
                )
                events: Final = tuple(
                    JSON_OBJECT.validate_json(line.removeprefix("data: "))
                    for line in response.text.splitlines()
                    if line.startswith("data: {")
                )
                content: Final = "".join(_event_text(event, endpoint) for event in events)
                if judge_status != 200 and fallback == "fail_closed":
                    assert "Decision model guardrail failed to respond" in response.text, response.text
                    assert content == "", response.text
                elif probability == 0:
                    assert content == "Bananas are fruit.", response.text
                    assert '"error":' not in response.text, response.text
                else:
                    assert "Violated decision model guardrail policy" in response.text, response.text
                    if buffering:
                        assert content == "", response.text
                    else:
                        assert content and "Bananas are fruit.".startswith(content), response.text
        finally:
            delete_scenario(decider)
    finally:
        delete_scenario(chat)


def _event_text(event: dict[str, JsonValue], endpoint: str) -> str:
    if endpoint == "responses":
        return string_value(event.get("delta", "")) if event.get("type") == "response.output_text.delta" else ""
    if endpoint == "messages":
        return string_value(object_value(event.get("delta", {})).get("text", ""))
    choices: Final = list_value(event.get("choices", [{}]))
    return string_value(object_value(object_value(choices[0]).get("delta", {})).get("content", ""))


def _frames(identity: str, endpoint: str) -> tuple[str, ...]:
    pieces: Final = ("Ban", "anas", " are", " fruit", ".")
    if endpoint == "messages":
        events: Final = (
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "scripted-chat",
                    "stop_reason": None,
                    "usage": {"input_tokens": 1},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            *(
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
                for text in pieces
            ),
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        )
        return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}" for event in events)
    if endpoint == "responses":
        message: Final = {
            "id": "msg_" + identity,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "".join(pieces), "annotations": []}],
        }
        response: Final = {
            "id": "resp_" + identity,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "scripted-chat",
            "output": [message],
            "usage": {"input_tokens": 1, "output_tokens": 5, "total_tokens": 6},
        }
        response_events: Final = (
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**response, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {**message, "content": []},
            },
            *(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": index + 2,
                    "item_id": message["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                }
                for index, text in enumerate(pieces)
            ),
            {"type": "response.output_item.done", "sequence_number": 7, "output_index": 0, "item": message},
            {"type": "response.completed", "sequence_number": 8, "response": response},
        )
        return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}" for event in response_events)
    return (
        *(
            f"data: {
                json.dumps(
                    {
                        'id': identity,
                        'object': 'chat.completion.chunk',
                        'created': 1,
                        'model': 'scripted-chat',
                        'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}],
                    }
                )
            }"
            for text in pieces
        ),
        "data: [DONE]",
    )
