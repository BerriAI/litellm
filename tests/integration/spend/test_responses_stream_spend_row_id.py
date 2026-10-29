import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.responses_stream import frame, response_object
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_NATIVE_MODEL: Final = "openai/gpt-5.6"
_BRIDGED_MODEL: Final = "deepseek/deepseek-v4-pro"
_ANSWER: Final = "streamed answer"
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}
_JSON_LIST: Final = TypeAdapter(list[dict[str, JsonValue]])


def _sse(payload: dict[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _native_stream(upstream_id: str) -> Reply:
    message: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}],
    }
    completed: Final = response_object(upstream_id, "completed", output=[message], usage=_USAGE)
    return Reply(
        content_type="text/event-stream",
        chunks=(
            frame(
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": response_object(upstream_id, "in_progress"),
                }
            ),
            frame(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": _ANSWER,
                }
            ),
            frame({"type": "response.completed", "sequence_number": 2, "response": completed}),
        ),
    )


def _chat_stream(upstream_id: str, model: str) -> Reply:
    def chunk(delta: dict[str, JsonValue], finish_reason: str | None) -> dict[str, JsonValue]:
        return {
            "id": upstream_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse(chunk({"role": "assistant", "content": _ANSWER}, None)),
            _sse(
                {
                    **chunk({}, "stop"),
                    "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
                }
            ),
            b"data: [DONE]\n\n",
        ),
    )


def _respond(request: Request) -> Reply:
    if request.method == "GET" and request.target == "/v1/models":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert body.get("stream") is True, body
    upstream_id: Final = f"upstream-{uuid4().hex}"
    if request.target == "/v1/responses":
        return _native_stream(f"resp_{upstream_id}")
    assert request.target == "/v1/chat/completions", request.target
    return _chat_stream(f"chatcmpl-{upstream_id}", string_value(body["model"]))


def _terminal_response_id(gateway: Gateway, key: str, body: dict[str, JsonValue]) -> str:
    response: Final = gateway.request("POST", "/v1/responses", body, key=key)
    assert response.status_code == 200, response.text
    frames: Final = tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert frames and frames[-1]["type"] == "response.completed", response.text
    advertised: Final = tuple(
        dict.fromkeys(string_value(object_value(event["response"])["id"]) for event in frames if "response" in event)
    )
    assert len(advertised) == 1, advertised
    return advertised[0]


def _request_ids_for_key(gateway: Gateway, key: str, count: int) -> list[str]:
    rows: Final = eventually(
        lambda: gateway.rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "endTime"',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda found: len(found) >= count,
        seconds=70,
    )
    return [string_value(row["request_id"]) for row in rows]


def _logs_page_search(gateway: Gateway, client_id: str) -> list[str]:
    now: Final = datetime.now(timezone.utc)
    page: Final = gateway.get(
        "/spend/logs/ui",
        params={
            "search": client_id,
            "start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return [string_value(object_value(row)["request_id"]) for row in list_value(page["data"])]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("litellm_model", [_NATIVE_MODEL, _BRIDGED_MODEL], ids=["native", "bridge"])
def test_streamed_responses_spend_row_is_filed_under_the_id_the_client_saw(
    gateway: Gateway, litellm_model: str
) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=litellm_model, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        key: Final = scenario.key(models=[model])

        client_id: Final = _terminal_response_id(
            gateway, key, {"model": model, "input": f"marker-{uuid4().hex}", "stream": True}
        )

        assert client_id.startswith("resp_"), client_id
        assert _request_ids_for_key(gateway, key, 1) == [client_id]
        assert _logs_page_search(gateway, client_id) == [client_id]


def _prompt_storing_config(directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "store-prompts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "general_settings": {**object_value(stock["general_settings"]), "store_prompts_in_spend_logs": True},
            }
        )
    )
    return path


def test_a_follow_up_turn_on_a_streamed_bridged_response_replays_the_first_turn(
    gateway: Gateway, tmp_path: Path
) -> None:
    first_prompt: Final = f"first turn {uuid4().hex}"
    with (
        wire_server(_respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storing_config(tmp_path)) as storing,
        storing.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_BRIDGED_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        key: Final = scenario.key(models=[model])
        first_id: Final = _terminal_response_id(storing, key, {"model": model, "input": first_prompt, "stream": True})
        assert _request_ids_for_key(storing, key, 1) == [first_id]
        wire.drain()

        second_id: Final = _terminal_response_id(
            storing,
            key,
            {"model": model, "input": "second turn", "previous_response_id": first_id, "stream": True},
        )

        replayed: Final = _JSON_LIST.validate_python(JSON_OBJECT.validate_json(wire.drain()[-1].body)["messages"])
        assert [(message["role"], message["content"]) for message in replayed] == [
            ("user", first_prompt),
            ("assistant", _ANSWER),
            ("user", "second turn"),
        ], replayed
        assert _request_ids_for_key(storing, key, 2) == [first_id, second_id]
