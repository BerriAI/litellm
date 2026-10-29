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


def _native_completed(upstream_id: str) -> dict[str, JsonValue]:
    message: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}],
    }
    return response_object(upstream_id, "completed", output=[message], usage=_USAGE)


def _native_stream(upstream_id: str) -> Reply:
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
            frame({"type": "response.completed", "sequence_number": 2, "response": _native_completed(upstream_id)}),
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
    if request.method == "GET":
        stored_id: Final = request.target.removeprefix("/v1/responses/")
        assert stored_id.startswith("resp_upstream-"), request.target
        return Reply(body=json.dumps(_native_completed(stored_id)).encode())
    body: Final = JSON_OBJECT.validate_json(request.body)
    upstream_id: Final = f"upstream-{uuid4().hex}"
    if request.target == "/v1/responses" and body.get("stream") is not True:
        return Reply(body=json.dumps(_native_completed(f"resp_{upstream_id}")).encode())
    assert body.get("stream") is True, body
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


def _created_response_id(gateway: Gateway, key: str, body: dict[str, JsonValue]) -> str:
    if body.get("stream") is True:
        return _terminal_response_id(gateway, key, body)
    return string_value(gateway.post("/v1/responses", body, key=key)["id"])


def _spend_logs_lookup(gateway: Gateway, request_id: str) -> list[JsonValue]:
    response: Final = gateway.request("GET", "/spend/logs", params={"request_id": request_id})
    assert response.status_code == 200, response.text
    return [row["request_id"] for row in _JSON_LIST.validate_json(response.content)]


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
        assert _spend_logs_lookup(gateway, client_id) == [client_id]


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


@pytest.mark.timeout(180)
@pytest.mark.parametrize("stream_first_turn", [True, False], ids=["streamed-first-turn", "non-streamed-first-turn"])
def test_a_follow_up_through_the_id_a_retrieve_returned_replays_the_first_turn_once(
    gateway: Gateway, tmp_path: Path, stream_first_turn: bool
) -> None:
    first_prompt: Final = f"first turn {uuid4().hex}"
    with (
        wire_server(_respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storing_config(tmp_path)) as storing,
        storing.scenario() as scenario,
    ):
        native: Final = scenario.model(model=_NATIVE_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        bridged: Final = scenario.model(model=_BRIDGED_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        key: Final = scenario.key(models=[native, bridged])
        first_id: Final = _created_response_id(
            storing, key, {"model": native, "input": first_prompt, "stream": stream_first_turn}
        )

        retrieved: Final = storing.request("GET", f"/v1/responses/{first_id}", key=key)

        assert retrieved.status_code == 200, retrieved.text
        retrieved_id: Final = string_value(JSON_OBJECT.validate_json(retrieved.content)["id"])
        assert retrieved_id == first_id
        read_row_id: Final = f"{first_id}_retrieve_{retrieved.headers['x-litellm-call-id']}"
        assert sorted(_request_ids_for_key(storing, key, 2)) == sorted([first_id, read_row_id])
        assert _spend_logs_lookup(storing, first_id) == [first_id]
        wire.drain()

        _terminal_response_id(
            storing,
            key,
            {"model": bridged, "input": "second turn", "previous_response_id": retrieved_id, "stream": True},
        )

        replayed: Final = _JSON_LIST.validate_python(JSON_OBJECT.validate_json(wire.drain()[-1].body)["messages"])
        assert [(message["role"], message["content"]) for message in replayed] == [
            ("user", first_prompt),
            ("assistant", _ANSWER),
            ("user", "second turn"),
        ], replayed
