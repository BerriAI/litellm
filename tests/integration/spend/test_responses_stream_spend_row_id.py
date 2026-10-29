import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final
from uuid import uuid4

import httpx
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
    if request.target.endswith("/cancel"):
        cancelled_id: Final = request.target.removeprefix("/v1/responses/").removesuffix("/cancel")
        assert cancelled_id.startswith("resp_upstream-"), request.target
        return Reply(body=json.dumps(response_object(cancelled_id, "cancelled")).encode())
    body: Final = JSON_OBJECT.validate_json(request.body)
    upstream_id: Final = f"upstream-{uuid4().hex}"
    if request.target == "/v1/responses" and body.get("stream") is not True:
        return Reply(body=json.dumps(_native_completed(f"resp_{upstream_id}")).encode())
    assert body.get("stream") is True, body
    if request.target == "/v1/responses":
        return _native_stream(f"resp_{upstream_id}")
    assert request.target == "/v1/chat/completions", request.target
    return _chat_stream(f"chatcmpl-{upstream_id}", string_value(body["model"]))


def _terminal_response_id(
    gateway: Gateway, key: str, body: dict[str, JsonValue], headers: Mapping[str, str] | None = None
) -> str:
    response: Final = gateway.request("POST", "/v1/responses", body, key=key, headers=headers)
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


def _created_response_id(
    gateway: Gateway, key: str, body: dict[str, JsonValue], headers: Mapping[str, str] | None = None
) -> str:
    if body.get("stream") is True:
        return _terminal_response_id(gateway, key, body, headers)
    response: Final = gateway.request("POST", "/v1/responses", body, key=key, headers=headers)
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def _read_and_cancel(
    gateway: Gateway, key: str, response_id: str, headers: Mapping[str, str] | None = None
) -> tuple[httpx.Response, ...]:
    responses: Final = tuple(
        gateway.request(method, path, key=key, headers=headers)
        for method, path in (
            ("GET", f"/v1/responses/{response_id}"),
            ("GET", f"/v1/responses/{response_id}"),
            ("POST", f"/v1/responses/{response_id}/cancel"),
        )
    )
    assert [response.status_code for response in responses] == [200, 200, 200], [r.text for r in responses]
    return responses


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


def _proxy_config(
    directory: Path,
    general_settings: Mapping[str, JsonValue],
    litellm_settings: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "responses-spend.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "general_settings": {**object_value(stock["general_settings"]), **general_settings},
                "litellm_settings": {**object_value(stock["litellm_settings"]), **litellm_settings},
            }
        )
    )
    return path


def _prompt_storing_config(directory: Path) -> Path:
    return _proxy_config(directory, {"store_prompts_in_spend_logs": True})


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
        stored: Final = _request_ids_for_key(storing, key, 2)
        assert stored.count(first_id) == 1, stored
        assert len(set(stored)) == 2, stored
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


def _replayed_turns(wire_body: bytes) -> list[tuple[JsonValue, JsonValue]]:
    replayed: Final = _JSON_LIST.validate_python(JSON_OBJECT.validate_json(wire_body)["messages"])
    return [(message["role"], message["content"]) for message in replayed]


@pytest.mark.timeout(180)
@pytest.mark.parametrize("reader", ["proxy-admin", "master-key"])
def test_a_reader_other_than_the_creator_continues_through_the_id_its_retrieve_returned(
    gateway: Gateway, tmp_path: Path, reader: str
) -> None:
    first_prompt: Final = f"first turn {uuid4().hex}"
    with (
        wire_server(_respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storing_config(tmp_path)) as storing,
        storing.scenario() as scenario,
    ):
        native: Final = scenario.model(model=_NATIVE_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        bridged: Final = scenario.model(model=_BRIDGED_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        creator: Final = scenario.key(user_id=scenario.user(), models=[native, bridged])
        reader_key: Final = (
            scenario.key(user_id=scenario.user(user_role="proxy_admin"), models=[native, bridged])
            if reader == "proxy-admin"
            else storing.key
        )
        first_id: Final = _terminal_response_id(
            storing, creator, {"model": native, "input": first_prompt, "stream": True}
        )
        assert _request_ids_for_key(storing, creator, 1) == [first_id]

        retrieved: Final = storing.request("GET", f"/v1/responses/{first_id}", key=reader_key)

        assert retrieved.status_code == 200, retrieved.text
        retrieved_id: Final = string_value(JSON_OBJECT.validate_json(retrieved.content)["id"])
        wire.drain()
        _terminal_response_id(
            storing,
            reader_key,
            {"model": bridged, "input": "second turn", "previous_response_id": retrieved_id, "stream": True},
        )
        assert _replayed_turns(wire.drain()[-1].body) == [
            ("user", first_prompt),
            ("assistant", _ANSWER),
            ("user", "second turn"),
        ]
        assert retrieved_id == first_id


@pytest.mark.timeout(180)
def test_reads_and_cancels_sent_with_a_session_header_add_no_replayed_turns(gateway: Gateway, tmp_path: Path) -> None:
    first_prompt: Final = f"first turn {uuid4().hex}"
    session: Final = {"x-litellm-session-id": f"session-{uuid4().hex}"}
    with (
        wire_server(_respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storing_config(tmp_path)) as storing,
        storing.scenario() as scenario,
    ):
        native: Final = scenario.model(model=_NATIVE_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        bridged: Final = scenario.model(model=_BRIDGED_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        key: Final = scenario.key(models=[native, bridged])
        first_id: Final = _created_response_id(storing, key, {"model": native, "input": first_prompt}, session)

        _read_and_cancel(storing, key, first_id, session)

        assert len(_request_ids_for_key(storing, key, 4)) == 4
        wire.drain()
        _terminal_response_id(
            storing,
            key,
            {"model": bridged, "input": "second turn", "previous_response_id": first_id, "stream": True},
            session,
        )
        assert _replayed_turns(wire.drain()[-1].body) == [
            ("user", first_prompt),
            ("assistant", _ANSWER),
            ("user", "second turn"),
        ]


@pytest.mark.timeout(180)
def test_each_read_and_cancel_reaches_callbacks_and_spend_logs_under_an_id_of_its_own(
    gateway: Gateway, tmp_path: Path
) -> None:
    logged_call_types: Final = frozenset({"aresponses", "aget_responses", "acancel_responses"})
    with (
        wire_server(_respond) as wire,
        wire_server(lambda _: Reply()) as sink,
        owned_proxy(
            gateway,
            tmp_path,
            {"GENERIC_LOGGER_ENDPOINT": sink.url},
            config=_proxy_config(tmp_path, {}, {"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1}),
        ) as logging_proxy,
        logging_proxy.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_NATIVE_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        key: Final = scenario.key(models=[model])
        first_id: Final = _created_response_id(logging_proxy, key, {"model": model, "input": f"marker-{uuid4().hex}"})

        reads: Final = _read_and_cancel(logging_proxy, key, first_id)

        assert [string_value(JSON_OBJECT.validate_json(read.content)["id"]) for read in reads] == [first_id] * 3
        batches: Final[list[bytes]] = []  # mutable-ok: drain() consumes, later polls keep earlier batches

        def delivered() -> tuple[tuple[str, str], ...]:
            batches.extend(batch.body for batch in sink.drain())
            return tuple(
                (string_value(event["call_type"]), string_value(event["id"]))
                for event in (object_value(item) for batch in batches for item in _JSON_LIST.validate_json(batch))
                if event["call_type"] in logged_call_types
            )

        events: Final = eventually(delivered, lambda found: len(found) >= 4, seconds=30)
        read_ids: Final = [logged_id for call_type, logged_id in events if call_type != "aresponses"]
        assert [logged_id for call_type, logged_id in events if call_type == "aresponses"] == [first_id], events
        assert len(read_ids) == 3, events
        assert first_id not in read_ids, events
        assert len(set(read_ids)) == 3, events
        assert sorted(_request_ids_for_key(logging_proxy, key, 4)) == sorted([first_id, *read_ids])


@pytest.mark.timeout(240)
def test_with_id_security_off_reads_and_cancels_add_no_rows_and_no_replayed_turns(
    gateway: Gateway, tmp_path: Path
) -> None:
    first_prompt: Final = f"first turn {uuid4().hex}"
    session: Final = {"x-litellm-session-id": f"session-{uuid4().hex}"}
    with (
        wire_server(_respond) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=_proxy_config(
                tmp_path, {"store_prompts_in_spend_logs": True, "disable_responses_id_security": True}
            ),
        ) as insecure,
        insecure.scenario() as scenario,
    ):
        native: Final = scenario.model(model=_NATIVE_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        bridged: Final = scenario.model(model=_BRIDGED_MODEL, api_base=f"{wire.url}/v1", api_key="synthetic-key")
        key: Final = scenario.key(models=[native, bridged])
        first_id: Final = _created_response_id(insecure, key, {"model": native, "input": first_prompt}, session)

        reads: Final = _read_and_cancel(insecure, key, first_id, session)

        assert [string_value(JSON_OBJECT.validate_json(read.content)["id"]) for read in reads] == [first_id] * 3
        later_id: Final = _created_response_id(insecure, key, {"model": native, "input": "a later request"})
        assert _request_ids_for_key(insecure, key, 2) == [first_id, later_id]
        wire.drain()

        _terminal_response_id(
            insecure,
            key,
            {"model": bridged, "input": "second turn", "previous_response_id": first_id, "stream": True},
            session,
        )

        replayed: Final = _JSON_LIST.validate_python(JSON_OBJECT.validate_json(wire.drain()[-1].body)["messages"])
        assert [(message["role"], message["content"]) for message in replayed] == [
            ("user", first_prompt),
            ("assistant", _ANSWER),
            ("user", "second turn"),
        ], replayed
