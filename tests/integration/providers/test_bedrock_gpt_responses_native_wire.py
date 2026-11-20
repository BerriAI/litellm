import base64
import json
import uuid
from dataclasses import dataclass
from typing import Final

import openai
from integration._support.bedrock_runtime_peer import NATIVE_RESPONSES, answer, marker_of, respond, target_of
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseCompletedEvent, ResponseFailedEvent, ResponseTextDeltaEvent
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

GPT: Final = "us.openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-bedrock-bearer"
SALT: Final = "sk-integration-salt"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
UPSTREAM_OUTAGE: Final = "The server had an error while processing your request. Sorry about that!"


@dataclass(frozen=True, slots=True)
class _IssuedId:
    issued: str
    upstream: str


def _prompt(marker: str) -> str:
    return f"synthetic responses request marker-{marker}"


def _deployment(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"bedrock/{GPT}",
        api_key=TOKEN,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=wire.url,
        api_base=None,
    )


def _issued_id(client_id: str) -> _IssuedId:
    decrypted: Final = decrypt_if_encrypted_with(client_id.removeprefix("resp_"), SALT)
    assert decrypted is not None, client_id
    issued: Final = decrypted.split(";")[0].split("response_id:")[-1]
    decoded: Final = base64.b64decode(issued.removeprefix("resp_")).decode()
    return _IssuedId(issued, decoded.split(";")[-1].removeprefix("response_id:"))


def _native_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert [(request.method, target_of(request)) for request in received] == [("POST", NATIVE_RESPONSES)], received
    assert received[0].headers["authorization"] == f"Bearer {TOKEN}", dict(received[0].headers)
    return received[0]


def _body(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


# TODO: a Bedrock non-stream /v1/responses spend row can carry the pre-encryption resp_<base64> id instead of the
# ciphertext the caller received, because the spend row id is read from response_obj["id"] before the
# ResponsesIDSecurity hook rewrites it in place; the row is looked up under both ids until that ordering is fixed on
# main
def _spend_row(*request_ids: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
            "WHERE request_id = ANY(%s)",
            (list(request_ids),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _success_row(model: str) -> dict[str, JsonValue]:
    return {"model_group": model, "status": "success", "prompt_tokens": 30, "completion_tokens": 5}


def test_openai_sdk_responses_request_is_served_by_the_native_responses_route(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        raw: Final = client.responses.with_raw_response.create(
            model=model, input=_prompt(marker), extra_body={"cache": {"no-cache": True}}
        )
        response: Final = raw.parse()
        assert response.output_text == answer(marker), raw.text
        assert response.usage is not None and (response.usage.input_tokens, response.usage.output_tokens) == (30, 5)
        issued: Final = _issued_id(response.id)
        assert issued.upstream == f"resp_upstream_{marker}", response.id
        request: Final = _native_request(wire)
        assert _body(request) == {"model": GPT, "input": _prompt(marker)}, request.body
        assert _spend_row(response.id, issued.issued) == _success_row(model)


async def test_async_openai_sdk_responses_stream_is_served_by_the_native_responses_route(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        stream: Final = await client.responses.create(
            model=model, input=_prompt(marker), stream=True, extra_body={"cache": {"no-cache": True}}
        )
        events: Final = [event async for event in stream]
        assert [event.type for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ], events
        deltas: Final = "".join(event.delta for event in events if isinstance(event, ResponseTextDeltaEvent))
        assert deltas == answer(marker), events
        completed: Final = events[-1]
        assert isinstance(completed, ResponseCompletedEvent), completed
        assert completed.response.output_text == answer(marker), completed
        issued: Final = _issued_id(completed.response.id)
        assert issued.upstream == f"resp_upstream_{marker}", completed.response.id
        request: Final = _native_request(wire)
        assert _body(request) == {"model": GPT, "input": _prompt(marker), "stream": True}, request.body
        assert _spend_row(completed.response.id) == _success_row(model)


def _stream_then_bare_error(request: Request) -> Reply:
    marker: Final = marker_of(request)
    created: Final[dict[str, JsonValue]] = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": f"resp_upstream_{marker}",
            "object": "response",
            "created_at": 1,
            "status": "in_progress",
            "model": GPT,
            "output": [],
        },
    }
    delta: Final[dict[str, JsonValue]] = {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": f"msg_{marker}",
        "output_index": 0,
        "content_index": 0,
        "delta": answer(marker),
    }
    outage: Final[dict[str, JsonValue]] = {
        "error": {"message": UPSTREAM_OUTAGE, "type": "server_error", "code": "service_unavailable"}
    }
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(event)}\n\n".encode() for event in (created, delta, outage)),
    )


async def test_mid_stream_upstream_error_without_an_event_type_reaches_the_client_as_response_failed(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_stream_then_bare_error) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        stream: Final = await client.responses.create(
            model=model, input=_prompt(marker), stream=True, extra_body={"cache": {"no-cache": True}}
        )
        events: Final = [event async for event in stream]
        assert [event.type for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.failed",
        ], events
        failed: Final = events[-1]
        assert isinstance(failed, ResponseFailedEvent), failed
        assert failed.response.error is not None and UPSTREAM_OUTAGE in failed.response.error.message, failed
        _native_request(wire)


def test_codex_agent_message_with_encrypted_content_reaches_the_native_responses_route_intact(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    agent_message: Final[JsonValue] = {
        "type": "agent_message",
        "author": "/root/reviewer",
        "recipient": "/root",
        "content": [
            {"type": "input_text", "text": "Worker message:\n"},
            {"type": "encrypted_content", "encrypted_content": f"gAAAAABsynthetic-agent-ciphertext-{marker}=="},
        ],
    }
    user_turn: Final[JsonValue] = {"type": "message", "role": "user", "content": _prompt(marker)}
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": [agent_message, user_turn], "cache": {"no-cache": True}},
        )
        assert response.status_code == 200, response.text
        assert _body(_native_request(wire))["input"] == [agent_message, user_turn]
