import base64
import uuid
from dataclasses import dataclass
from typing import Final

import openai
from integration._support.bedrock_runtime_peer import NATIVE_RESPONSES, answer, respond, target_of
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Request, Wire, wire_server
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

GPT: Final = "us.openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-bedrock-bearer"
SALT: Final = "sk-integration-salt"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


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
def _spend_row(client_id: str, issued_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
            "WHERE request_id = ANY(%s)",
            ([client_id, issued_id],),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
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
        assert _spend_row(completed.response.id, issued.issued) == _success_row(model)
