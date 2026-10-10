import json
import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

import httpx
import openai
import pytest
from integration._support.bedrock_runtime_peer import NATIVE_RESPONSES, answer, respond, target_of
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.sigv4 import signature
from integration._support.wire import Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import JsonValue, TypeAdapter

GPT: Final = "us.openai.gpt-5.6-sol"
GLOBAL_GPT: Final = "global.openai.gpt-5.6-sol"
GPT_OSS: Final = "openai.gpt-oss-120b-1:0"
TOKEN: Final = "synthetic-bedrock-bearer"
ACCESS_KEY: Final = "AKIASYNTHETICKEY0001"
SECRET_KEY: Final = "synthetic-secret-key-for-testing"
PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4e5f6"
NATIVE_TARGET: Final = "/openai/v1/chat/completions"
CONVERSE_TARGET: Final = f"/model/{GPT}/converse"
GPT_DEPLOYMENT: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"model": f"bedrock/{GPT}", "api_key": TOKEN, "aws_region_name": "us-east-1"}
)
GUARDRAIL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"guardrailIdentifier": "gr-synthetic", "guardrailVersion": "1"}
)
TOOL_PARAMETERS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
)
TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "function",
        "function": {
            "name": "lookup_invoice",
            "description": "Look up an invoice",
            "parameters": dict(TOOL_PARAMETERS),
        },
    }
)
CONVERSE_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "toolSpec": {
            "inputSchema": {"json": dict(TOOL_PARAMETERS)},
            "name": "lookup_invoice",
            "description": "Look up an invoice",
        }
    }
)
JSON_SCHEMA: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "json_schema",
        "json_schema": {
            "name": "verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OBSERVATIONS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _prompt(marker: str) -> str:
    return f"synthetic native request marker-{marker}"


def _messages(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": _prompt(marker)}]


def _sdk_messages(marker: str) -> list[ChatCompletionMessageParam]:
    return [{"role": "user", "content": _prompt(marker)}]


def _converse_messages(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": [{"text": _prompt(marker)}]}]


def _native_body(model: str, marker: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": _messages(marker), "stream": False, **params}


def _streamed_native_body(model: str, marker: str) -> dict[str, JsonValue]:
    return _native_body(model, marker, stream=True, stream_options={"include_usage": True})


def _deployment(scenario: Scenario, wire: Wire, **overrides: JsonValue) -> str:
    return scenario.model(model_info=None, **{**GPT_DEPLOYMENT, "aws_bedrock_runtime_endpoint": wire.url, **overrides})


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _chat(gateway: Gateway, model: str, marker: str, **params: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": _messages(marker), "cache": {"no-cache": True}, **params},
    )


def _payload(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, target_of(request)) for request in received]
    return received[0]


def _body(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _native_request(wire: Wire) -> Request:
    request: Final = _only_request(wire)
    assert (request.method, target_of(request)) == ("POST", NATIVE_TARGET), request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}", dict(request.headers)
    return request


def _converse_request(wire: Wire, target: str = CONVERSE_TARGET) -> Request:
    request: Final = _only_request(wire)
    assert (request.method, target_of(request)) == ("POST", target), request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}", dict(request.headers)
    return request


def _spend_row(identity: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, status, prompt_tokens, completion_tokens, api_base FROM "LiteLLM_SpendLogs"'
            " WHERE request_id=%s",
            (identity,),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _only_spend_row_of(model: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, status, prompt_tokens, completion_tokens, api_base FROM "LiteLLM_SpendLogs"'
            " WHERE model_group=%s",
            (model,),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _success_row(model: str, api_base: str) -> dict[str, JsonValue]:
    return {"model_group": model, "status": "success", "prompt_tokens": 9, "completion_tokens": 5, "api_base": api_base}


def _delta_text(delta: ChoiceDelta, field: str) -> str:
    value: Final = delta.model_dump().get(field)
    return value if isinstance(value, str) else ""


def _chunk_text(chunk: ChatCompletionChunk, field: str) -> str:
    return "".join(_delta_text(choice.delta, field) for choice in chunk.choices)


def _joined(chunks: Sequence[ChatCompletionChunk], field: str) -> str:
    return "".join(_chunk_text(chunk, field) for chunk in chunks)


def _upstream_requests_mentioning(gateway: Gateway, marker: str) -> list[dict[str, JsonValue]]:
    observed: Final = httpx.get(f"{gateway.upstream_url}/__observations", trust_env=False, timeout=15)
    observed.raise_for_status()
    requests: Final = _OBSERVATIONS.validate_python(_JSON_OBJECT.validate_json(observed.content)["requests"])
    return [request for request in requests if marker in json.dumps(request["body"])]


def _authorization_field(part: str) -> tuple[str, str]:
    name, _, value = part.partition("=")
    return name, value


def _assert_sigv4_signed(request: Request, path: str) -> None:
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 "), dict(request.headers)
    fields: Final = dict(
        _authorization_field(part) for part in authorization.removeprefix("AWS4-HMAC-SHA256 ").split(", ")
    )
    access_key, scope = fields["Credential"].split("/", 1)
    assert access_key == ACCESS_KEY, authorization
    assert scope == f"{request.headers['x-amz-date'][:8]}/us-east-1/bedrock/aws4_request", authorization
    assert {"host", "x-amz-date"}.issubset(fields["SignedHeaders"].split(";")), authorization
    expected: Final = signature("POST", path, request.headers, fields["SignedHeaders"], request.body, SECRET_KEY, scope)
    assert fields["Signature"] == expected[1], authorization


def test_openai_sdk_reasoning_request_is_served_by_native_chat_completions(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        raw: Final = _openai_client(gateway).chat.completions.with_raw_response.create(
            model=model,
            messages=_sdk_messages(marker),
            reasoning_effort="high",
            max_tokens=16,
            extra_body={"cache": {"no-cache": True}},
        )
        completion: Final = raw.parse()
        assert completion.id == f"chatcmpl-{marker}", raw.text
        assert completion.choices[0].message.content == answer(marker), raw.text
        assert completion.usage is not None and completion.usage.model_dump(exclude_none=True) == {
            "prompt_tokens": 9,
            "completion_tokens": 5,
            "total_tokens": 14,
            "completion_tokens_details": {"reasoning_tokens": 3},
        }, raw.text
        assert raw.headers["llm_provider-x-amzn-requestid"] == marker, dict(raw.headers)
        request: Final = _native_request(wire)
        assert _body(request) == _native_body(GPT, marker, max_completion_tokens=16, reasoning_effort="high")
        assert _spend_row(completion.id) == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


async def test_async_openai_sdk_stream_keeps_the_upstream_id_and_usage(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"chatcmpl-{marker}"
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        stream: Final = await _async_openai_client(gateway).chat.completions.create(
            model=model,
            messages=_sdk_messages(marker),
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"cache": {"no-cache": True}},
        )
        chunks: Final = [chunk async for chunk in stream]
        assert {chunk.id for chunk in chunks} == {identity}, chunks
        assert _joined(chunks, "content") == answer(marker), chunks
        usage: Final = chunks[-1].usage
        assert usage is not None and (usage.prompt_tokens, usage.completion_tokens) == (9, 5), chunks[-1]
        assert usage.completion_tokens_details is not None and usage.completion_tokens_details.reasoning_tokens == 3
        assert all(chunk.usage is None for chunk in chunks[:-1]), chunks
        assert _body(_native_request(wire)) == _streamed_native_body(GPT, marker)
        assert _spend_row(identity) == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_temperature_is_forwarded_natively_when_reasoning_is_off(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, temperature=0.2, reasoning_effort="none")
        payload: Final = _payload(response)
        assert payload["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker, temperature=0.2, reasoning_effort="none")
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_temperature_while_reasoning_is_refused_before_any_wire_request(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, temperature=0.2, reasoning_effort="high")
        assert response.status_code == 400, response.text
        assert "UnsupportedParamsError" in response.text and "'temperature'" in response.text, response.text
        assert wire.drain() == (), response.text
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["model_group"], row["prompt_tokens"]) == ("failure", model, 0), row
        assert "while reasoning is active" in response.text, response.text


def test_drop_params_deployment_drops_temperature_while_reasoning(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, drop_params=True)
        response: Final = _chat(gateway, model, marker, temperature=0.2, reasoning_effort="high")
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker, reasoning_effort="high")
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_guardrail_config_keeps_converse(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, guardrailConfig=dict(GUARDRAIL))
        payload: Final = _payload(response)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"content": answer(marker), "role": "assistant"}}
        ], response.text
        assert response.headers["llm_provider-x-amzn-requestid"] == marker, dict(response.headers)
        body: Final = _body(_converse_request(wire))
        assert body["guardrailConfig"] == GUARDRAIL, body
        assert body["messages"] == [
            {"role": "user", "content": [{"guardContent": {"text": {"text": _prompt(marker)}}}]}
        ], body
        assert _spend_row(str(payload["id"])) == _success_row(model, f"{wire.url}{CONVERSE_TARGET}")


def test_converse_prefix_pins_the_model_to_converse(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, model=f"bedrock/converse/{GPT}")
        response: Final = _chat(gateway, model, marker, reasoning_effort="high")
        payload: Final = _payload(response)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"content": answer(marker), "role": "assistant"}}
        ], response.text
        body: Final = _body(_converse_request(wire))
        assert body["messages"] == _converse_messages(marker), body
        assert body["additionalModelRequestFields"] == {"reasoning": {"effort": "high"}}, body
        assert _spend_row(str(payload["id"])) == _success_row(model, f"{wire.url}{CONVERSE_TARGET}")


def test_application_inference_profile_arn_keeps_converse(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, model=f"bedrock/{PROFILE_ARN}")
        response: Final = _chat(gateway, model, marker)
        payload: Final = _payload(response)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"content": answer(marker), "role": "assistant"}}
        ], response.text
        request: Final = _converse_request(wire, f"/model/{PROFILE_ARN}/converse")
        assert request.target == f"/model/{quote(PROFILE_ARN, safe='')}/converse", request.target
        assert _body(request)["messages"] == _converse_messages(marker), request.body
        assert _spend_row(str(payload["id"])) == _success_row(
            model, f"{wire.url}/model/{quote(PROFILE_ARN, safe='')}/converse"
        )


def test_model_id_application_inference_profile_keeps_converse_at_the_profile_url(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, model_id=PROFILE_ARN)
        response: Final = _chat(gateway, model, marker)
        payload: Final = _payload(response)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"content": answer(marker), "role": "assistant"}}
        ], response.text
        request: Final = _converse_request(wire, f"/model/{PROFILE_ARN}/converse")
        assert request.target == f"/model/{quote(PROFILE_ARN, safe='')}/converse", request.target
        body: Final = _body(request)
        assert body["messages"] == _converse_messages(marker), request.body
        assert "model_id" not in body and "model" not in body, request.body
        assert _spend_row(str(payload["id"])) == _success_row(
            model, f"{wire.url}/model/{quote(PROFILE_ARN, safe='')}/converse"
        )


def test_stop_sequences_are_dropped_and_the_request_stays_native(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, stop=["END"])
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker)
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_json_object_response_format_keeps_converse(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, response_format={"type": "json_object"})
        payload: Final = _payload(response)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"content": answer(marker), "role": "assistant"}}
        ], response.text
        assert _body(_converse_request(wire))["messages"] == _converse_messages(marker), response.text
        assert _spend_row(str(payload["id"])) == _success_row(model, f"{wire.url}{CONVERSE_TARGET}")


def test_json_schema_response_format_is_forwarded_natively(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, response_format=dict(JSON_SCHEMA))
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker, response_format=dict(JSON_SCHEMA))
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_tools_while_reasoning_are_bridged_to_native_responses(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, tools=[dict(TOOL)], reasoning_effort="high")
        payload: Final = _payload(response)
        assert payload["id"] == f"resp_upstream_{marker}", response.text
        choices: Final = payload["choices"]
        assert isinstance(choices, list) and len(choices) == 1, response.text
        assert choices[0]["finish_reason"] == "stop", response.text
        assert choices[0]["message"]["content"] == answer(marker), response.text
        body: Final = _body(_converse_request(wire, NATIVE_RESPONSES))
        assert body["model"] == GPT, body
        assert body["reasoning"] == {"effort": "high"}, body
        assert [(tool["type"], tool["name"], tool["parameters"]) for tool in body["tools"]] == [
            ("function", "lookup_invoice", dict(TOOL_PARAMETERS))
        ], body
        assert _upstream_requests_mentioning(gateway, marker) == [], response.text
        assert _only_spend_row_of(model) == {
            **_success_row(model, f"{wire.url}{NATIVE_RESPONSES}"),
            "prompt_tokens": 30,
        }


def test_tools_while_reasoning_with_a_name_openai_rejects_keep_converse(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    tool: Final = {**TOOL, "function": {**TOOL["function"], "name": "lookup invoice!"}}
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, tools=[tool], reasoning_effort="high")
        payload: Final = _payload(response)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"content": answer(marker), "role": "assistant"}}
        ], response.text
        body: Final = _body(_converse_request(wire))
        assert body["toolConfig"] == {
            "tools": [{"toolSpec": {**CONVERSE_TOOL["toolSpec"], "name": "lookup_invoice_"}}]
        }, body
        assert body["additionalModelRequestFields"] == {"reasoning": {"effort": "high"}}, body
        assert _spend_row(str(payload["id"])) == _success_row(model, f"{wire.url}{CONVERSE_TARGET}")


def test_tools_with_reasoning_off_are_forwarded_natively(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, tools=[dict(TOOL)], reasoning_effort="none")
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker, tools=[dict(TOOL)], reasoning_effort="none")
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_empty_tools_list_while_reasoning_stays_native(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker, tools=[], reasoning_effort="high")
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker, tools=[], reasoning_effort="high")
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_chat_completions_prefix_splits_gpt_oss_reasoning_tag(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, model=f"bedrock/chat_completions/{GPT_OSS}")
        raw: Final = _openai_client(gateway).chat.completions.with_raw_response.create(
            model=model, messages=_sdk_messages(marker), extra_body={"cache": {"no-cache": True}}
        )
        completion: Final = raw.parse()
        assert completion.id == f"chatcmpl-{marker}", raw.text
        message: Final = completion.choices[0].message
        assert message.content == answer(marker), raw.text
        assert (message.model_extra or {}).get("reasoning_content") == f"why marker-{marker}", raw.text
        assert _body(_native_request(wire)) == _native_body(GPT_OSS, marker)
        assert _spend_row(completion.id) == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_chat_completions_prefix_splits_gpt_oss_reasoning_tag_across_stream_deltas(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"chatcmpl-{marker}"
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, model=f"bedrock/chat_completions/{GPT_OSS}")
        stream: Final = _openai_client(gateway).chat.completions.create(
            model=model,
            messages=_sdk_messages(marker),
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"cache": {"no-cache": True}},
        )
        chunks: Final = list(stream)
        assert {chunk.id for chunk in chunks} == {identity}, chunks
        assert _joined(chunks, "reasoning_content") == f"why marker-{marker}", chunks
        assert _joined(chunks, "content") == answer(marker), chunks
        assert _body(_native_request(wire)) == _streamed_native_body(GPT_OSS, marker)
        assert _spend_row(identity) == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_region_path_model_is_served_natively_without_the_region(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/us-west-2/{GLOBAL_GPT}", api_key=TOKEN, aws_bedrock_runtime_endpoint=wire.url
        )
        response: Final = _chat(gateway, model, marker)
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GLOBAL_GPT, marker)
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_sigv4_deployment_signs_the_native_request(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{GPT}",
            api_key=None,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = _chat(gateway, model, marker)
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        request: Final = _only_request(wire)
        assert (request.method, target_of(request)) == ("POST", NATIVE_TARGET), request.target
        _assert_sigv4_signed(request, NATIVE_TARGET)
        assert _body(request) == _native_body(GPT, marker)
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_blank_api_key_on_a_sigv4_deployment_is_signed_not_sent_as_an_empty_bearer(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{GPT}",
            api_key="",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = _chat(gateway, model, marker)
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        request: Final = _only_request(wire)
        assert (request.method, target_of(request)) == ("POST", NATIVE_TARGET), request.target
        _assert_sigv4_signed(request, NATIVE_TARGET)
        assert _body(request) == _native_body(GPT, marker)
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_runtime_endpoint_without_api_base_is_used_natively(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, api_base=None)
        response: Final = _chat(gateway, model, marker)
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker)
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


def test_runtime_endpoint_wins_over_an_unrelated_api_base(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = _chat(gateway, model, marker)
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(_native_request(wire)) == _native_body(GPT, marker)
        assert _upstream_requests_mentioning(gateway, marker) == [], response.text
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")


@pytest.mark.parametrize("suffix", ["/openai/v1", "/openai/v1/chat/completions"])
def test_api_base_already_naming_the_native_path_is_not_doubled(gateway: Gateway, suffix: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model_info=None, **{**GPT_DEPLOYMENT, "api_base": f"{wire.url}{suffix}"})
        response: Final = _chat(gateway, model, marker)
        request: Final = _only_request(wire)
        assert (request.method, request.target) == ("POST", NATIVE_TARGET), response.text
        assert _payload(response)["id"] == f"chatcmpl-{marker}", response.text
        assert _body(request) == _native_body(GPT, marker)
        assert _spend_row(f"chatcmpl-{marker}") == _success_row(model, f"{wire.url}{NATIVE_TARGET}")
