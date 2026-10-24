import json
from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from integration._support.client import Scenario, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_COUNT: Final = 4242
_TOOL_SCHEMA: Final[dict[str, JsonValue]] = {"type": "object", "properties": {}}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@pytest.fixture(scope="module")
def counting_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, tmp_path_factory.mktemp("openai-count-tools"), {}, workers=1) as owned,
    ):
        yield owned


def _deployment(scenario: Scenario, api_base: str) -> str:
    return scenario.model(
        model="openai/gpt-4o-mini",
        api_base=f"{api_base}/v1",
        api_key="sk-integration-openai",
    )


def _count_response(_request: Request) -> Reply:
    return Reply(body=json.dumps({"input_tokens": _COUNT}).encode())


_PDF_BASE64: Final = "JVBERi0xLjQK"
_PNG_BASE64: Final = "aW1hZ2U="
_LOOKUP_TOOL: Final[dict[str, JsonValue]] = {
    "name": "lookup",
    "description": "Find a record",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
_RESPONSES_REPLY: Final[Mapping[str, JsonValue]] = {
    "id": "resp_count_parity",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_count_parity",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "OK", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
}
_WRAPPED_REASONING: Final = "litellm_enc:bW9kZWxfaWQ6ZGVwbG95bWVudC0x;gAAAAoriginal"
_ANSWER_SCHEMA: Final[dict[str, JsonValue]] = {"type": "object", "properties": {"answer": {"type": "string"}}}
_ANTHROPIC_BODIES: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    "structured-output": {
        "output_config": {"format": {"type": "json_schema", "schema": _ANSWER_SCHEMA}},
        "messages": [{"role": "user", "content": "Answer in JSON."}],
    },
    "affinity-wrapped-reasoning": {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": f"litellm_encrypted_reasoning:{_WRAPPED_REASONING}"},
                    {"type": "text", "text": "OK"},
                ],
            },
            {"role": "user", "content": "again"},
        ],
    },
    "base64-pdf": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the attached contract."},
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": _PDF_BASE64},
                    },
                ],
            }
        ],
    },
    "base64-image": {
        "system": [{"type": "text", "text": "Describe images briefly."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64}},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ],
    },
    "replayed-web-search": {
        "messages": [
            {"role": "user", "content": "What is the newest LiteLLM release?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {"query": "newest LiteLLM release"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://docs.litellm.ai/release_notes",
                                "title": "Release notes",
                                "encrypted_content": "",
                                "snippet": "LiteLLM ships a weekly release candidate and a stable line.",
                            }
                        ],
                    },
                    {"type": "text", "text": "The newest release is in the release notes."},
                ],
            },
            {"role": "user", "content": "Thanks."},
        ],
    },
    "tool-turns": {
        "system": "Use the lookup tool.",
        "tools": [_LOOKUP_TOOL],
        "messages": [
            {"role": "user", "content": "Find record 14."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking it up."},
                    {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {"query": "14"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "14. Indemnification."}],
            },
        ],
    },
}


def _responses_peer(request: Request) -> Reply:
    if request.target == "/v1/responses/input_tokens":
        return _count_response(request)
    if request.target == "/v1/responses":
        return Reply(body=json.dumps(_RESPONSES_REPLY).encode())
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _body_for(wire_requests: tuple[Request, ...], target: str) -> dict[str, JsonValue]:
    matching: Final = tuple(request for request in wire_requests if request.target == target)
    assert len(matching) == 1, [request.target for request in wire_requests]
    return _JSON_OBJECT.validate_json(matching[0].body)


def _counted_fields(body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {key: body[key] for key in ("input", "instructions", "tools", "text") if body.get(key) is not None}


def _count_body(wire: Wire) -> dict[str, JsonValue]:
    requests: Final = wire.drain()
    assert len(requests) == 1, requests
    request: Final = requests[0]
    assert request.method == "POST"
    assert request.target == "/v1/responses/input_tokens", request.target
    return _JSON_OBJECT.validate_json(request.body)


def test_openai_count_tokens_uses_deployment_api_base(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": model, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNT
        assert "tools" not in _count_body(wire)


def test_google_count_tokens_maps_anthropic_tools_to_openai(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:countTokens",
            {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "lookup",
                                "description": "Find a record",
                                "parameters": _TOOL_SCHEMA,
                            }
                        ]
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["totalTokens"] == _COUNT
        assert _count_body(wire)["tools"] == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Find a record",
                "parameters": _TOOL_SCHEMA,
                "strict": False,
            }
        ]


def test_messages_count_tokens_maps_anthropic_tools_to_openai(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Find a record",
                        "input_schema": _TOOL_SCHEMA,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNT
        assert _count_body(wire)["tools"] == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Find a record",
                "parameters": _TOOL_SCHEMA,
                "strict": False,
            }
        ]


def test_utils_token_counter_keeps_chat_tool_conversion(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/utils/token_counter",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Find a record",
                            "parameters": _TOOL_SCHEMA,
                        },
                    }
                ],
            },
            params={"call_endpoint": "true"},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["total_tokens"] == _COUNT
        assert _count_body(wire)["tools"] == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Find a record",
                "parameters": _TOOL_SCHEMA,
            }
        ]


@pytest.mark.parametrize("case", sorted(_ANTHROPIC_BODIES))
def test_messages_count_tokens_sends_the_input_v1_messages_inference_sends(
    counting_proxy: OwnedProxy, case: str
) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_responses_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        body: Final = {"model": model, **_ANTHROPIC_BODIES[case]}
        generated: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 16})
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body)
        assert generated.status_code == 200, generated.text
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _COUNT
        requests: Final = wire.drain()
        inference_body: Final = _body_for(requests, "/v1/responses")
        count_body: Final = _body_for(requests, "/v1/responses/input_tokens")
        assert _counted_fields(count_body) == _counted_fields(inference_body)
        assert count_body["model"] == inference_body["model"]


_UNCONVERTIBLE_BODY: Final[Mapping[str, JsonValue]] = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize the notes."},
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "notes"}},
                {"type": "search_result", "source": "https://example.com/notes", "title": "Notes", "content": []},
            ],
        },
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "search", "input": {}}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": [
                        {
                            "type": "search_result",
                            "source": "https://example.com/hits",
                            "title": "Hits",
                            "content": [{"type": "text", "text": "hit"}],
                        }
                    ],
                }
            ],
        },
    ],
}


def test_messages_count_tokens_rejects_blocks_it_cannot_convert(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_responses_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/messages/count_tokens", {"model": model, **_UNCONVERTIBLE_BODY})
        assert response.status_code == 400, response.text
        assert "document with a text source, search_result, search_result inside a tool_result" in response.text
        assert wire.drain() == ()


def test_messages_count_tokens_without_a_role_answers_400_before_calling_openai(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": model, "messages": [{"role": "user", "content": "hello"}, {"content": "no role"}]},
        )
        assert response.status_code == 400, response.text
        assert "messages.1.role: Field required" in response.text
        assert wire.drain() == ()


def test_google_count_tokens_sends_the_system_instruction_text(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:countTokens",
            {
                "systemInstruction": {"parts": [{"text": "Be brief."}, {"text": "Answer in English."}]},
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            },
        )
        assert response.status_code == 200, response.text
        assert _count_body(wire)["instructions"] == "Be brief.\nAnswer in English."


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            {"message": "Invalid input", "type": "invalid_request_error", "param": "input", "code": "invalid_value"},
            id="full",
        ),
        pytest.param({"message": "Invalid input", "type": "invalid_request_error"}, id="bare"),
    ],
)
def test_responses_input_tokens_passes_an_openai_count_error_through(
    counting_proxy: OwnedProxy, error: dict[str, JsonValue]
) -> None:
    def rejecting(_request: Request) -> Reply:
        return Reply(status=400, body=json.dumps({"error": error}).encode())

    gateway: Final = counting_proxy.gateway
    with wire_server(rejecting) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/responses/input_tokens", {"model": model, "input": "hello"})
        assert response.status_code == 400, response.text
        assert response.json() == {"error": {"param": None, "code": None, **error}}, response.text
        assert _count_body(wire)["input"] == "hello"


_NATIVE_TEXT: Final[dict[str, JsonValue]] = {
    "format": {"type": "json_schema", "name": "answer", "schema": _ANSWER_SCHEMA},
}
_NATIVE_BODY: Final[Mapping[str, JsonValue]] = {
    "input": [
        {"role": "user", "content": "hi"},
        {"type": "reasoning", "summary": [], "encrypted_content": _WRAPPED_REASONING},
        {"role": "user", "content": "again"},
    ],
    "text": _NATIVE_TEXT,
    "max_output_tokens": "lots",
    "metadata": {"run": "nightly"},
}


@pytest.mark.parametrize(
    ("drop_params", "expected_extra"),
    [pytest.param(False, {"max_output_tokens": "lots"}, id="as-sent"), pytest.param(True, {}, id="drop-params")],
)
def test_responses_input_tokens_sends_the_callers_body_to_openai(
    counting_proxy: OwnedProxy, drop_params: bool, expected_extra: dict[str, JsonValue]
) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=f"{wire.url}/v1",
            api_key="sk-integration-openai",
            drop_params=drop_params,
        )
        response: Final = gateway.request("POST", "/v1/responses/input_tokens", {"model": model, **_NATIVE_BODY})
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNT
        assert _count_body(wire) == {
            "model": "gpt-4o-mini",
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "reasoning", "summary": [], "encrypted_content": "gAAAAoriginal"},
                {"role": "user", "content": "again"},
            ],
            "text": _NATIVE_TEXT,
            **expected_extra,
        }


def test_responses_input_tokens_sends_openai_the_owners_previous_response_id(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_responses_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        team: Final = scenario.team(models=[model])
        owner_key: Final = scenario.key(team_id=team, user_id=scenario.member(team), models=[model])
        other_key: Final = scenario.key(team_id=team, user_id=scenario.member(team), models=[model])
        generated: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": "hi"}, key=owner_key)
        assert generated.status_code == 200, generated.text
        follow_up: Final = {
            "model": model,
            "input": "again",
            "previous_response_id": _JSON_OBJECT.validate_json(generated.content)["id"],
        }
        _body_for(wire.drain(), "/v1/responses")

        refused: Final = gateway.request("POST", "/v1/responses/input_tokens", follow_up, key=other_key)
        assert refused.status_code == 403, refused.text
        assert _JSON_OBJECT.validate_json(refused.content)["error"]["type"] == "permission_error", refused.text
        assert wire.drain() == ()

        counted: Final = gateway.request("POST", "/v1/responses/input_tokens", follow_up, key=owner_key)
        assert counted.status_code == 200, counted.text
        count_body: Final = _body_for(wire.drain(), "/v1/responses/input_tokens")
        assert count_body["previous_response_id"] == _RESPONSES_REPLY["id"]
