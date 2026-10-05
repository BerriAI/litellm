import base64
import json
import time
import uuid
from typing import Final

import anthropic
import pytest
from anthropic.types import Message
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGE: Final = TypeAdapter(Message)


@pytest.mark.covers(
    "other.provider_wire.anthropic.tool_history_system_cache_and_internal_fields",
    "quota_management.spend_tracking.cache_tokens.disjoint_classes_use_explicit_rates",
)
def test_anthropic_tool_history_and_cache_tokens_keep_wire_and_accounting_contracts(gateway: Gateway) -> None:
    identity: Final = "anthropic-wire-" + uuid.uuid4().hex
    tool_schema: Final = {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        "required": ["x", "y"],
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-sonnet-4-5-20250929"
        assert body["system"] == [{"type": "text", "text": "synthetic policy", "cache_control": {"type": "ephemeral"}}]
        assert body["tools"][0]["name"] == "add" and body["tools"][0]["input_schema"] == tool_schema
        assert body["max_tokens"] == 16
        assert not {"timeout", "stream_chunk_size", "litellm_params", "litellm_metadata", "rpm", "tpm"}.intersection(
            body
        )
        messages: Final = body["messages"]
        assert [message["role"] for message in messages] == ["user", "assistant", "user"]
        assert messages[0]["content"] == [{"type": "text", "text": "first"}]
        assert messages[1]["content"] == [
            {"type": "tool_use", "id": "history-call", "name": "add", "input": {"x": 1, "y": 2}}
        ]
        assert messages[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "history-call", "content": "3"},
            {"type": "text", "text": "next"},
        ]
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "tool_use", "id": "next-call", "name": "add", "input": {"x": 3, "y": 4}}],
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 7,
                    },
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            cache_read_input_token_cost=0.0001,
            cache_creation_input_token_cost=0.002,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 16,
                "timeout": 5,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "synthetic policy", "cache_control": {"type": "ephemeral"}}
                        ],
                    },
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "history-call",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"x":1,"y":2}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "history-call", "content": "3"},
                    {"role": "user", "content": "next"},
                ],
                "tools": [{"type": "function", "function": {"name": "add", "parameters": tool_schema}}],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["id"].startswith("chatcmpl-")
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        tool: Final = body["choices"][0]["message"]["tool_calls"][0]
        assert tool["id"] == "next-call" and tool["function"]["name"] == "add"
        assert json.loads(tool["function"]["arguments"]) == {"x": 3, "y": 4}
        assert body["usage"]["prompt_tokens"] == 22 and body["usage"]["completion_tokens"] == 4
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (body["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(10 * 0.001 + 5 * 0.0001 + 7 * 0.002 + 4 * 0.002)
        assert rows[0]["prompt_tokens"] == 22 and rows[0]["completion_tokens"] == 4
        metadata: Final = rows[0]["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        assert parsed["cost_breakdown"]["input_cost"] == pytest.approx(0.0245)
        assert parsed["cost_breakdown"]["output_cost"] == pytest.approx(0.008)


@pytest.mark.covers("other.provider_wire.anthropic.bare_string_content_item_is_client_error")
@pytest.mark.parametrize(
    "text", [pytest.param("what type of file is this?", id="type_word"), pytest.param("hello", id="plain")]
)
def test_anthropic_bare_string_content_item_is_rejected_as_client_error_before_the_wire(
    gateway: Gateway, text: str
) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"upstream must not be reached: {request.target}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 16, "timeout": 5, "messages": [{"role": "system", "content": [text]}]},
        )
        assert response.status_code == 400, response.text
        assert wire.drain() == ()


@pytest.mark.covers("other.provider_wire.anthropic.messages_request_timeout_reaches_transport")
def test_anthropic_messages_slow_upstream_is_cut_off_at_the_deployment_request_timeout(gateway: Gateway) -> None:
    identity: Final = "anthropic-timeout-" + uuid.uuid4().hex
    prompt: Final = f"slow answer {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-sonnet-4-5-20250929"
        assert body["max_tokens"] == 16
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert not {
            "timeout",
            "request_timeout",
            "stream_chunk_size",
            "litellm_params",
            "litellm_metadata",
            "rpm",
            "tpm",
        }.intersection(body)
        time.sleep(1.5)
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "late"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            request_timeout=0.3,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 408, response.text
        assert "Timeout" in response.json()["error"]["message"], response.text
        assert eventually(wire.drain, lambda requests: len(requests) == 1, seconds=5, return_last_on_timeout=True)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_anthropic_messages_preserves_cli_beta_safeguards_and_result(
    gateway: Gateway, stream: bool
) -> None:
    identity: Final = "anthropic-safeguards-" + uuid.uuid4().hex
    beta: Final = cc.CLI_BETA + ",made-up-future-beta-2099-01-01"
    expected_beta: Final = ",".join(
        sorted(set((*beta.split(","), "context-management-2025-06-27", "prompt-caching-scope-2026-01-05")))
    )
    safeguards: Final = {"mode": "strict", "policies": ["sensitive-data"]}
    safeguard_results: Final = {"status": "allowed", "policy": "sensitive-data"}
    system: Final = [
        {"type": "text", "text": "Synthetic policy.", "cache_control": {"type": "ephemeral"}}
    ]
    tools: Final = [
        {
            "name": "lookup",
            "description": "Look up a synthetic record.",
            "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        },
        {
            "name": "update",
            "description": "Update a synthetic record.",
            "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        },
    ]
    response_body: Final = {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "Record checked."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 6, "output_tokens": 3},
        "safeguard_results": safeguard_results,
    }
    stream_reply: Final = (
        cc.sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **response_body,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 6, "output_tokens": 0},
                },
            },
        ),
        cc.sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Record checked."}},
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        cc.sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        cc.sse_frame("message_stop", {"type": "message_stop"}),
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == "synthetic-anthropic-key", request.headers
        assert request.headers["anthropic-beta"] == expected_beta, request.headers
        expected_body: Final = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Check the synthetic record."}],
            "system": system,
            "tools": tools,
            "metadata": {"user_id": cc.METADATA_USER_ID},
            "safeguards": safeguards,
            "stream": stream,
        }
        assert _JSON_OBJECT.validate_json(request.body) == expected_body, request.body
        if not stream:
            return Reply(body=json.dumps(response_body).encode())
        return Reply(chunks=stream_reply, content_type="text/event-stream")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-6",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        client_body: Final = {
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Check the synthetic record."}],
            "system": system,
            "tools": tools,
            "metadata": {"user_id": cc.METADATA_USER_ID},
            "safeguards": safeguards,
            "stream": stream,
        }
        response: Final = gateway.client.post(
            "/v1/messages?beta=true",
            json=client_body,
            headers=cc.cli_headers(gateway.key, beta=beta),
        )
        assert response.status_code == 200, response.text
        (sent,) = wire.drain()
        assert sent.target == "/v1/messages", sent.target

    if stream:
        expected_stream_events: Final = tuple(
            (
                event,
                {
                    **data,
                    **(
                        {
                            "message": {
                                **_JSON_OBJECT.validate_python(data["message"]),
                                "model": model,
                            }
                        }
                        if event == "message_start"
                        else {}
                    ),
                },
            )
            for event, data in cc.sse_events(b"".join(stream_reply).decode())
        )
        assert cc.sse_events(response.text) == expected_stream_events, response.text

    typed_message: Final = (
        _MESSAGE.validate_python(
            next(data["message"] for event, data in cc.sse_events(response.text) if event == "message_start")
        )
        if stream
        else _MESSAGE.validate_json(response.content)
    )
    assert typed_message.model_extra == {"safeguard_results": safeguard_results}, response.text


@pytest.mark.parametrize(
    ("status", "error_type", "error_class", "litellm_error"),
    [
        pytest.param(429, "rate_limit_error", anthropic.RateLimitError, "RateLimitError", id="rate_limit"),
        pytest.param(400, "invalid_request_error", anthropic.BadRequestError, "BadRequestError", id="invalid_request"),
    ],
)
def test_anthropic_messages_preserves_upstream_error_status_and_body(
    gateway: Gateway,
    status: int,
    error_type: str,
    error_class: type[anthropic.APIStatusError],
    litellm_error: str,
) -> None:
    message: Final = f"synthetic {error_type}"
    expected_error: Final = {"type": "error", "error": {"type": error_type, "message": message}}

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == "synthetic-anthropic-key", request.headers
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Trigger the scripted error."}],
            "stream": False,
        }, request.body
        return Reply(
            status=status,
            body=json.dumps(expected_error).encode(),
            headers={"retry-after": "7"} if status == 429 else {},
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-6",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        with anthropic.Anthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
        ) as client, pytest.raises(error_class) as caught:
            client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Trigger the scripted error."}],
            )
        assert caught.value.status_code == status, caught.value.response.text
        assert caught.value.response.json() == {
            "type": "error",
            "error": {
                "type": error_type,
                "message": (
                    f"litellm.{litellm_error}: AnthropicException - {json.dumps(expected_error)}\n\n"
                    f"LiteLLM: model group '{model}' failed with the error above. No fallback was attempted."
                ),
            },
        }, caught.value.response.text
        assert len(wire.drain()) == 1


def test_anthropic_messages_preserves_upstream_rate_limit_retry_after(gateway: Gateway) -> None:
    pytest.skip("BUG: /v1/messages drops the upstream retry-after header on 429")

    message: Final = "synthetic rate_limit_error"
    upstream_error: Final = {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": message},
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == "synthetic-anthropic-key", request.headers
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Trigger the scripted error."}],
            "stream": False,
        }, request.body
        return Reply(status=429, body=json.dumps(upstream_error).encode(), headers={"retry-after": "7"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-6",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        with anthropic.Anthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
        ) as client, pytest.raises(anthropic.RateLimitError) as caught:
            client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Trigger the scripted error."}],
            )
        assert caught.value.status_code == 429, caught.value.response.text
        assert caught.value.response.headers.get("retry-after") == "7", caught.value.response.text
        assert caught.value.response.json() == {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": (
                    f"litellm.RateLimitError: AnthropicException - {json.dumps(upstream_error)}\n\n"
                    f"LiteLLM: model group '{model}' failed with the error above. No fallback was attempted."
                ),
            },
        }, caught.value.response.text
        assert len(wire.drain()) == 1


def test_anthropic_messages_preserves_upstream_overloaded_error(gateway: Gateway) -> None:
    pytest.skip("BUG: /v1/messages maps an upstream 529 overloaded_error to HTTP 500")

    upstream_error: Final = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "synthetic overloaded error"},
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == "synthetic-anthropic-key", request.headers
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Trigger the scripted overload."}],
            "stream": False,
        }, request.body
        return Reply(status=529, body=json.dumps(upstream_error).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-6",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        with anthropic.Anthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
        ) as client, pytest.raises(anthropic.InternalServerError) as caught:
            client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Trigger the scripted overload."}],
            )
        assert caught.value.status_code == 529, caught.value.response.text
        response_body: Final = _JSON_OBJECT.validate_json(caught.value.response.content)
        error_body: Final = response_body.get("error")
        assert response_body.get("type") == "error" and isinstance(error_body, dict), caught.value.response.text
        assert error_body.get("type") == "overloaded_error", caught.value.response.text
        assert len(wire.drain()) == 1


def test_anthropic_messages_stream_surfaces_upstream_error_event(gateway: Gateway) -> None:
    error: Final = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "synthetic stream overload"},
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == "synthetic-anthropic-key", request.headers
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Trigger the scripted stream error."}],
            "stream": True,
        }, request.body
        return Reply(
            chunks=(
                cc.sse_frame(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_stream_error",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-sonnet-4-6",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 5, "output_tokens": 0},
                        },
                    },
                ),
                cc.sse_frame("error", error),
            ),
            content_type="text/event-stream",
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-6",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        response: Final = gateway.client.post(
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Trigger the scripted stream error."}],
                "stream": True,
            },
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 200, response.text
        events: Final = cc.sse_events(response.text)
        assert ("error", error) in events, response.text
        assert not any(event == "message_stop" for event, _ in events), response.text
        assert len(wire.drain()) == 1


def test_anthropic_messages_preserves_image_document_and_tool_result_blocks(gateway: Gateway) -> None:
    image_data: Final = base64.b64encode(b"synthetic png bytes").decode()
    pdf_data: Final = base64.b64encode(b"%PDF-1.7 synthetic document").decode()
    messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect these attachments."},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image_data},
                },
                {"type": "image", "source": {"type": "url", "url": "https://example.invalid/synthetic.png"}},
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "lookup-one", "name": "lookup", "input": {"id": "one"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "lookup-one",
                    "is_error": True,
                    "content": [
                        {"type": "text", "text": "Synthetic lookup failed."},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": image_data},
                        },
                    ],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "lookup-two", "name": "lookup", "input": {"id": "two"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "lookup-two", "content": "Synthetic lookup succeeded."},
                {"type": "text", "text": "Continue with the first result."},
            ],
        },
    ]
    content: Final = [
        {"type": "text", "text": "The requested blocks were preserved."},
        {"type": "tool_use", "id": "lookup-three", "name": "lookup", "input": {"id": "three"}},
        {"type": "thinking", "thinking": "The history contains two tool calls.", "signature": "synthetic-signature"},
    ]
    expected_response: Final = {
        "id": "msg_anthropic_blocks",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 12},
    }
    tools: Final = [
        {
            "name": "lookup",
            "description": "Look up a synthetic record.",
            "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        }
    ]

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == "synthetic-anthropic-key", request.headers
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 128,
            "messages": messages,
            "stream": False,
            "tools": tools,
        }, request.body
        return Reply(body=json.dumps(expected_response).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-6",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        with anthropic.Anthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
        ) as client:
            response: Final = client.messages.create(
                model=model,
                max_tokens=128,
                messages=messages,
                tools=tools,
            )
        expected_message: Final = _MESSAGE.validate_python({**expected_response, "model": model})
        assert response.model_dump(mode="json") == expected_message.model_dump(mode="json"), response
        assert len(wire.drain()) == 1
