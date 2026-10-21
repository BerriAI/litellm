import json
import time
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server


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
