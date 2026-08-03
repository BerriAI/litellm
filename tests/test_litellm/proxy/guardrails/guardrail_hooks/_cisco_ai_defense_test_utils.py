import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import HTTPException
from httpx import Request, Response
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    TextChoices,
    TextCompletionResponse,
)


def _make_text_completion_response(text: str) -> TextCompletionResponse:
    return TextCompletionResponse(
        choices=[{"text": text, "index": 0, "finish_reason": "stop"}]
    )


def _make_model_response_with_content(content: str) -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content=content),
            )
        ]
    )


sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.cisco_ai_defense import (
    CiscoAIDefenseGuardrail,
    CiscoAIDefenseGuardrailMissingSecrets,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2

CISCO_BASE = "https://us.api.inspect.aidefense.security.cisco.com"
CHAT_URL = f"{CISCO_BASE}/api/v1/inspect/chat"
MCP_URL = f"{CISCO_BASE}/api/v1/inspect/mcp"


@contextmanager
def _patch_inspection_post(g: CiscoAIDefenseGuardrail, post_mock: Any):
    async def _send(request: Request, **kwargs: Any) -> Response:
        return await post_mock(
            url=str(request.url),
            headers=request.headers,
            json=json.loads(request.content.decode("utf-8")),
            follow_redirects=kwargs.get("follow_redirects"),
        )

    with patch.object(g.async_handler.client, "send", new=_send):
        yield post_mock


def _mock_inspect_response(
    json_body: dict, *, status: int = 200, url: str = CHAT_URL
) -> Response:
    return Response(
        status_code=status,
        json=json_body,
        request=Request(method="POST", url=url),
    )


def _safe_response(url: str = CHAT_URL) -> Response:
    return _mock_inspect_response(
        {
            "is_safe": True,
            "classifications": [],
            "severity": "NONE_SEVERITY",
            "rules": [],
            "action": "allow",
        },
        url=url,
    )


def _violation_response(url: str = CHAT_URL) -> Response:
    return _mock_inspect_response(
        {
            "is_safe": False,
            "classifications": ["SECURITY_VIOLATION", "PRIVACY_VIOLATION"],
            "severity": "HIGH",
            "rules": [
                {"rule_name": "Prompt Injection"},
                {"rule_name": "PII", "entity_types": ["Email Address"]},
            ],
            "explanation": "Detected jailbreak attempt with PII exfiltration",
            "event_id": "evt_123",
            "action": "block",
        },
        url=url,
    )


def _mcp_request(name="lookup", args=None, jsonrpc=False, **extra):
    args = args if args is not None else {}
    if jsonrpc:
        return {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
            **extra,
        }
    return {"mcp_tool_name": name, "mcp_arguments": args, **extra}


def _mcp_response(content=None, response_cost=0.0):
    if content is None:
        content = [{"type": "text", "text": "ok"}]
    return SimpleNamespace(
        mcp_tool_call_response=content,
        hidden_params=SimpleNamespace(response_cost=response_cost),
    )


def _mcp_result_text(content) -> str:
    if not content:
        return ""
    item = content[0] if isinstance(content, list) else content
    return getattr(item, "text", None) or item.get("text", "")


def _chat_request_tool_call_args(arguments: str) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "send_data",
                            "arguments": arguments,
                        },
                    }
                ],
            }
        ]
    }


def _chat_request_function_call_args(arguments: str) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "exfil",
                    "arguments": arguments,
                },
            }
        ]
    }


def _redact_response(
    *,
    sanitized_text=None,
    sanitized_messages=None,
    sanitized_mcp_arguments=None,
    sanitized_payload=None,
    classifications=("PRIVACY_VIOLATION",),
    rules=({"rule_name": "PII"},),
    severity="HIGH",
    url=CHAT_URL,
):
    body = {
        "is_safe": False,
        "classifications": list(classifications),
        "severity": severity,
        "rules": list(rules),
        "action": "redact",
    }
    if sanitized_text is not None:
        body["sanitized_text"] = sanitized_text
    if sanitized_messages is not None:
        body["sanitized_messages"] = sanitized_messages
    if sanitized_mcp_arguments is not None:
        body["sanitized_mcp_arguments"] = sanitized_mcp_arguments
    if sanitized_payload is not None:
        body["sanitized_payload"] = sanitized_payload
    return _mock_inspect_response(body, url=url)


def _responses_api_response(text, role="assistant"):
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.responses.main import GenericResponseOutputItem, OutputText

    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        output=[
            GenericResponseOutputItem(
                type="message",
                id="msg_1",
                status="completed",
                role=role,
                content=[OutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice=None,
        tools=None,
        top_p=None,
        usage=None,
    )


def _make_guardrail(
    inspection_type="chat",
    event_hook="pre_call",
    *,
    name="t",
    api_key="x",
    default_on=True,
    **kwargs,
):
    return CiscoAIDefenseGuardrail(
        guardrail_name=name,
        api_key=api_key,
        inspection_type=inspection_type,
        event_hook=event_hook,
        default_on=default_on,
        **kwargs,
    )


def _find_callback(name):
    from litellm.proxy.guardrails.guardrail_hooks.cisco_ai_defense import (
        CiscoAIDefenseGuardrail,
    )

    for cb in litellm.callbacks:
        if isinstance(cb, CiscoAIDefenseGuardrail) and cb.guardrail_name == name:
            return cb
    raise AssertionError(f"Cisco guardrail {name!r} not in litellm.callbacks")


def _make_streaming_chunks(parts):
    chunks = []
    for i, part in enumerate(parts):
        chunks.append(
            ModelResponseStream(
                id="resp_1",
                choices=[
                    StreamingChoices(
                        delta=Delta(content=part, role="assistant" if i == 0 else None),
                        finish_reason="stop" if i == len(parts) - 1 else None,
                        index=0,
                    )
                ],
                created=1234567890,
                model="gpt-4",
                object="chat.completion.chunk",
            )
        )
    return chunks


async def _aiter(items):
    for item in items:
        yield item


async def _streaming_setup(
    g,
    chunks,
    cisco_response=None,
    upstream=None,
    request_data=None,
    post_mock=None,
):
    if post_mock is None:
        post_mock = (
            AsyncMock(return_value=cisco_response) if cisco_response else AsyncMock()
        )
    stream_source = upstream if upstream is not None else _aiter(chunks)
    if request_data is None:
        request_data = {"messages": [{"role": "user", "content": "hi"}]}
    received: list = []
    with _patch_inspection_post(g, post_mock):
        async for chunk in g.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=stream_source,
            request_data=request_data,
        ):
            received.append(chunk)
    return received, post_mock


__all__ = [
    "Any",
    "AsyncMock",
    "CHAT_URL",
    "CISCO_BASE",
    "Choices",
    "CiscoAIDefenseGuardrail",
    "CiscoAIDefenseGuardrailMissingSecrets",
    "Delta",
    "Dict",
    "DualCache",
    "HTTPException",
    "MCP_URL",
    "Message",
    "ModelResponse",
    "ModelResponseStream",
    "Request",
    "Response",
    "SimpleNamespace",
    "StreamingChoices",
    "TextChoices",
    "TextCompletionResponse",
    "UserAPIKeyAuth",
    "_aiter",
    "_chat_request_function_call_args",
    "_chat_request_tool_call_args",
    "_find_callback",
    "_make_guardrail",
    "_make_model_response_with_content",
    "_make_streaming_chunks",
    "_make_text_completion_response",
    "_mcp_request",
    "_mcp_response",
    "_mcp_result_text",
    "_mock_inspect_response",
    "_patch_inspection_post",
    "_redact_response",
    "_responses_api_response",
    "_safe_response",
    "_streaming_setup",
    "_violation_response",
    "contextmanager",
    "datetime",
    "init_guardrails_v2",
    "json",
    "litellm",
    "os",
    "patch",
    "pytest",
    "sys",
]
