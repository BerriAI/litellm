"""Differential parity for the response path: v1 transform_response vs v2.

Each case is a full cycle: the OpenAI request goes through both request
translators (so per-request state like the tool-name reverse map flows the
same way), then a recorded anthropic response JSON goes through v1's
``transform_response`` and v2's ``parse_response`` -> ``serialize_response``
-> ``ModelResponse(**body)``, and the two ``ModelResponse`` dumps must be
identical. uuid/time are frozen because both sides mint ambient ids.
"""

import copy
import json

import httpx
import pytest

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.utils import ModelResponse

from litellm.translation_seam import build_translation_deps, to_model_response

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.anthropic.response import parse_response

MODEL = "claude-sonnet-4-5"

_REQUESTS = {
    "text": {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "tools": {
        "model": MODEL,
        "max_tokens": 64,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "mcp.server/get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "messages": [{"role": "user", "content": "weather in Paris"}],
    },
    "thinking": {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": "think"}],
    },
    "json_tool": {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "capital of France?"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"capital": {"type": "string"}},
                },
            },
        },
    },
}

_RESPONSES = {
    "text": {
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Hello there."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 6,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 0,
            },
            "service_tier": "standard",
        },
    },
    "tools": {
        "id": "msg_01ABC",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "mcp_server_get_weather",
                "input": {"city": "Paris"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 30,
            "output_tokens": 20,
            "cache_creation_input_tokens": 4,
            "cache_read_input_tokens": 8,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 4,
                "ephemeral_1h_input_tokens": 0,
            },
        },
    },
    "thinking": {
        "id": "msg_01THINK",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet-20250219",
        "content": [
            {
                "type": "thinking",
                "thinking": "The capital of France is Paris.",
                "signature": "sig==",
            },
            {"type": "text", "text": "Paris."},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 40, "output_tokens": 30},
    },
    "json_tool": {
        "id": "msg_01JSON",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-haiku-20241022",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_02",
                "name": "json_tool_call",
                "input": {"capital": "Paris"},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 25, "output_tokens": 12},
    },
}

FROZEN_TIME = 1718064000.0


@pytest.fixture(autouse=True)
def _deterministic_ambient(monkeypatch):
    import time
    import uuid

    counter = iter(range(1, 10_000))
    monkeypatch.setattr(
        uuid,
        "uuid4",
        lambda: uuid.UUID(int=next(counter)),
    )
    monkeypatch.setattr(time, "time", lambda: FROZEN_TIME)


def _v1_model_response(name: str) -> dict:
    request = copy.deepcopy(_REQUESTS[name])
    config = AnthropicConfig()
    model = request["model"]
    params = {
        key: value for key, value in request.items() if key not in ("model", "messages")
    }
    optional = config.map_openai_params(
        copy.deepcopy(params), {}, model, drop_params=False
    )
    litellm_params: dict = {}
    config.transform_request(
        model, copy.deepcopy(request["messages"]), optional, litellm_params, {}
    )
    json_mode = optional.pop("json_mode", False)
    raw = httpx.Response(
        status_code=200,
        json=_RESPONSES[name],
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    logging = litellm.litellm_core_utils.litellm_logging.Logging(
        model=model,
        messages=copy.deepcopy(request["messages"]),
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="char-test",
        function_id="char-test",
    )
    result = config.transform_response(
        model=model,
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=logging,
        request_data={},
        messages=copy.deepcopy(request["messages"]),
        optional_params=optional,
        litellm_params=litellm_params,
        encoding=None,
        json_mode=json_mode,
    )
    return result.model_dump()


def _v2_model_response(name: str) -> dict:
    request = copy.deepcopy(_REQUESTS[name])
    deps = build_translation_deps()
    parsed = parse_request(request)
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(_RESPONSES[name]), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, deps)
    return to_model_response(body).model_dump()


def _norm(payload: dict) -> str:
    # The chatcmpl id is ambient (each side mints its own from the frozen
    # uuid counter in a different order); everything else must match.
    return json.dumps({**payload, "id": "chatcmpl-X"}, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_REQUESTS))
def test_v2_response_matches_v1(name: str) -> None:
    v1 = _v1_model_response(name)
    v2 = _v2_model_response(name)
    assert _norm(v2) == _norm(v1)
