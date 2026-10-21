"""
Tests for embedding thought signatures in tool call IDs for OpenAI client compatibility.

When using OpenAI clients (instead of LiteLLM SDK), provider_specific_fields are not preserved.
This test suite validates that thought signatures can be embedded in tool call IDs and extracted
when converting back to Gemini format.

Note: Embedding signatures in tool call IDs is a beta feature that requires
enable_preview_features=True to be enabled.
"""

import json
import re
from collections.abc import Mapping
from itertools import chain
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
    encode_tool_call_id_with_signature,
    get_thought_signature_from_tool,
    convert_to_gemini_tool_call_invoke,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.vertex_ai import HttpxPartType


def test_encode_decode_tool_call_id_with_signature():
    """Test that thought signatures can be encoded in and decoded from tool call IDs"""
    base_id = "call_abc123"
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Test encoding
    encoded_id = encode_tool_call_id_with_signature(base_id, test_signature)
    assert THOUGHT_SIGNATURE_SEPARATOR in encoded_id
    assert encoded_id.startswith(base_id)

    # Test decoding using factory function with realistic tool call structure
    tool = {
        "id": encoded_id,
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "arguments": '{"location": "Paris"}',
        },
    }

    extracted_signature = get_thought_signature_from_tool(tool)
    assert extracted_signature == test_signature

    # Verify base ID is preserved
    decoded_base_id = encoded_id.split(THOUGHT_SIGNATURE_SEPARATOR)[0]
    assert decoded_base_id == base_id


def test_encode_tool_call_id_without_signature():
    """Test that IDs without signatures are returned unchanged"""
    base_id = "call_abc123def456"

    # Encode without signature
    encoded_id = encode_tool_call_id_with_signature(base_id, None)
    assert encoded_id == base_id
    assert THOUGHT_SIGNATURE_SEPARATOR not in encoded_id

    # Decode ID without signature using factory function
    tool_obj = {"id": base_id, "type": "function"}
    decoded_signature = get_thought_signature_from_tool(tool_obj)
    assert decoded_signature is None


@pytest.mark.parametrize("enable_preview_features", [True, False])
def test_tool_call_id_includes_signature_in_response(enable_preview_features):
    """Test that tool call IDs in responses include embedded thought signatures only when preview features are enabled"""
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    parts_with_signature = [
        HttpxPartType(
            functionCall={
                "name": "get_current_temperature",
                "args": {"location": "Paris"},
            },
            thoughtSignature=test_signature,
        )
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=parts_with_signature,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    # Verify tool call exists
    assert tools is not None
    assert len(tools) == 1
    tool_call_id = tools[0]["id"]

    # Verify signature is always in provider_specific_fields
    assert (
        tools[0].get("provider_specific_fields", {}).get("thought_signature")
        == test_signature
    )

    # When preview features enabled, signature should be embedded in ID
    assert THOUGHT_SIGNATURE_SEPARATOR in tool_call_id
    # Verify we can decode it using the factory function
    tool_obj = {"id": tool_call_id, "type": "function"}
    decoded_sig = get_thought_signature_from_tool(tool_obj)
    assert decoded_sig == test_signature


def test_get_thought_signature_backward_compatibility():
    """Test that provider_specific_fields still works (backward compatibility)"""
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Test with provider_specific_fields (LiteLLM SDK scenario)
    tool = {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "arguments": '{"location": "Paris"}',
        },
        "provider_specific_fields": {"thought_signature": test_signature},
    }

    extracted_signature = get_thought_signature_from_tool(tool)
    assert extracted_signature == test_signature


def test_get_thought_signature_prioritizes_provider_fields():
    """Test that provider_specific_fields takes priority over tool call ID"""
    signature_in_fields = "signature_from_fields"
    signature_in_id = "signature_from_id"

    encoded_id = encode_tool_call_id_with_signature("call_abc123", signature_in_id)

    tool = {
        "id": encoded_id,
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "arguments": '{"location": "Paris"}',
        },
        "provider_specific_fields": {"thought_signature": signature_in_fields},
    }

    extracted_signature = get_thought_signature_from_tool(tool)
    # Should prioritize provider_specific_fields
    assert extracted_signature == signature_in_fields


def test_convert_to_gemini_with_embedded_signature():
    """Test that convert_to_gemini_tool_call_invoke extracts signatures from tool call IDs"""
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Create tool call ID with embedded signature (as OpenAI client would send)
    base_id = "call_abc123"
    encoded_id = encode_tool_call_id_with_signature(base_id, test_signature)

    # Assistant message as sent by OpenAI client (no provider_specific_fields)
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": encoded_id,  # ID has signature embedded
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    }

    gemini_parts = convert_to_gemini_tool_call_invoke(assistant_message)

    # Verify thought signature is extracted and sent to Gemini
    assert len(gemini_parts) == 1
    assert "function_call" in gemini_parts[0]
    assert "thoughtSignature" in gemini_parts[0]
    assert gemini_parts[0]["thoughtSignature"] == test_signature


@pytest.mark.parametrize("enable_preview_features", [True, False])
def test_openai_client_e2e_flow(enable_preview_features):
    """
    End-to-end test simulating OpenAI client usage:
    1. LiteLLM receives response from Gemini with thought signature
    2. LiteLLM embeds signature in tool call ID (if preview features enabled)
    3. OpenAI client sends message back with same tool call ID
    4. LiteLLM extracts signature from ID/provider_specific_fields and sends to Gemini
    """
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Step 1: Gemini returns function call with thought signature
    gemini_parts = [
        HttpxPartType(
            functionCall={
                "name": "get_current_temperature",
                "args": {"location": "Paris"},
            },
            thoughtSignature=test_signature,
        )
    ]

    # Step 2: LiteLLM transforms to OpenAI format
    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=gemini_parts,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    assert tools is not None
    assert len(tools) == 1
    tool_call_id = tools[0]["id"]

    assert THOUGHT_SIGNATURE_SEPARATOR in tool_call_id

    # Step 3: OpenAI client sends back assistant message
    # For the disabled case, we simulate that the client might have provider_specific_fields
    # or we use the embedded ID if preview features were enabled
    openai_assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,  # Preserved from response (with embedded signature)
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    }
    # Step 4: LiteLLM converts back to Gemini format, extracting signature
    gemini_parts_converted = convert_to_gemini_tool_call_invoke(
        openai_assistant_message
    )

    # Verify signature is preserved through the round trip
    assert len(gemini_parts_converted) == 1
    assert "thoughtSignature" in gemini_parts_converted[0]
    assert gemini_parts_converted[0]["thoughtSignature"] == test_signature


@pytest.mark.parametrize("enable_preview_features", [True, False])
def test_parallel_tool_calls_with_signatures(enable_preview_features):
    """Test that parallel tool calls preserve signatures correctly"""
    signature1 = "signature_for_first_call"
    # Only first call has signature (Gemini behavior for parallel calls)

    gemini_parts = [
        HttpxPartType(
            functionCall={"name": "get_temperature", "args": {"location": "Paris"}},
            thoughtSignature=signature1,
        ),
        HttpxPartType(
            functionCall={"name": "get_temperature", "args": {"location": "London"}},
            # No signature for second parallel call
        ),
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=gemini_parts,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    assert tools is not None
    assert len(tools) == 2

    # First tool call should have signature in provider_specific_fields
    assert (
        tools[0].get("provider_specific_fields", {}).get("thought_signature")
        == signature1
    )

    # When preview features enabled, first tool call has signature in ID
    assert THOUGHT_SIGNATURE_SEPARATOR in tools[0]["id"]
    sig1 = get_thought_signature_from_tool({"id": tools[0]["id"], "type": "function"})
    assert sig1 == signature1

    # Second tool call has no signature in ID (regardless of flag)
    assert THOUGHT_SIGNATURE_SEPARATOR not in tools[1]["id"]
    sig2 = get_thought_signature_from_tool({"id": tools[1]["id"], "type": "function"})
    assert sig2 is None


GEMINI_URL_PATTERN: Final = r"https://generativelanguage\.googleapis\.com/v1[a-z]*/models/gemini-3\.5-flash:.*"
BOUNDED_TOOL_CALL_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,64}")
LONG_SIGNATURE: Final = "Ab+/" * 200
WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
USER_TURN: Final = {"role": "user", "content": "What is the weather in Paris?"}


def _function_call_candidate(native_id: str | None) -> dict[str, object]:
    function_call: Final = {"name": "get_weather", "args": {"city": "Paris"}}
    return {
        "content": {
            "role": "model",
            "parts": [
                {
                    "functionCall": function_call if native_id is None else {**function_call, "id": native_id},
                    "thoughtSignature": LONG_SIGNATURE,
                }
            ],
        },
        "finishReason": "STOP",
        "index": 0,
    }


def _gemini_body(native_id: str | None) -> dict[str, object]:
    return {
        "candidates": [_function_call_candidate(native_id)],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7, "totalTokenCount": 12},
    }


def _gemini_response(native_id: str | None, stream: bool) -> httpx.Response:
    body: Final = _gemini_body(native_id)
    if stream:
        return httpx.Response(200, text=f"data: {json.dumps(body)}\n\n", headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=body)


def _returned_tool_calls(
    native_id: str | None, stream: bool, request_params: Mapping[str, object]
) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
    with respx.mock:
        route: Final = respx.post(url__regex=GEMINI_URL_PATTERN).mock(return_value=_gemini_response(native_id, stream))
        response: Final = litellm.completion(
            model="gemini/gemini-3.5-flash",
            messages=[USER_TURN],
            tools=[WEATHER_TOOL],
            stream=stream,
            api_key="test-gemini-key",
            **request_params,
        )
        returned_calls: Final = (
            chain.from_iterable(chunk.choices[0].delta.tool_calls or () for chunk in response)
            if stream
            else response.choices[0].message.tool_calls
        )
        tool_calls: Final = [call.model_dump() for call in returned_calls]
        sent_body: Final = json.loads(route.calls.last.request.content)
    return tool_calls, sent_body


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("native_id", [None, "kq1z0x7p"])
def test_tool_call_id_stays_bounded_when_thought_signature_kept_out_of_it(stream: bool, native_id: str | None):
    tool_calls, sent_body = _returned_tool_calls(native_id, stream, {"thought_signature_in_tool_call_id": False})

    tool_call_ids: Final = [call["id"] for call in tool_calls if call["id"]]
    assert len(tool_call_ids) == 1
    assert BOUNDED_TOOL_CALL_ID.fullmatch(tool_call_ids[0])
    assert native_id is None or tool_call_ids[0] == native_id
    assert tool_calls[0]["provider_specific_fields"]["thought_signature"] == LONG_SIGNATURE
    assert "thought_signature_in_tool_call_id" not in json.dumps(sent_body)


@pytest.mark.parametrize("stream", [False, True])
def test_tool_call_id_carries_thought_signature_by_default(stream: bool):
    tool_calls, _ = _returned_tool_calls(None, stream, {})

    tool_call_ids: Final = [call["id"] for call in tool_calls if call["id"]]
    assert len(tool_call_ids) == 1
    assert get_thought_signature_from_tool({"id": tool_call_ids[0]}) == LONG_SIGNATURE


@pytest.mark.parametrize("client_keeps_provider_fields", [True, False])
def test_bounded_tool_call_id_round_trips_to_gemini(client_keeps_provider_fields: bool):
    tool_calls, _ = _returned_tool_calls(None, False, {"thought_signature_in_tool_call_id": False})
    returned_call: Final = tool_calls[0]
    replayed_call: Final = {
        "id": returned_call["id"],
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        **(
            {"provider_specific_fields": returned_call["provider_specific_fields"]}
            if client_keeps_provider_fields
            else {}
        ),
    }

    with respx.mock:
        route: Final = respx.post(url__regex=GEMINI_URL_PATTERN).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": "Sunny"}]}, "finishReason": "STOP", "index": 0}
                    ],
                    "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1, "totalTokenCount": 10},
                },
            )
        )
        litellm.completion(
            model="gemini/gemini-3.5-flash",
            messages=[
                USER_TURN,
                {"role": "assistant", "content": None, "tool_calls": [replayed_call]},
                {"role": "tool", "tool_call_id": returned_call["id"], "content": '{"temp_c": 21}'},
            ],
            tools=[WEATHER_TOOL],
            api_key="test-gemini-key",
            thought_signature_in_tool_call_id=False,
        )
        sent_body: Final = json.loads(route.calls.last.request.content)

    model_turns: Final = [turn for turn in sent_body["contents"] if turn["role"] == "model"]
    model_parts: Final = chain.from_iterable(turn["parts"] for turn in model_turns)
    replayed_signatures: Final = [part["thoughtSignature"] for part in model_parts if "function_call" in part]
    assert len(replayed_signatures) == 1
    assert (replayed_signatures[0] == LONG_SIGNATURE) is client_keeps_provider_fields
    assert replayed_signatures[0]


def test_tool_result_with_bare_id_matches_call_whose_id_the_client_re_signed():
    bare_id: Final = "call_601401"
    with respx.mock:
        route: Final = respx.post(url__regex=GEMINI_URL_PATTERN).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": "Sunny"}]}, "finishReason": "STOP", "index": 0}
                    ],
                    "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1, "totalTokenCount": 10},
                },
            )
        )
        litellm.completion(
            model="gemini/gemini-3.5-flash",
            messages=[
                USER_TURN,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": encode_tool_call_id_with_signature(bare_id, LONG_SIGNATURE),
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": bare_id, "content": '{"temp_c": 21}'},
            ],
            tools=[WEATHER_TOOL],
            api_key="test-gemini-key",
            thought_signature_in_tool_call_id=False,
        )
        sent_body: Final = json.loads(route.calls.last.request.content)

    parts: Final = list(chain.from_iterable(turn["parts"] for turn in sent_body["contents"]))
    function_calls: Final = [part for part in parts if "function_call" in part]
    function_responses: Final = [part["function_response"] for part in parts if "function_response" in part]
    assert [part["thoughtSignature"] for part in function_calls] == [LONG_SIGNATURE]
    assert [response["name"] for response in function_responses] == ["get_weather"]
