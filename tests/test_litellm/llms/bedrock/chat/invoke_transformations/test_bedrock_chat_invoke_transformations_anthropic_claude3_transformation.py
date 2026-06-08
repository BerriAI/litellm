import json
import os
import sys
from unittest.mock import patch

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)


def test_get_supported_params_thinking():
    config = AmazonAnthropicClaudeConfig()
    params = config.get_supported_openai_params(
        model="anthropic.claude-sonnet-4-20250514-v1:0"
    )
    assert "thinking" in params


def test_aws_params_filtered_from_request_body():
    """
    Test that AWS authentication parameters are filtered out from the request body.

    This is a security test to ensure AWS credentials are not leaked in the request
    body sent to Bedrock. AWS params should only be used for request signing.

    Regression test for: AWS params (aws_role_name, aws_session_name, etc.)
    being included in the Bedrock InvokeModel request body.
    """
    config = AmazonAnthropicClaudeConfig()

    # Test messages
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    # Optional params with AWS authentication parameters that should be filtered out
    optional_params = {
        # Regular Anthropic params - these SHOULD be in the request
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        # AWS authentication params - these should NOT be in the request body
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "aws_session_token": "FwoGZXIvYXdzEBYaDH...",
        "aws_region_name": "us-west-2",
        "aws_role_name": "arn:aws:iam::123456789012:role/test-role",
        "aws_session_name": "test-session",
        "aws_profile_name": "default",
        "aws_web_identity_token": "token123",
        "aws_sts_endpoint": "https://sts.amazonaws.com",
        "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
        "aws_external_id": "external-id-123",
    }

    # Transform the request
    result = config.transform_request(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=messages,
        optional_params=optional_params.copy(),  # Copy to avoid mutation
        litellm_params={},
        headers={},
    )

    # Convert result to JSON string to check what would be sent in the request
    result_json = json.dumps(result)

    # Verify AWS authentication params are NOT in the request body
    assert (
        "aws_access_key_id" not in result_json
    ), "AWS access key should not be in request body"
    assert (
        "aws_secret_access_key" not in result_json
    ), "AWS secret key should not be in request body"
    assert (
        "aws_session_token" not in result_json
    ), "AWS session token should not be in request body"
    assert (
        "aws_region_name" not in result_json
    ), "AWS region should not be in request body"
    assert (
        "aws_role_name" not in result_json
    ), "AWS role name should not be in request body"
    assert (
        "aws_session_name" not in result_json
    ), "AWS session name should not be in request body"
    assert (
        "aws_profile_name" not in result_json
    ), "AWS profile name should not be in request body"
    assert (
        "aws_web_identity_token" not in result_json
    ), "AWS web identity token should not be in request body"
    assert (
        "aws_sts_endpoint" not in result_json
    ), "AWS STS endpoint should not be in request body"
    assert (
        "aws_bedrock_runtime_endpoint" not in result_json
    ), "AWS bedrock endpoint should not be in request body"
    assert (
        "aws_external_id" not in result_json
    ), "AWS external ID should not be in request body"

    # Also check that the sensitive values themselves are not in the response
    assert (
        "AKIAIOSFODNN7EXAMPLE" not in result_json
    ), "AWS access key value leaked in request body"
    assert (
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in result_json
    ), "AWS secret key value leaked in request body"
    assert (
        "arn:aws:iam::123456789012:role/test-role" not in result_json
    ), "AWS role ARN leaked in request body"
    assert "test-session" not in result_json, "AWS session name leaked in request body"

    # Verify normal params ARE still in the request body
    assert result["max_tokens"] == 100, "max_tokens should be in request body"
    assert result["temperature"] == 0.7, "temperature should be in request body"
    assert result["top_p"] == 0.9, "top_p should be in request body"

    # Verify Bedrock-specific params are added
    assert (
        result["anthropic_version"] == "bedrock-2023-05-31"
    ), "anthropic_version should be set"
    assert "model" not in result, "model should be removed for Bedrock Invoke API"
    assert "stream" not in result, "stream should be removed for Bedrock Invoke API"

    # Verify messages are present
    assert "messages" in result, "messages should be in request body"
    assert len(result["messages"]) == 1, "should have 1 message"


def test_output_format_conversion_to_inline_schema():
    """
    Test that output_format is converted to inline schema in message content for Bedrock Invoke.

    Bedrock Invoke doesn't support the output_format parameter, so LiteLLM converts it by
    embedding the schema directly into the user message content.
    """
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )

    config = AmazonAnthropicClaudeMessagesConfig()

    # Test messages
    messages = [
        {
            "role": "user",
            "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan.",
        }
    ]

    # Output format with schema
    output_format_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "plan_interest": {"type": "string"},
        },
        "required": ["name", "email", "plan_interest"],
        "additionalProperties": False,
    }

    anthropic_messages_optional_request_params = {
        "max_tokens": 1024,
        "output_format": {"type": "json_schema", "schema": output_format_schema},
    }

    # Transform the request
    result = config.transform_anthropic_messages_request(
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
        litellm_params={},
        headers={},
    )

    # Verify output_format was removed from the request
    assert (
        "output_format" not in result
    ), "output_format should be removed from request body"

    # Verify the schema was added to the last user message content
    assert "messages" in result
    last_user_message = result["messages"][0]
    assert last_user_message["role"] == "user"

    content = last_user_message["content"]
    assert isinstance(content, list), "content should be a list"
    assert len(content) == 2, "content should have 2 items (original text + schema)"

    # Check original text is preserved
    assert content[0]["type"] == "text"
    assert "John Smith" in content[0]["text"]

    # Check schema was added as JSON string
    assert content[1]["type"] == "text"
    schema_text = content[1]["text"]

    # Parse the schema JSON
    parsed_schema = json.loads(schema_text)
    assert parsed_schema["type"] == "object"
    assert "name" in parsed_schema["properties"]
    assert "email" in parsed_schema["properties"]
    assert "plan_interest" in parsed_schema["properties"]
    assert parsed_schema["required"] == ["name", "email", "plan_interest"]

    # Verify other params are preserved
    assert result["max_tokens"] == 1024
    assert result["anthropic_version"] == "bedrock-2023-05-31"


def test_output_format_conversion_with_string_content():
    """
    Test that output_format conversion works when message content is a string (not a list).
    """
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )

    config = AmazonAnthropicClaudeMessagesConfig()

    # Test messages with string content
    messages = [{"role": "user", "content": "What is 2+2?"}]

    output_format_schema = {
        "type": "object",
        "properties": {"result": {"type": "integer"}},
    }

    anthropic_messages_optional_request_params = {
        "max_tokens": 100,
        "output_format": {"type": "json_schema", "schema": output_format_schema},
    }

    # Transform the request
    result = config.transform_anthropic_messages_request(
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
        litellm_params={},
        headers={},
    )

    # Verify the content was converted to list format
    last_user_message = result["messages"][0]
    content = last_user_message["content"]
    assert isinstance(content, list), "content should be converted to list"
    assert len(content) == 2, "content should have 2 items"

    # Check original text
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What is 2+2?"

    # Check schema was added
    assert content[1]["type"] == "text"
    parsed_schema = json.loads(content[1]["text"])
    assert "result" in parsed_schema["properties"]


def test_output_format_with_no_schema():
    """
    Test that if output_format has no schema, the conversion is skipped gracefully.
    """
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )

    config = AmazonAnthropicClaudeMessagesConfig()

    messages = [{"role": "user", "content": "Hello"}]

    anthropic_messages_optional_request_params = {
        "max_tokens": 100,
        "output_format": {
            "type": "json_schema"
            # No schema field
        },
    }

    # Transform the request
    result = config.transform_anthropic_messages_request(
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        messages=messages,
        anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
        litellm_params={},
        headers={},
    )

    # Verify output_format was removed but no schema was added
    assert "output_format" not in result
    last_user_message = result["messages"][0]

    # Content should remain as string (not converted to list)
    assert isinstance(last_user_message["content"], str)
    assert last_user_message["content"] == "Hello"


def test_opus_4_5_model_detection():
    """
    Test that the _is_claude_opus_4_5 method correctly identifies Opus 4.5 models
    with various naming conventions.
    """
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )

    config = AmazonAnthropicClaudeMessagesConfig()

    # Test various Opus 4.5 naming patterns
    opus_4_5_models = [
        "anthropic.claude-opus-4-5-20250514-v1:0",
        "anthropic.claude-opus-4.5-20250514-v1:0",
        "anthropic.claude-opus_4_5-20250514-v1:0",
        "anthropic.claude-opus_4.5-20250514-v1:0",
        "us.anthropic.claude-opus-4-5-20250514-v1:0",
        "ANTHROPIC.CLAUDE-OPUS-4-5-20250514-V1:0",  # Case insensitive
    ]

    for model in opus_4_5_models:
        assert config._is_claude_opus_4_5(model), f"Should detect {model} as Opus 4.5"

    # Test non-Opus 4.5 models
    non_opus_4_5_models = [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-opus-4-20250514-v1:0",  # Opus 4, not 4.5
        "anthropic.claude-opus-4-1-20250514-v1:0",  # Opus 4.1, not 4.5
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ]

    for model in non_opus_4_5_models:
        assert not config._is_claude_opus_4_5(
            model
        ), f"Should not detect {model} as Opus 4.5"


# def test_structured_outputs_beta_header_filtered_for_bedrock_invoke():
#     """
#     Test that unsupported beta headers are filtered out for Bedrock Invoke API.

#     Bedrock Invoke API only supports a specific whitelist of beta flags and returns
#     "invalid beta flag" error for others (e.g., structured-outputs, mcp-servers).
#     This test ensures unsupported headers are filtered while keeping supported ones.

#     Fixes: https://github.com/BerriAI/litellm/issues/16726
#     """
#     config = AmazonAnthropicClaudeConfig()

#     messages = [{"role": "user", "content": "test"}]

#     # Test 1: structured-outputs beta header (unsupported)
#     headers = {"anthropic-beta": "structured-outputs-2025-11-13"}

#     result = config.transform_request(
#         model="anthropic.claude-4-0-sonnet-20250514-v1:0",
#         messages=messages,
#         optional_params={},
#         litellm_params={},
#         headers=headers,
#     )

#     # Verify structured-outputs beta is filtered out
#     anthropic_beta = result.get("anthropic_beta", [])
#     assert not any("structured-outputs" in beta for beta in anthropic_beta), \
#         f"structured-outputs beta should be filtered, got: {anthropic_beta}"

#     # Test 2: mcp-servers beta header (unsupported - the main issue from #16726)
#     headers = {"anthropic-beta": "mcp-servers-2025-12-04"}

#     result = config.transform_request(
#         model="anthropic.claude-4-0-sonnet-20250514-v1:0",
#         messages=messages,
#         optional_params={},
#         litellm_params={},
#         headers=headers,
#     )

#     # Verify mcp-servers beta is filtered out
#     anthropic_beta = result.get("anthropic_beta", [])
#     assert not any("mcp-servers" in beta for beta in anthropic_beta), \
#         f"mcp-servers beta should be filtered, got: {anthropic_beta}"

#     # Test 3: Mix of supported and unsupported beta headers
#     headers = {"anthropic-beta": "computer-use-2024-10-22,mcp-servers-2025-12-04,structured-outputs-2025-11-13"}

#     result = config.transform_request(
#         model="anthropic.claude-4-0-sonnet-20250514-v1:0",
#         messages=messages,
#         optional_params={},
#         litellm_params={},
#         headers=headers,
#     )

#     # Verify only supported betas are kept
#     anthropic_beta = result.get("anthropic_beta", [])
#     assert not any("structured-outputs" in beta for beta in anthropic_beta), \
#         f"structured-outputs beta should be filtered, got: {anthropic_beta}"
#     assert not any("mcp-servers" in beta for beta in anthropic_beta), \
#         f"mcp-servers beta should be filtered, got: {anthropic_beta}"
#     assert any("computer-use" in beta for beta in anthropic_beta), \
#         f"computer-use beta should be kept, got: {anthropic_beta}"


def test_output_config_forwarded_for_bedrock_chat_invoke_request():
    """Bedrock Invoke chat path forwards ``output_config`` for adaptive Claude models."""
    config = AmazonAnthropicClaudeConfig()

    messages = [{"role": "user", "content": "test"}]

    optional_params = {
        "max_tokens": 100,
        "output_config": {"effort": "high"},
    }

    result = config.transform_request(
        model="anthropic.claude-opus-4-7",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"effort": "high"}
    assert result["max_tokens"] == 100


def test_output_config_format_converted_for_bedrock_chat_invoke_request():
    """Bedrock Invoke chat path consumes ``output_config.format`` before forwarding."""
    config = AmazonAnthropicClaudeConfig()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    result = config.transform_request(
        model="anthropic.claude-opus-4-7",
        messages=[{"role": "user", "content": "test"}],
        optional_params={
            "max_tokens": 100,
            "output_config": {
                "effort": "xhigh",
                "format": {"type": "json_schema", "schema": schema},
            },
        },
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"effort": "xhigh"}
    last_content = result["messages"][0]["content"]
    assert json.loads(last_content[-1]["text"]) == schema


@pytest.mark.parametrize(
    "model,expected_effort",
    [
        ("anthropic.claude-opus-4-5-20251101-v1:0", "high"),
        ("anthropic.claude-opus-4-6-v1", "max"),
        ("anthropic.claude-opus-4-7", "xhigh"),
    ],
)
def test_output_config_effort_normalized_for_bedrock_chat_invoke_request(
    model, expected_effort
):
    """Bedrock Invoke chat path accepts ``xhigh`` and forwards the provider-safe effort."""
    config = AmazonAnthropicClaudeConfig()

    result = config.transform_request(
        model=model,
        messages=[{"role": "user", "content": "test"}],
        optional_params={
            "max_tokens": 100,
            "output_config": {"effort": "xhigh"},
        },
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"effort": expected_effort}


def test_bedrock_chat_invoke_checks_output_config_support_with_bedrock_provider():
    config = AmazonAnthropicClaudeConfig()
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"max_tokens": 100, "output_config": {"effort": "high"}}

    with patch(
        "litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        return_value=True,
    ) as mock_supports_factory:
        result = config.transform_request(
            model="us.anthropic.claude-opus-4-7",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

    # Two distinct flag lookups now happen (both scoped to bedrock provider):
    #   - supports_native_structured_output (decides whether to strip output_format)
    #   - supports_output_config             (decides whether to strip output_config)
    mock_supports_factory.assert_any_call(
        model="us.anthropic.claude-opus-4-7",
        custom_llm_provider="bedrock",
        key="supports_output_config",
    )
    mock_supports_factory.assert_any_call(
        model="us.anthropic.claude-opus-4-7",
        custom_llm_provider="bedrock",
        key="supports_native_structured_output",
    )
    assert result["output_config"] == {"effort": "high"}


def test_output_format_removed_from_bedrock_invoke_request():
    """
    Test that output_format parameter is removed from Bedrock Invoke requests.

    Bedrock Invoke API doesn't support the output_format parameter (only supported
    in Anthropic Messages API). This test ensures it's removed to prevent errors.
    """
    config = AmazonAnthropicClaudeConfig()

    messages = [{"role": "user", "content": "test"}]

    # Create a request with output_format via map_openai_params
    non_default_params = {"response_format": {"type": "json_object"}}
    optional_params = {}

    # This should trigger tool-based structured outputs
    optional_params = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="anthropic.claude-4-0-sonnet-20250514-v1:0",
        drop_params=False,
    )

    result = config.transform_request(
        model="anthropic.claude-4-0-sonnet-20250514-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # Verify output_format is not in the request
    assert (
        "output_format" not in result
    ), f"output_format should be removed for Bedrock Invoke, got keys: {result.keys()}"


# -----------------------------------------------------------------------
# Native ``output_format`` vs tool-injection — Bedrock invoke per-model
# -----------------------------------------------------------------------
# AWS Bedrock invoke accepts native ``output_format`` for Claude 4.5+ except
# opus-4-7 and opus-4-8 (verified Jun 2026). For those two, LiteLLM falls
# back to the synthetic ``json_tool_call`` tool. For every other Claude 4.5+
# model the native field flows to AWS, and the gating beta header
# ``structured-outputs-2025-11-13`` is auto-added.

_REVENUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "revenue_b": {"type": "number"},
        "yoy_growth_pct": {"type": "number"},
    },
    "required": ["revenue_b", "yoy_growth_pct"],
}

_OPENAI_RESPONSE_FORMAT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "RevenueData", "schema": _REVENUE_SCHEMA},
}


def _build_invoke_request(
    model: str,
    non_default_params: dict,
    *,
    bedrock_supports_native_output_format: bool,
) -> dict:
    """Run map_openai_params + transform_request with the in-module
    ``_supports_factory`` patched so the native-``output_format`` capability
    lookup is deterministic regardless of remote model_cost state.

    ``bedrock_supports_native_output_format`` is what
    ``_bedrock_invoke_supports_native_output_format(model)`` should return —
    i.e. the value of ``supports_native_structured_output`` for ``model``'s
    Bedrock entry in ``model_prices_and_context_window.json``.
    """

    def fake_supports_factory(model, custom_llm_provider, key):
        if key == "supports_native_structured_output":
            return bedrock_supports_native_output_format
        # supports_output_config is True for opus-4-X in the cost map; preserve
        # that so the downstream output_config strip behaves as in production.
        if key == "supports_output_config":
            return True
        return False

    config = AmazonAnthropicClaudeConfig()
    messages = [{"role": "user", "content": "test"}]

    with patch(
        "litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation._supports_factory",
        side_effect=fake_supports_factory,
    ):
        optional_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )
        return config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )


def test_response_format_emits_native_output_format_for_opus_4_6_invoke():
    """Models AWS accepts native ``output_format`` for must receive it on the wire
    (no synthetic tool-injection) plus the gating beta header."""
    result = _build_invoke_request(
        model="anthropic.claude-opus-4-6-v1",
        non_default_params={"response_format": _OPENAI_RESPONSE_FORMAT_JSON_SCHEMA},
        bedrock_supports_native_output_format=True,
    )

    assert "output_format" in result, (
        f"opus-4-6 invoke should keep native output_format on the wire; "
        f"got keys: {sorted(result.keys())}"
    )
    assert result["output_format"].get("type") == "json_schema"

    assert "tools" not in result, (
        f"opus-4-6 invoke with native output_format must NOT inject json_tool_call; "
        f"got tools={result.get('tools')!r}"
    )
    assert "tool_choice" not in result

    beta = result.get("anthropic_beta", [])
    assert "structured-outputs-2025-11-13" in beta, (
        f"structured-outputs-2025-11-13 beta header must be present when "
        f"output_format is sent; got anthropic_beta={beta!r}"
    )


def test_response_format_uses_tool_workaround_for_opus_4_7_invoke():
    """opus-4-7's Bedrock entry has ``supports_native_structured_output: false``
    — AWS rejects native ``output_format`` here, so LiteLLM must fall back to
    the synthetic ``json_tool_call`` tool."""
    result = _build_invoke_request(
        model="anthropic.claude-opus-4-7",
        non_default_params={"response_format": _OPENAI_RESPONSE_FORMAT_JSON_SCHEMA},
        bedrock_supports_native_output_format=False,
    )

    assert "output_format" not in result, (
        f"opus-4-7 invoke must NOT send native output_format (AWS rejects); "
        f"got output_format={result.get('output_format')!r}"
    )

    tools = result.get("tools") or []
    assert any(
        t.get("name") == "json_tool_call" for t in tools
    ), f"opus-4-7 invoke must inject json_tool_call tool; got tools={tools!r}"
    assert (
        result.get("tool_choice", {}).get("name") == "json_tool_call"
    ), "opus-4-7 invoke must force tool_choice to json_tool_call (no thinking)"

    beta = result.get("anthropic_beta", [])
    assert (
        "structured-outputs-2025-11-13" not in beta
    ), "structured-outputs beta must NOT be set when output_format is absent"


def test_response_format_plus_reasoning_preserves_adaptive_thinking_on_opus_4_7():
    """Regression test for the model-spoof side-effect: when ``response_format`` is
    combined with ``reasoning_effort``, the previous workaround spoofed the model
    name and corrupted ``_is_adaptive_thinking_model`` to return False, producing
    legacy ``thinking.enabled+budget_tokens`` that AWS opus-4-7 rejects. After the
    fix, the real model name flows through and adaptive thinking is preserved."""
    result = _build_invoke_request(
        model="anthropic.claude-opus-4-7",
        non_default_params={
            "response_format": _OPENAI_RESPONSE_FORMAT_JSON_SCHEMA,
            "reasoning_effort": "medium",
        },
        bedrock_supports_native_output_format=False,
    )

    # Tool injection still happens (AWS rejects native output_format on 4-7)
    tools = result.get("tools") or []
    assert any(t.get("name") == "json_tool_call" for t in tools), (
        f"opus-4-7 invoke must still inject json_tool_call when reasoning is on; "
        f"got tools={tools!r}"
    )

    # No tool_choice — AWS rejects forcing alongside thinking
    assert (
        "tool_choice" not in result
    ), f"AWS rejects tool_choice forcing when thinking is on; got tool_choice={result.get('tool_choice')!r}"

    # Modern adaptive-thinking interface (NOT legacy enabled+budget)
    thinking = result.get("thinking") or {}
    assert (
        thinking.get("type") == "adaptive"
    ), f"opus-4-7 requires thinking.type=adaptive; got thinking={thinking!r}"
    assert (
        "budget_tokens" not in thinking
    ), f"opus-4-7 rejects budget_tokens; got thinking={thinking!r}"

    output_config = result.get("output_config") or {}
    assert output_config.get("effort") == "medium", (
        f"opus-4-7 requires output_config.effort alongside adaptive thinking; "
        f"got output_config={output_config!r}"
    )


def test_response_format_plus_reasoning_uses_native_with_adaptive_on_opus_4_6():
    """Combined case for a model AWS accepts native output_format on — both
    ``output_format`` (structured output) and ``thinking.adaptive +
    output_config.effort`` (reasoning) coexist in the wire body, no tools."""
    result = _build_invoke_request(
        model="anthropic.claude-opus-4-6-v1",
        non_default_params={
            "response_format": _OPENAI_RESPONSE_FORMAT_JSON_SCHEMA,
            "reasoning_effort": "medium",
        },
        bedrock_supports_native_output_format=True,
    )

    assert "output_format" in result
    assert result["output_format"].get("type") == "json_schema"

    thinking = result.get("thinking") or {}
    assert (
        thinking.get("type") == "adaptive"
    ), f"opus-4-6 should use adaptive thinking; got thinking={thinking!r}"
    assert "budget_tokens" not in thinking

    output_config = result.get("output_config") or {}
    assert output_config.get("effort") == "medium"

    assert "tools" not in result, (
        f"opus-4-6 with native output_format must not inject tools; "
        f"got tools={result.get('tools')!r}"
    )

    beta = result.get("anthropic_beta", [])
    assert "structured-outputs-2025-11-13" in beta
