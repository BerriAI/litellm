import pytest

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    encrypted_reasoning_signature,
)
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.llms.bedrock_mantle.messages.transformation import BedrockMantleAnthropicMessagesConfig
from litellm.llms.databricks.messages.transformation import DatabricksAnthropicMessagesConfig
from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.messages.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


def _transform(messages):
    return AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params={},
        headers={},
    )


def test_reasoning_replayed_from_the_responses_bridge_never_reaches_anthropic():
    """Claude Code resumed on a Claude model echoes the thinking blocks a gpt turn produced."""
    messages = [
        {"role": "user", "content": "Solve it."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": encrypted_reasoning_signature("gAAAA_1")},
                {"type": "redacted_thinking", "data": encrypted_reasoning_signature("gAAAA_2")},
                {"type": "text", "text": "The answer."},
            ],
        },
        {"role": "user", "content": "And the next one?"},
    ]
    request = _transform(messages)
    assert request["messages"][1]["content"] == [{"type": "text", "text": "The answer."}]
    assert len(messages[1]["content"]) == 3


def test_anthropic_signed_thinking_blocks_are_forwarded_untouched():
    messages = [
        {"role": "user", "content": "Solve it."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
                {"type": "text", "text": "The answer."},
            ],
        },
    ]
    assert _transform(messages)["messages"] == messages


def _history_with_a_gemini_thought(signature_fields):
    return [
        {"role": "user", "content": "Read numbers.txt."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "numbers.txt"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "3 7 12"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Gemini reasoning", **signature_fields},
                {"type": "text", "text": "Two subsets."},
            ],
        },
        {"role": "user", "content": "Now count the odd ones."},
    ]


@pytest.mark.parametrize(
    ("config", "model"),
    [
        (AnthropicMessagesConfig(), "claude-sonnet-4-5"),
        (AzureAnthropicMessagesConfig(), "claude-sonnet-4-5"),
        (VertexAIPartnerModelsAnthropicMessagesConfig(), "claude-sonnet-4-5@20250929"),
        (AmazonAnthropicClaudeMessagesConfig(), "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        (BedrockMantleAnthropicMessagesConfig(), "anthropic.claude-mythos-preview"),
        (DatabricksAnthropicMessagesConfig(), "databricks-claude-sonnet-4-5"),
    ],
)
@pytest.mark.parametrize("signature_fields", [{"signature": ""}, {"signature": None}, {}])
def test_a_thought_another_provider_served_without_a_signature_never_reaches_claude(
    config, model, signature_fields
):
    """A load-balanced group's Gemini turn hands Claude Code a thought Anthropic cannot verify."""
    messages = _history_with_a_gemini_thought(signature_fields)
    request = config.transform_anthropic_messages_request(
        model=model,
        messages=messages,
        anthropic_messages_optional_request_params={
            "max_tokens": 8192,
            "thinking": {"type": "enabled", "budget_tokens": 4000},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request["messages"][1] == messages[1]
    assert request["messages"][3]["content"] == [{"type": "text", "text": "Two subsets."}]
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 4000}


@pytest.mark.parametrize(
    ("config", "model"),
    [
        (DeepSeekAnthropicMessagesConfig(), "deepseek-v4-pro"),
        (BedrockMantleAnthropicMessagesConfig(), "openai.gpt-oss-120b"),
    ],
)
def test_a_backend_that_does_not_verify_claude_signatures_gets_its_unsigned_thoughts_back(config, model):
    messages = _history_with_a_gemini_thought({"signature": ""})
    request = config.transform_anthropic_messages_request(
        model=model,
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 8192},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request["messages"][3]["content"] == messages[3]["content"]
