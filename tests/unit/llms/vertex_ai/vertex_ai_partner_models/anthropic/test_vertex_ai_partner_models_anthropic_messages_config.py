import copy
import json
import os
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.messages.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)
from litellm.llms.vertex_ai.vertex_ai_partner_models.main import VertexAIPartnerModels
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES
from litellm.types.router import GenericLiteLLMParams


def test_transform_anthropic_messages_request_removes_scope_from_cache_control():
    """Ensure scope field is removed from cache_control for Vertex AI (not supported)."""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                    "cache_control": {"type": "ephemeral", "scope": "global"},
                }
            ],
        }
    ]
    anthropic_messages_optional_request_params = {
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": "You are an AI assistant.",
                "cache_control": {"type": "ephemeral", "scope": "global"},
            }
        ],
    }

    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=messages,
        anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    # scope removed from system
    assert "scope" not in result["system"][0]["cache_control"]
    assert result["system"][0]["cache_control"]["type"] == "ephemeral"

    # scope removed from message content
    assert "scope" not in result["messages"][0]["content"][0]["cache_control"]
    assert result["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"


def test_messages_request_strips_effort_for_haiku_45():
    """Regression: Claude Code (``claude --model claude-haiku-4.5``) sends
    ``output_config.effort`` in its default Messages payload. Haiku 4.5 on
    Vertex rejects it with 400 ``output_config.effort: Extra inputs are not
    permitted``, so the pass-through must strip it for Haiku while keeping it
    for Opus/Sonnet 4.6+."""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    messages = [{"role": "user", "content": "Hello"}]

    haiku_result = config.transform_anthropic_messages_request(
        model="claude-haiku-4-5@20251001",
        messages=messages,
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "output_config": {"effort": "high"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "output_config" not in haiku_result

    opus_result = config.transform_anthropic_messages_request(
        model="claude-opus-4-6",
        messages=messages,
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "output_config": {"effort": "high"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert opus_result["output_config"] == {"effort": "high"}


def test_messages_request_moves_output_format_into_output_config():
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    output_format: Final = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]},
    }

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "What is 6 times 7?"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "output_format": output_format,
            "output_config": {"effort": "high"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "output_format" not in result
    assert result["output_config"] == {"effort": "high", "format": output_format}


_CALLER_OUTPUT_CONFIG: Final = {
    "effort": "high",
    "format": {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}}},
}


def _messages_request_with_output_format(output_format: object, drop_params: bool) -> Mapping[str, object]:
    return VertexAIPartnerModelsAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "What is 6 times 7?"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "output_format": output_format,
            "output_config": copy.deepcopy(_CALLER_OUTPUT_CONFIG),
        },
        litellm_params=GenericLiteLLMParams(drop_params=drop_params),
        headers={},
    )


@pytest.mark.parametrize("output_format", ["json", ["json_schema"], 42])
def test_messages_request_drops_a_malformed_output_format_under_drop_params(output_format: object):
    result = _messages_request_with_output_format(output_format, drop_params=True)

    assert "output_format" not in result
    assert result["output_config"] == _CALLER_OUTPUT_CONFIG


@pytest.mark.parametrize("output_format", ["json", ["json_schema"], 42])
def test_messages_request_rejects_a_malformed_output_format_without_drop_params(output_format: object):

    with pytest.raises(litellm.UnsupportedParamsError) as error:
        _messages_request_with_output_format(output_format, drop_params=False)

    assert error.value.status_code == 400


def test_provider_config_manager_reuses_vertex_anthropic_messages_config_instance():
    """
    Regression test: repeated provider config lookups for the same Vertex Claude model
    should return the same config instance (which preserves auth cache state).
    """
    from litellm.utils import ProviderConfigManager

    ProviderConfigManager._get_provider_anthropic_messages_config_cached.cache_clear()
    try:
        first_config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-opus-4-6",
            provider=litellm.LlmProviders.VERTEX_AI,
        )
        second_config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-opus-4-6",
            provider=litellm.LlmProviders.VERTEX_AI,
        )

        assert isinstance(first_config, VertexAIPartnerModelsAnthropicMessagesConfig)
        assert isinstance(second_config, VertexAIPartnerModelsAnthropicMessagesConfig)
        assert first_config is second_config
    finally:
        ProviderConfigManager._get_provider_anthropic_messages_config_cached.cache_clear()


def test_vertex_claude_completion_does_not_mutate_shared_extra_headers():
    """Regression: router shallow-copies litellm_params so extra_headers is a shared
    reference. Verify that the chat/completions path builds a new headers dict instead
    of calling .update() on the shared object."""
    handler = VertexAIPartnerModels()
    shared_extra_headers = {}  # simulates deployment["litellm_params"]["extra_headers"]

    mock_response = MagicMock()

    with (
        patch.object(handler, "_ensure_access_token", return_value=("ya29.fresh", "proj")),
        patch.object(handler, "get_complete_vertex_url", return_value="https://mock-url"),
        patch(
            "litellm.llms.anthropic.chat.AnthropicChatCompletion.completion",
            return_value=mock_response,
        ),
    ):
        handler.completion(
            model="claude-haiku-4-5@20251001",
            messages=[{"role": "user", "content": "hi"}],
            model_response=MagicMock(),
            print_verbose=lambda *a, **k: None,
            encoding=None,
            logging_obj=MagicMock(),
            api_base=None,
            optional_params={},
            custom_prompt_dict={},
            headers=shared_extra_headers,
            timeout=30,
            litellm_params={},
        )

    assert shared_extra_headers == {}, "extra_headers must not be mutated by completion()"


def test_messages_thinking_shape_follows_exact_vertex_entry_flag(local_model_cost_map, monkeypatch):
    """The Vertex messages config must probe capabilities under ``vertex_ai`` so an
    operator setting ``supports_adaptive_thinking: false`` on the exact
    ``vertex_ai/claude-opus-4-8`` entry beats the unmodified ``anthropic`` entry.
    With the inherited ``"anthropic"`` provider default the flip was ignored and
    the transform kept emitting ``thinking.type='adaptive'``."""

    config = VertexAIPartnerModelsAnthropicMessagesConfig()

    def transform():
        return config.transform_anthropic_messages_request(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_optional_request_params={
                "max_tokens": 4096,
                "reasoning_effort": "medium",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    result = transform()
    assert result.get("thinking") == {"type": "adaptive", "display": "summarized"}
    assert result.get("output_config") == {"effort": "medium"}

    monkeypatch.setitem(litellm.model_cost["vertex_ai/claude-opus-4-8"], "supports_adaptive_thinking", False)
    litellm.get_model_info.cache_clear()
    assert litellm.model_cost["claude-opus-4-8"]["supports_adaptive_thinking"] is True

    flipped = transform()
    thinking = flipped.get("thinking")
    assert isinstance(thinking, dict)
    assert thinking.get("type") == "enabled"
    assert isinstance(thinking.get("budget_tokens"), int)
    assert "output_config" not in flipped


def _vertex_transform(model, messages, system=None):
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    params = {"max_tokens": 256}
    if system is not None:
        params["system"] = system
    return config.transform_anthropic_messages_request(
        model=model,
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


class TestVertexAnthropicMidConversationSystem:
    """Vertex serves Claude on the first-party Anthropic /v1/messages contract: a
    mid-conversation ``role: "system"`` reminder is accepted in place on Claude
    4.8+/5 but 400s ("role 'system' is not supported on this model") on older
    Claude, and a *leading* system entry 400s on every model ("messages.0: use
    the top-level 'system' parameter"). These tests pin the model-aware hoist so
    Claude Code sessions neither collapse the prompt cache on 4.8+ nor hard-fail
    on 4.7 and older (RCA: customer high-spend)."""

    def test_supported_model_keeps_mid_conversation_system_in_place(self, local_model_cost_map):
        messages = [
            {"role": "user", "content": "read the file"},
            {"role": "system", "content": "[Truncated: PARTIAL view of big1.txt]"},
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        result = _vertex_transform("claude-opus-4-8", messages)
        assert result["messages"] == messages

    def test_supported_model_hoists_only_leading_system_run(self, local_model_cost_map):
        messages = [
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": "Cite sources."},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "mid-conversation reminder"},
            {"role": "user", "content": "continue"},
        ]
        result = _vertex_transform("claude-opus-4-8", messages)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "mid-conversation reminder"},
            {"role": "user", "content": "continue"},
        ]
        assert result["system"] == [
            {"type": "text", "text": "You are terse."},
            {"type": "text", "text": "Cite sources."},
        ]

    @pytest.mark.parametrize("drop_params", (False, True))
    def test_unsupported_model_sends_a_non_object_message_on_for_the_provider_to_reject(
        self, local_model_cost_map, local_beta_headers_config, drop_params
    ):
        messages: Final = [{"role": "user", "content": "hi"}, "oops"]
        result: Final = VertexAIPartnerModelsAnthropicMessagesConfig().transform_anthropic_messages_request(
            model="claude-sonnet-4-6",
            messages=list(messages),
            anthropic_messages_optional_request_params={"max_tokens": 64},
            litellm_params=GenericLiteLLMParams(drop_params=drop_params),
            headers={},
        )
        assert result["messages"] == messages

    def test_unsupported_model_converts_mid_conversation_system_in_place(self, local_model_cost_map):
        messages = [
            {"role": "user", "content": "read the file"},
            {"role": "system", "content": "[Truncated: PARTIAL view of big1.txt]"},
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        result = _vertex_transform("claude-sonnet-4-6", messages, system=[{"type": "text", "text": "Base."}])
        assert result["messages"] == [
            {"role": "user", "content": "read the file"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Operator note (not from the user): the following was "
                            "originally a mid-conversation system-role reminder."
                        ),
                    },
                    {"type": "text", "text": "[Truncated: PARTIAL view of big1.txt]"},
                ],
            },
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        assert result["system"] == [{"type": "text", "text": "Base."}]

    def test_unsupported_model_still_hoists_leading_system_run(self, local_model_cost_map):
        messages = [
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": "Cite sources."},
            {"role": "user", "content": "hi"},
        ]
        result = _vertex_transform("claude-sonnet-4-6", messages)
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["system"] == [
            {"type": "text", "text": "You are terse."},
            {"type": "text", "text": "Cite sources."},
        ]


def test_vertex_claude_4_8_plus_cost_map_entries_carry_mid_conversation_system_flag():
    """Exact cost-map hits win over the ``claude-mid-conversation-system``
    fallback rule, so a ``vertex_ai`` Claude 4.8+/5 entry missing the flag would
    be treated as unsupported and hoist every reminder, collapsing the prompt
    cache. Every mapped vertex_ai entry the rule matches must carry the flag."""
    import re

    cost_map_path = os.path.join(os.path.dirname(litellm.__file__), "model_prices_and_context_window_backup.json")
    with open(cost_map_path) as f:
        cost_map = json.load(f)
    rules = cost_map["fallback_generalizations"]["rules"]
    rule_pattern = next(
        (r["pattern"] for r in rules if r["name"] == "claude-mid-conversation-system"),
        None,
    )
    assert rule_pattern is not None, "claude-mid-conversation-system rule not found in fallback_generalizations"
    pattern = re.compile(rule_pattern, re.IGNORECASE)
    missing = [
        key
        for key, info in cost_map.items()
        if isinstance(info, dict)
        and str(info.get("litellm_provider", "")).startswith("vertex_ai")
        and "claude" in key
        and pattern.search(key)
        and info.get("supports_mid_conversation_system") is not True
    ]
    assert missing == []


@pytest.mark.usefixtures("local_model_cost_map", "local_beta_headers_config")
@pytest.mark.parametrize("feature", ("thinking", "output_config", "tool_addition", "tool_removal", "none"))
@pytest.mark.parametrize("explicit_beta", (False, True))
def test_messages_feature_beta_headers(feature: str, explicit_beta: bool) -> None:
    from typing import Final
    from unittest.mock import patch

    from google.oauth2.credentials import Credentials

    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.messages.transformation import (
        VertexAIPartnerModelsAnthropicMessagesConfig,
    )
    from litellm.types.llms.anthropic import (
        ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER,
        ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER,
    )

    beta: Final = (
        ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER
        if feature == "thinking"
        else ANTHROPIC_BETA_HEADER_VALUES.PER_TURN_CONTROL_2026_07_01.value
        if feature == "output_config"
        else ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER
    )
    message: Final = (
        {"role": "system", "content": [], "output_config": {"effort": "high"}}
        if feature == "output_config"
        else {"role": "system", "content": [{"type": feature, "tool": {"type": "tool_reference", "name": "ping"}}]}
        if feature in ("tool_addition", "tool_removal")
        else {"role": "user", "content": "Reply with OK"}
    )
    with patch(
        "google.auth.default",
        return_value=(MagicMock(spec=Credentials, token="test-token", expired=False), "test-project"),
    ):
        headers, _ = VertexAIPartnerModelsAnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers={"anthropic-beta": f"existing-beta,{beta}" if explicit_beta else "existing-beta"},
            model="claude-fable-5-1",
            messages=[{"role": "user", "content": "Hello"}, message],
            optional_params={"thinking": {"type": "adaptive", "display": "updates"}} if feature == "thinking" else {},
            litellm_params={"vertex_ai_project": "test-project", "vertex_ai_location": "us-east5"},
            api_key="test-key",
            api_base="https://example.com/anthropic",
        )

    betas: Final = headers["anthropic-beta"].split(",")
    assert betas.count(beta) == int(feature != "none" or explicit_beta)
    assert betas.count("existing-beta") == 1

    if feature in ("tool_addition", "tool_removal"):
        filtered_headers: Final = update_headers_with_filtered_beta(headers=headers, provider="vertex_ai")
        assert filtered_headers.get("anthropic-beta", "").split(",").count(beta) == 1


@pytest.mark.parametrize("thinking", ("enabled", {"type": "bogus"}, {"type": "adaptive", "display": "future-mode"}))
def test_beta_detection_preserves_unrecognized_thinking(thinking: object) -> None:
    from typing import Final
    from unittest.mock import patch

    from google.oauth2.credentials import Credentials

    from litellm.types.llms.anthropic import ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER

    params: Final = {"thinking": thinking}
    with patch(
        "google.auth.default",
        return_value=(MagicMock(spec=Credentials, token="test-token", expired=False), "test-project"),
    ):
        headers, _ = VertexAIPartnerModelsAnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers={},
            model="claude-fable-5-1",
            messages=[],
            optional_params=params,
            litellm_params={"vertex_ai_project": "test-project", "vertex_ai_location": "us-east5"},
            api_key="test-key",
            api_base="https://example.com/anthropic",
        )

    assert params["thinking"] == thinking
    assert ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER not in headers.get("anthropic-beta", "").split(",")
