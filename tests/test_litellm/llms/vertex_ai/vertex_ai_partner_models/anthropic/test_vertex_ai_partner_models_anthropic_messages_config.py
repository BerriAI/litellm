import copy
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)
from litellm.llms.vertex_ai.vertex_ai_partner_models.main import VertexAIPartnerModels
from litellm.types.router import GenericLiteLLMParams


def test_validate_environment_uses_vertex_ai_location():
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "europe-west1",
        "vertex_credentials": "{}",
    }
    optional_params = {}

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-url"
        ) as mock_get_url,
    ):
        config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-sonnet",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        assert mock_get_url.call_args.kwargs["vertex_location"] == "europe-west1"


def test_web_search_header_added_for_messages_endpoint():
    """Test that web search tool adds the required beta header for Vertex AI /v1/messages requests"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include web search tool in optional_params
    optional_params = {
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    }

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-url"
        ),
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )

        # Assert that the anthropic-beta header with web-search is present
        assert (
            "anthropic-beta" in updated_headers
        ), "anthropic-beta header should be present"
        assert (
            updated_headers["anthropic-beta"] == "web-search-2025-03-05"
        ), f"anthropic-beta should be 'web-search-2025-03-05', got: {updated_headers['anthropic-beta']}"


def test_web_search_header_not_added_without_tool():
    """Test that beta header is NOT added when web search tool is not present"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # No web search tool
    optional_params = {}

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-url"
        ),
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )

        # Assert that the anthropic-beta header is NOT present when no web search tool
        assert (
            "anthropic-beta" not in updated_headers
        ), "anthropic-beta header should not be present without web search tool"


def test_compact_context_management_header_added():
    """Test that compact-2026-01-12 beta header is added when context_management with compact_20260112 is used"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include context_management with compact_20260112
    optional_params = {"context_management": {"edits": [{"type": "compact_20260112"}]}}

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-url"
        ),
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-vertex-ai-opus-4-6",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )

        # Assert that the anthropic-beta header with compact-2026-01-12 is present
        assert (
            "anthropic-beta" in updated_headers
        ), "anthropic-beta header should be present"
        assert (
            "compact-2026-01-12" in updated_headers["anthropic-beta"]
        ), f"anthropic-beta should contain 'compact-2026-01-12', got: {updated_headers['anthropic-beta']}"


def test_context_management_header_added_for_other_edits():
    """Test that context-management-2025-06-27 beta header is added for non-compact edits"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include context_management with other edit types
    optional_params = {"context_management": {"edits": [{"type": "some_other_type"}]}}

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-url"
        ),
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-vertex-ai-opus-4-6",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )

        # Assert that the anthropic-beta header with context-management-2025-06-27 is present
        assert (
            "anthropic-beta" in updated_headers
        ), "anthropic-beta header should be present"
        assert (
            "context-management-2025-06-27" in updated_headers["anthropic-beta"]
        ), f"anthropic-beta should contain 'context-management-2025-06-27', got: {updated_headers['anthropic-beta']}"


def test_both_compact_and_context_management_headers_added():
    """Test that both compact and context-management beta headers are added when both edit types are present"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include context_management with both compact and other edit types
    optional_params = {
        "context_management": {
            "edits": [{"type": "compact_20260112"}, {"type": "some_other_type"}]
        }
    }

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-url"
        ),
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-vertex-ai-opus-4-6",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )

        # Assert that both beta headers are present
        assert (
            "anthropic-beta" in updated_headers
        ), "anthropic-beta header should be present"
        assert (
            "compact-2026-01-12" in updated_headers["anthropic-beta"]
        ), f"anthropic-beta should contain 'compact-2026-01-12', got: {updated_headers['anthropic-beta']}"
        assert (
            "context-management-2025-06-27" in updated_headers["anthropic-beta"]
        ), f"anthropic-beta should contain 'context-management-2025-06-27', got: {updated_headers['anthropic-beta']}"


def test_validate_environment_always_refreshes_token_ignoring_stale_bearer():
    """Regression: stale Authorization in shared deployment extra_headers must not
    skip token refresh on /v1/messages — _ensure_access_token is always called."""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {"Authorization": "Bearer EXPIRED"}
    litellm_params = {
        "vertex_project": "test-project",
        "vertex_location": "us-central1",
    }

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("fresh-token", "test-project")
        ) as mock_ensure,
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-vertex-url"
        ),
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
            api_base=None,
        )

        mock_ensure.assert_called_once()
        assert updated_headers["Authorization"] == "Bearer fresh-token"
        assert api_base == "https://mock-vertex-url"


def test_validate_environment_appends_stream_raw_predict_with_custom_api_base():
    """Regression: a custom api_base on /v1/messages must still get the endpoint
    suffix appended. The old `if api_base is None` guard skipped
    get_complete_vertex_url entirely, leaving the api_base without
    `:streamRawPredict`."""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
    }

    with (
        patch.object(
            config,
            "get_complete_vertex_url",
            wraps=config.get_complete_vertex_url,
        ) as spy_get_url,
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
    ):
        _, api_base = config.validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4",
            messages=[],
            optional_params={"stream": True},
            litellm_params=litellm_params,
            api_base="https://my-proxy.example.com",
        )

    spy_get_url.assert_called_once()
    assert api_base is not None
    assert ":streamRawPredict" in api_base


def test_validate_environment_appends_raw_predict_with_custom_api_base():
    """Regression: non-streaming custom api_base must end with `:rawPredict`."""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
    }

    with (
        patch.object(
            config,
            "get_complete_vertex_url",
            wraps=config.get_complete_vertex_url,
        ) as spy_get_url,
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
    ):
        _, api_base = config.validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
            api_base="https://my-proxy.example.com",
        )

    spy_get_url.assert_called_once()
    assert api_base is not None
    assert api_base.endswith(":rawPredict")


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


def test_provider_config_manager_reuses_vertex_anthropic_messages_config_instance():
    """
    Regression test: repeated provider config lookups for the same Vertex Claude model
    should return the same config instance (which preserves auth cache state).
    """
    import litellm
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


def test_validate_environment_does_not_mutate_caller_headers():
    """Regression: beta headers (e.g. web-search) must not leak into the caller's
    headers dict — which may be the shared deployment extra_headers object."""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    caller_headers: dict = {}

    with (
        patch.object(
            config, "_ensure_access_token", return_value=("token", "test-project")
        ),
        patch.object(
            config, "get_complete_vertex_url", return_value="https://mock-url"
        ),
    ):
        config.validate_anthropic_messages_environment(
            headers=caller_headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={
                "tools": [{"type": "web_search_20250305", "name": "web_search"}]
            },
            litellm_params={
                "vertex_ai_project": "p",
                "vertex_ai_location": "us-central1",
            },
            api_base=None,
        )

    assert (
        caller_headers == {}
    ), "validate_anthropic_messages_environment must not mutate the caller's headers dict"


def test_vertex_claude_completion_does_not_mutate_shared_extra_headers():
    """Regression: router shallow-copies litellm_params so extra_headers is a shared
    reference. Verify that the chat/completions path builds a new headers dict instead
    of calling .update() on the shared object."""
    handler = VertexAIPartnerModels()
    shared_extra_headers = {}  # simulates deployment["litellm_params"]["extra_headers"]

    mock_response = MagicMock()

    with (
        patch.object(
            handler, "_ensure_access_token", return_value=("ya29.fresh", "proj")
        ),
        patch.object(
            handler, "get_complete_vertex_url", return_value="https://mock-url"
        ),
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

    assert (
        shared_extra_headers == {}
    ), "extra_headers must not be mutated by completion()"


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled backup cost map so capability flags match this branch."""
    import litellm

    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


def test_messages_thinking_shape_follows_exact_vertex_entry_flag(local_model_cost_map, monkeypatch):
    """The Vertex messages config must probe capabilities under ``vertex_ai`` so an
    operator setting ``supports_adaptive_thinking: false`` on the exact
    ``vertex_ai/claude-opus-4-8`` entry beats the unmodified ``anthropic`` entry.
    With the inherited ``"anthropic"`` provider default the flip was ignored and
    the transform kept emitting ``thinking.type='adaptive'``."""
    import litellm

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
    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": "medium"}

    monkeypatch.setitem(
        litellm.model_cost["vertex_ai/claude-opus-4-8"], "supports_adaptive_thinking", False
    )
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

    def test_unsupported_model_hoists_mid_conversation_system(self, local_model_cost_map):
        messages = [
            {"role": "user", "content": "read the file"},
            {"role": "system", "content": "[Truncated: PARTIAL view of big1.txt]"},
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        result = _vertex_transform(
            "claude-sonnet-4-6", messages, system=[{"type": "text", "text": "Base."}]
        )
        assert result["messages"] == [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        assert result["system"] == [
            {"type": "text", "text": "Base."},
            {"type": "text", "text": "[Truncated: PARTIAL view of big1.txt]"},
        ]


def test_vertex_claude_4_8_plus_cost_map_entries_carry_mid_conversation_system_flag():
    """Exact cost-map hits win over the ``claude-mid-conversation-system``
    fallback rule, so a ``vertex_ai`` Claude 4.8+/5 entry missing the flag would
    be treated as unsupported and hoist every reminder, collapsing the prompt
    cache. Every mapped vertex_ai entry the rule matches must carry the flag."""
    import re

    import litellm

    cost_map_path = os.path.join(
        os.path.dirname(litellm.__file__), "model_prices_and_context_window_backup.json"
    )
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
