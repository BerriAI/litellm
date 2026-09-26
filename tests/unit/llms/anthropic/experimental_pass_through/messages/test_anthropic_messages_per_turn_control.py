import pytest

from litellm import anthropic_beta_headers_manager
from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.openai_like.json_loader import SimpleProviderConfig
from litellm.llms.openai_like.messages.transformation import (
    JSONProviderAnthropicMessagesConfig,
)

PER_TURN_CONTROL = "per-turn-control-2026-07-01"

CLAUDE_CODE_BETAS = (
    "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,"
    "per-turn-control-2026-07-01,effort-2025-11-24"
)


def _claude_code_turn(system_output_config):
    return [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {
            "role": "system",
            "content": [{"type": "text", "text": "# Environment"}],
            "output_config": system_output_config,
        },
    ]


def _betas(headers):
    return {beta for beta in headers.get("anthropic-beta", "").split(",") if beta}


def _validate(messages, headers=None, optional_params=None):
    validated, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers=dict(headers or {}),
        model="claude-fable-5-1",
        messages=messages,
        optional_params=dict(optional_params or {"max_tokens": 64000, "output_config": {"effort": "high"}}),
        litellm_params={},
        api_key="sk-ant-test",
    )
    return validated


@pytest.fixture(autouse=True)
def bundled_beta_allowlist(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    monkeypatch.setattr(anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None)
    yield
    monkeypatch.setattr(anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None)


def test_per_message_output_config_adds_per_turn_control_beta():
    headers = _validate(_claude_code_turn({"effort": "high"}))

    assert PER_TURN_CONTROL in _betas(headers)


def test_top_level_output_config_alone_does_not_add_per_turn_control_beta():
    headers = _validate([{"role": "user", "content": "Hello"}])

    assert PER_TURN_CONTROL not in _betas(headers)


def test_string_messages_are_skipped_when_scanning_for_output_config():
    headers = _validate(["not a message dict", {"role": "user", "content": "Hello"}])

    assert PER_TURN_CONTROL not in _betas(headers)


def test_forwarded_client_betas_survive_alongside_the_added_one():
    headers = _validate(_claude_code_turn({"effort": "low"}), headers={"anthropic-beta": CLAUDE_CODE_BETAS})

    assert _betas(headers) >= set(CLAUDE_CODE_BETAS.split(","))
    assert PER_TURN_CONTROL in _betas(headers)


def test_case_variant_client_beta_header_is_merged():
    headers = _validate(
        _claude_code_turn({"effort": "low"}), headers={"Anthropic-Beta": "interleaved-thinking-2025-05-14"}
    )

    assert [key for key in headers if key.lower() == "anthropic-beta"] == ["anthropic-beta"]
    assert _betas(headers) == {"interleaved-thinking-2025-05-14", PER_TURN_CONTROL}


def test_added_per_turn_control_beta_survives_the_anthropic_allowlist():
    headers = _validate(_claude_code_turn({"effort": "high"}))

    filtered = update_headers_with_filtered_beta(headers=headers, provider="anthropic")

    assert PER_TURN_CONTROL in _betas(filtered)


@pytest.mark.parametrize("provider", ["bedrock", "bedrock_converse", "vertex_ai", "databricks"])
def test_per_turn_control_beta_is_dropped_for_providers_without_it(provider):
    filtered = update_headers_with_filtered_beta(headers={"anthropic-beta": PER_TURN_CONTROL}, provider=provider)

    assert "anthropic-beta" not in filtered


def test_per_turn_control_beta_is_forwarded_for_azure_ai():
    # Azure AI Foundry supports the per-turn-control beta, see #42959.
    filtered = update_headers_with_filtered_beta(headers={"anthropic-beta": PER_TURN_CONTROL}, provider="azure_ai")

    assert PER_TURN_CONTROL in _betas(filtered)


def test_json_provider_passthrough_adds_per_turn_control_beta():
    config = JSONProviderAnthropicMessagesConfig(
        SimpleProviderConfig(
            "anthropic_like",
            {
                "base_url": "https://example.invalid",
                "api_key_env": "ANTHROPIC_LIKE_API_KEY",
                "supported_endpoints": ["/v1/messages"],
            },
        )
    )
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="claude-fable-5-1",
        messages=_claude_code_turn({"effort": "medium"}),
        optional_params={"max_tokens": 1024},
        litellm_params={},
        api_key="test",
    )

    assert PER_TURN_CONTROL in _betas(headers)
