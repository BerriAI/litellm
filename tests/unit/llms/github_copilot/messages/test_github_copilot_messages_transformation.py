from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.exceptions import AuthenticationError, UnsupportedParamsError
from litellm.litellm_core_utils.prompt_templates.factory import EMPTY_TEXT_PLACEHOLDER
from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig
from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE, GetAccessTokenError
from litellm.llms.github_copilot.messages.transformation import (
    GithubCopilotAnthropicMessagesConfig,
)
from litellm.llms.github_copilot.github_host import github_host_endpoints
from litellm.llms.github_copilot.per_user_auth import (
    GITHUB_COPILOT_USER_SESSION_KWARG_KEY,
    GithubCopilotUserSession,
)
from litellm.types.router import GenericLiteLLMParams


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("in_tool_result", [False, True])
@pytest.mark.parametrize("model", ["claude-opus-5.5", "claude-haiku-5.5", "claude-fable-5.2"])
def test_text_attachment_is_readable_on_every_turn_without_mutating_history(
    stream: bool, in_tool_result: bool, model: str
) -> None:
    document: Final = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "<h1>Résumé &amp; café</h1>\n"},
        "title": "report.html",
        "context": "Uploaded source code",
        "citations": {"enabled": False},
        "cache_control": {"type": "ephemeral"},
    }
    pdf: Final = {"type": "document", "source": {"type": "url", "url": "https://example.com/report.pdf"}}
    image: Final = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}}
    blocks: Final = [document, pdf, image, {"type": "text", "text": "Explain this attachment."}]
    content: Final = [{"type": "tool_result", "tool_use_id": "tool_1", "content": blocks}] if in_tool_result else blocks
    attachment: Final = {"role": "user", "content": content}
    expected_blocks: Final = [
        {
            "type": "text",
            "text": "Title: report.html\n\nContext: Uploaded source code\n\n<h1>Résumé &amp; café</h1>\n",
            "cache_control": {"type": "ephemeral"},
        },
        *blocks[1:],
    ]
    expected_content: Final = (
        [{"type": "tool_result", "tool_use_id": "tool_1", "content": expected_blocks}]
        if in_tool_result
        else expected_blocks
    )
    for followups in (
        [],
        [{"role": "user", "content": "Retry after the failed attachment."}],
        [{"role": "assistant", "content": "Read it."}, {"role": "user", "content": "Summarize again."}],
    ):
        messages: Final = [attachment, *followups]
        original: Final = deepcopy(messages)
        result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 128, "stream": stream},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert result == {
            "model": model,
            "max_tokens": 128,
            "stream": stream,
            "messages": [{"role": "user", "content": expected_content}, *followups],
        }
        assert messages == original


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
        {"type": "url", "url": "https://example.com/report.pdf"},
        {"type": "file", "file_id": "file_123"},
        {"type": "content", "content": [{"type": "text", "text": "A custom document"}]},
        {"type": "text", "media_type": "text/plain", "data": None},
        {"type": "text", "media_type": "text/plain"},
    ],
)
def test_other_document_sources_keep_the_existing_wire_format(source: dict[str, object]) -> None:
    messages: Final = [{"role": "user", "content": [{"type": "document", "source": source}]}]
    result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["messages"] == messages


@pytest.mark.parametrize("text", ["<html>Keep <script> and whitespace.\n</html>", "  \n  indented\n  "])
def test_plain_document_text_is_not_parsed_or_base64_encoded(text: str) -> None:
    messages: Final = [{"role": "user", "content": [{"type": "document", "source": {"type": "text", "data": text}}]}]
    result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["messages"] == [{"role": "user", "content": [{"type": "text", "text": text}]}]


@pytest.mark.parametrize(
    ("data", "document_fields", "expected"),
    [
        ("", {}, [{"type": "text", "text": "What is in this file?"}]),
        (" \n\t ", {}, [{"type": "text", "text": "What is in this file?"}]),
        (
            "",
            {"title": "empty.txt"},
            [{"type": "text", "text": "Title: empty.txt"}, {"type": "text", "text": "What is in this file?"}],
        ),
    ],
)
def test_empty_text_document_never_becomes_an_empty_text_block(
    data: str, document_fields: dict[str, object], expected: list[dict[str, object]]
) -> None:
    document: Final = {"type": "document", "source": {"type": "text", "data": data}, **document_fields}
    messages: Final = [{"role": "user", "content": [document, {"type": "text", "text": "What is in this file?"}]}]
    result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["messages"] == [{"role": "user", "content": expected}]


@pytest.mark.parametrize("in_tool_result", [False, True])
@pytest.mark.parametrize("cache_fields", [{}, {"cache_control": {"type": "ephemeral"}}])
def test_content_of_only_empty_text_documents_stays_non_empty(
    cache_fields: dict[str, object], in_tool_result: bool
) -> None:
    document: Final = {"type": "document", "source": {"type": "text", "data": ""}}
    documents: Final = [document, {**document, **cache_fields}]
    placeholder: Final = [{"type": "text", "text": EMPTY_TEXT_PLACEHOLDER, **cache_fields}]
    messages: Final = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": documents}]
            if in_tool_result
            else documents,
        },
        {"role": "assistant", "content": "The file is empty."},
        {"role": "user", "content": "Thanks"},
    ]
    result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["messages"] == [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": placeholder}]
            if in_tool_result
            else placeholder,
        },
        *messages[1:],
    ]


_CACHED: Final = {"cache_control": {"type": "ephemeral"}}
_CACHED_EMPTY_DOCUMENT: Final = {"type": "document", "source": {"type": "text", "data": ""}, **_CACHED}
_QUESTION: Final = {"type": "text", "text": "What is in this file?"}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([_QUESTION, _CACHED_EMPTY_DOCUMENT], [{**_QUESTION, **_CACHED}]),
        ([_CACHED_EMPTY_DOCUMENT, _QUESTION], [{**_QUESTION, **_CACHED}]),
        (
            [{**_QUESTION, "cache_control": {"type": "ephemeral", "ttl": "1h"}}, _CACHED_EMPTY_DOCUMENT],
            [{**_QUESTION, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        ),
        (
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": [{"type": "text", "text": "Notes"}, _CACHED_EMPTY_DOCUMENT],
                },
                _QUESTION,
            ],
            [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": [{"type": "text", "text": "Notes", **_CACHED}]},
                _QUESTION,
            ],
        ),
    ],
)
def test_dropped_empty_document_keeps_its_cache_breakpoint(
    content: list[dict[str, object]], expected: list[dict[str, object]]
) -> None:
    result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=[{"role": "user", "content": content}],
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["messages"] == [{"role": "user", "content": expected}]


@pytest.mark.parametrize(
    "content",
    ["Hello", [{"type": "document", "source": {"type": "text", "data": "Source"}}, _QUESTION]],
)
def test_request_params_are_left_for_the_shared_handler_as_the_anthropic_transformer_leaves_them(
    content: object,
) -> None:
    def params_after(config: AnthropicMessagesConfig) -> dict[str, object]:
        params: Final[dict[str, object]] = {"max_tokens": 32000, "stream": True}
        config.transform_anthropic_messages_request(
            model="claude-sonnet-5.5",
            messages=[{"role": "user", "content": content}],
            anthropic_messages_optional_request_params=params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        return params

    assert params_after(GithubCopilotAnthropicMessagesConfig()) == params_after(AnthropicMessagesConfig())


def test_anthropic_keeps_text_documents_and_their_citations() -> None:
    messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "text", "data": "Source"}, "citations": {"enabled": True}}
            ],
        }
    ]
    result: Final = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["messages"] == messages


def test_pdf_citations_and_tool_inputs_are_not_rewritten() -> None:
    messages: Final = [
        {"role": "user", "content": "Read the PDF."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "read_file",
                    "input": {"type": "document", "source": {"type": "text", "data": "Tool input"}},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "Done"},
                {
                    "type": "document",
                    "source": {"type": "url", "url": "https://example.com/report.pdf"},
                    "citations": {"enabled": True},
                },
            ],
        },
    ]
    result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert result["messages"] == messages


@pytest.mark.parametrize("cited_source", ["text", "url"])
def test_text_document_fallback_does_not_silently_break_document_citations(cited_source: str) -> None:
    messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "text", "data": "Source"}},
                {
                    "type": "document",
                    "source": {"type": cited_source, "data": "Cited source", "url": "https://example.com/report.pdf"},
                    "citations": {"enabled": True},
                },
            ],
        }
    ]
    with pytest.raises(UnsupportedParamsError, match=r"text documents.*citations"):
        GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
            model="claude-sonnet-5.5",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 128},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


_CITED_TEXT_DOCUMENT: Final = {
    "type": "document",
    "source": {"type": "text", "data": "Source"},
    "citations": {"enabled": True},
}
_CONVERTED_TEXT_DOCUMENT: Final = {"type": "text", "text": "Source"}
_PDF: Final = {"type": "document", "source": {"type": "url", "url": "https://example.com/report.pdf"}}
_CITED_PDF: Final = {**_PDF, "citations": {"enabled": True}}
_REPLY: Final = {"role": "assistant", "content": "Read it."}


def _user(content: list[object]) -> dict[str, object]:
    return {"role": "user", "content": content}


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [_user([_CITED_TEXT_DOCUMENT, _CITED_PDF])],
            [_user([_CONVERTED_TEXT_DOCUMENT, _PDF])],
            id="pdf-after-converted-document",
        ),
        pytest.param(
            [_user([_CITED_TEXT_DOCUMENT]), _REPLY, _user([_CITED_PDF])],
            [_user([_CONVERTED_TEXT_DOCUMENT]), _REPLY, _user([_PDF])],
            id="pdf-in-a-later-turn",
        ),
        pytest.param(
            [_user([_CITED_PDF, _CITED_TEXT_DOCUMENT])],
            [_user([_CITED_PDF, _CONVERTED_TEXT_DOCUMENT])],
            id="pdf-before-converted-document",
        ),
        pytest.param(
            [_user([_CITED_PDF]), _REPLY, _user([_CITED_TEXT_DOCUMENT])],
            [_user([_CITED_PDF]), _REPLY, _user([_CONVERTED_TEXT_DOCUMENT])],
            id="pdf-in-an-earlier-turn",
        ),
    ],
)
def test_drop_params_drops_citations_only_when_converting_shifts_a_document_index(
    messages: Sequence[Mapping[str, object]], expected: Sequence[Mapping[str, object]]
) -> None:
    result: Final = GithubCopilotAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-5.5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 128},
        litellm_params=GenericLiteLLMParams(drop_params=True),
        headers={},
    )
    assert result["messages"] == expected


@pytest.mark.parametrize(
    ("resolved_base", "environment_base", "expected_base"),
    [
        (None, "https://copilot.example.test/", "https://copilot.example.test"),
        (None, "http://localhost:8080/", "http://localhost:8080"),
        ("https://resolved.example.test", "https://env.example.test", "https://resolved.example.test"),
        ("https://resolved.example.test/", "https://env.example.test", "https://resolved.example.test"),
        ("https://resolved.example.test/v1/messages", None, "https://resolved.example.test"),
    ],
)
def test_github_copilot_anthropic_messages_get_complete_url(
    monkeypatch: pytest.MonkeyPatch,
    resolved_base: str | None,
    environment_base: str | None,
    expected_base: str,
) -> None:
    if environment_base is None:
        monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_API_BASE", environment_base)
    config: Final = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()

    url: Final = config.get_complete_url(
        api_base=resolved_base,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )

    assert url == f"{expected_base}/v1/messages"
    assert config.authenticator.mock_calls == []


def test_github_copilot_anthropic_messages_url_falls_back_to_the_default_base_without_a_known_plan_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    config: Final = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "shared-github-token"

    url: Final = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )

    assert url == f"{DEFAULT_GITHUB_COPILOT_API_BASE}/v1/messages"


def test_github_copilot_anthropic_messages_validate_environment(monkeypatch: pytest.MonkeyPatch):
    """Test environment validation and header injection."""
    monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"

    headers = {}
    validated_headers, api_base = config.validate_anthropic_messages_environment(
        headers=headers,
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "copilot-integration-id" in validated_headers
    assert validated_headers["copilot-integration-id"] == "vscode-chat"
    assert validated_headers["Authorization"] == "Bearer gh.test-key-123"
    assert "anthropic-version" in validated_headers
    assert validated_headers["anthropic-version"] == "2023-06-01"
    # /v1/messages must use the messages-proxy intent so the Copilot backend
    # enables Anthropic-native features (context_management, thinking, etc.).
    assert validated_headers["openai-intent"] == "messages-proxy"
    assert validated_headers["x-interaction-type"] == "messages-proxy"
    assert validated_headers["x-github-api-version"] == "2026-06-01"
    assert api_base == DEFAULT_GITHUB_COPILOT_API_BASE


def test_github_copilot_anthropic_messages_validate_environment_injects_beta_headers():
    """Anthropic-beta headers must be auto-injected for advanced features
    (context_management, output_format, etc.) — matches the parent
    AnthropicMessagesConfig contract."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"output_format": {"type": "json_object"}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "anthropic-beta" in validated_headers
    assert "structured-outputs-2025-11-13" in validated_headers["anthropic-beta"]


def test_github_copilot_anthropic_messages_validate_environment_preserves_caller_anthropic_version():
    """Caller-supplied anthropic-version must be forwarded verbatim."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={"anthropic-version": "2024-10-22"},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["anthropic-version"] == "2024-10-22"


def test_github_copilot_anthropic_messages_validate_environment_injects_context_management_beta():
    """context_management in optional_params must trigger the corresponding
    anthropic-beta header so the Copilot backend accepts the field."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["openai-intent"] == "messages-proxy"
    assert validated_headers["x-interaction-type"] == "messages-proxy"
    assert "anthropic-beta" in validated_headers
    assert "context-management-2025-06-27" in validated_headers["anthropic-beta"]


@pytest.mark.usefixtures("local_beta_headers_config")
def test_github_copilot_safeguards_do_not_derive_the_dangerous_tool_use_beta() -> None:
    session: Final = GithubCopilotUserSession(token="gh.test-key", endpoints=github_host_endpoints("github.com"))

    validated_headers, _ = GithubCopilotAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "safeguards": ({"type": "dangerous_tool_use"},),
            "context_management": {"edits": ({"type": "clear_tool_uses_20250919"},)},
        },
        litellm_params={GITHUB_COPILOT_USER_SESSION_KWARG_KEY: session},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["anthropic-beta"] == "context-management-2025-06-27"


def test_github_copilot_anthropic_messages_validate_environment_auth_error():
    """Test error handling when authentication fails."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator to raise an error
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = GetAccessTokenError(
        status_code=401, message="No valid API key found"
    )

    with pytest.raises(AuthenticationError):
        config.validate_anthropic_messages_environment(
            headers={},
            model="github_copilot/claude-haiku-4.5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )


def test_github_copilot_anthropic_messages_supported_params():
    """Test supported parameters list."""
    config = GithubCopilotAnthropicMessagesConfig()
    params = config.get_supported_anthropic_messages_params("github_copilot/claude-haiku-4.5")

    # Should inherit from AnthropicMessagesConfig
    assert "messages" in params
    assert "model" in params
    assert "max_tokens" in params
    assert "thinking" in params


def test_provider_config_manager_dispatches_claude_to_copilot_messages_config():
    """ProviderConfigManager must return the Copilot Anthropic Messages config
    for Claude models served via github_copilot."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="github_copilot/claude-haiku-4.5",
        provider=LlmProviders.GITHUB_COPILOT,
    )

    assert isinstance(config, GithubCopilotAnthropicMessagesConfig)


def test_provider_config_manager_skips_non_claude_copilot_models():
    """Non-Claude github_copilot models (e.g. gpt-*) must not be routed through
    the Anthropic Messages dispatch."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="github_copilot/gpt-5-mini",
        provider=LlmProviders.GITHUB_COPILOT,
    )

    assert config is None


@pytest.mark.parametrize(
    ("deployment_base", "environment_base", "expected_base"),
    [
        (None, None, DEFAULT_GITHUB_COPILOT_API_BASE),
        (None, "https://copilot.example.test/", "https://copilot.example.test"),
        (None, "http://localhost:8080/", "http://localhost:8080"),
        ("https://deployment.example.test/", "https://copilot.example.test/", "https://deployment.example.test"),
    ],
)
def test_validate_environment_uses_the_deployment_base_else_the_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch, deployment_base: str | None, environment_base: str | None, expected_base: str
) -> None:
    if environment_base is None:
        monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_API_BASE", environment_base)
    config: Final = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=deployment_base,
    )

    assert api_base == expected_base
    assert headers["Authorization"] == "Bearer gh.test-key"
    assert (
        config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="github_copilot/claude-haiku-4.5",
            optional_params={},
            litellm_params={},
        )
        == f"{expected_base}/v1/messages"
    )


def test_github_copilot_config_disables_anthropic_beta_filtering():
    """Copilot's /v1/messages is a native Anthropic passthrough, so injected
    anthropic-beta values (context_management, structured outputs, ...) must be
    forwarded verbatim. The default provider-scoped filter would drop them
    because github_copilot has no entry in the beta headers config; a regression
    here would silently disable header-gated Anthropic features for Copilot."""
    from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
    from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig

    config = GithubCopilotAnthropicMessagesConfig()
    assert config.should_filter_anthropic_beta_headers() is False
    assert AzureAnthropicMessagesConfig().should_filter_anthropic_beta_headers() is True

    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "context-management-2025-06-27" in headers["anthropic-beta"]

    # The override is load-bearing: had the config opted into the provider-scoped
    # filter, the handler would have run it and dropped every value, since
    # github_copilot has no mapping. Prove that here so a regression that flips
    # should_filter back on is caught as the silent feature breakage it causes.
    stripped = update_headers_with_filtered_beta(headers={**headers}, provider="github_copilot")
    assert "anthropic-beta" not in stripped


def test_github_copilot_config_does_not_handle_web_search_natively():
    """Copilot's /v1/messages does not run web_search, so its config must report
    handles_web_search_natively() == False. This is what keeps the web-search
    interception handler short-circuiting Copilot instead of routing to it, even
    though Copilot now has a BaseAnthropicMessagesConfig. The base Anthropic
    config (bedrock/vertex/anthropic path) must report True."""
    from litellm.llms.anthropic.messages.native.transformation import (
        AnthropicMessagesConfig,
    )

    assert GithubCopilotAnthropicMessagesConfig().handles_web_search_natively() is False
    assert AnthropicMessagesConfig().handles_web_search_natively() is True


def test_github_copilot_messages_config_probes_capabilities_under_copilot_namespace():
    """Capability probes in the shared pass-through helpers read
    ``self.custom_llm_provider``; without this override they probed the
    ``anthropic`` namespace and ignored the exact ``github_copilot/claude-*``
    cost-map entries."""
    assert GithubCopilotAnthropicMessagesConfig().custom_llm_provider == "github_copilot"


@pytest.mark.parametrize(
    ("deployment_base", "environment_base", "expected_base"),
    [
        (None, None, DEFAULT_GITHUB_COPILOT_API_BASE),
        (None, "http://localhost:8080/", "http://localhost:8080"),
        ("https://deployment.example.test", "http://localhost:8080/", "http://localhost:8080"),
        ("https://deployment.example.test", None, DEFAULT_GITHUB_COPILOT_API_BASE),
    ],
)
def test_validate_environment_uses_per_user_session_and_skips_authenticator(
    monkeypatch: pytest.MonkeyPatch, deployment_base: str | None, environment_base: str | None, expected_base: str
) -> None:
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    if environment_base is None:
        monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_API_BASE", environment_base)

    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = AssertionError("shared authenticator must not run")

    session = GithubCopilotUserSession(token="user-copilot-token", endpoints=github_host_endpoints("github.com"))
    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-sonnet-5.5",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
        api_base=deployment_base,
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    assert api_base == expected_base
    assert config.authenticator.mock_calls == []
    assert (
        config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="github_copilot/claude-sonnet-5.5",
            optional_params={},
            litellm_params={"github_copilot_user_session": session},
        )
        == f"{expected_base}/v1/messages"
    )


@pytest.mark.parametrize("authorization_header", ["Authorization", "authorization", "AUTHORIZATION"])
def test_per_user_session_token_wins_over_caller_authorization(authorization_header: str):
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    session = GithubCopilotUserSession(token="user-copilot-token", endpoints=github_host_endpoints("github.com"))
    headers, _ = config.validate_anthropic_messages_environment(
        headers={authorization_header: "Bearer caller-token", "x-custom": "preserved"},
        model="github_copilot/claude-sonnet-4.5",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    assert {key: value for key, value in headers.items() if key.lower() == "authorization"} == {
        "Authorization": "Bearer user-copilot-token"
    }
    assert headers["x-custom"] == "preserved"
    assert config.authenticator.mock_calls == []


def test_transform_response_carries_upstream_usage():
    config = GithubCopilotAnthropicMessagesConfig()
    raw = httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "github_copilot/claude-sonnet-5.5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 14, "output_tokens": 3},
        },
    )
    result = config.transform_anthropic_messages_response(
        model="github_copilot/claude-sonnet-5.5",
        raw_response=raw,
        logging_obj=MagicMock(),
    )
    assert result["usage"]["input_tokens"] == 14
    assert result["usage"]["output_tokens"] == 3
