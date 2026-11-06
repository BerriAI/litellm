"""Regression tests for #49062: caller headers must not wipe the provider's own
request headers on the chat completions path.

Two ways the headers got lost, both covered here:

* the copilot editor-version set that main.py folds into
  optional_params["extra_headers"] never reached the wire when the caller also
  sent headers=, because the openai SDK handler assigned over the merged dict
  instead of merging. That takes the copilot headers out as request headers and
  as the provider's auth set, so github answers 400 "missing Editor-Version"
* the generic http handler ignored optional_params["extra_headers"] entirely and
  only looked at headers=, so every provider routed through it lost the
  provider-specific headers main.py had collected

No sockets are opened; respx intercepts the outbound request.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import litellm

API_BASE = "https://api.githubcopilot.com"

CHAT_COMPLETION_BODY = {
    "id": "chatcmpl-headers",
    "object": "chat.completion",
    "created": 0,
    "model": "claude-sonnet-4",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

COPILOT_HEADERS = {"editor-version": "vscode/1.95.0", "copilot-integration-id": "vscode-chat"}


@pytest.fixture
def copilot_endpoint():
    with respx.mock(assert_all_called=False) as mock:
        yield mock.post(url__regex=rf"{API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=CHAT_COMPLETION_BODY)
        )


@pytest.fixture
def no_copilot_login(monkeypatch):
    """Skip the on-disk copilot login; the headers are what is under test."""
    from litellm.llms.github_copilot.authenticator import Authenticator

    monkeypatch.setattr(Authenticator, "get_api_key", lambda self: "ghu_fake_token")


@respx.mock
def test_extra_headers_survive_a_forwarded_client_header(copilot_endpoint, no_copilot_login):
    """headers= is what the proxy fills in when it forwards client headers."""
    litellm.completion(
        model="github_copilot/claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        api_key="ghu_fake_token",
        api_base=API_BASE,
        headers={"x-litellm-session-id": "sess_123"},
    )

    sent = copilot_endpoint.calls.last.request.headers
    assert sent["editor-version"] == "vscode/1.95.0"
    assert sent["copilot-integration-id"] == "vscode-chat"
    assert sent["x-litellm-session-id"] == "sess_123"


@respx.mock
def test_extra_headers_survive_a_second_extra_header(copilot_endpoint, no_copilot_login):
    """A caller adding one header of its own keeps the provider's set intact."""
    litellm.completion(
        model="github_copilot/claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        api_key="ghu_fake_token",
        api_base=API_BASE,
        extra_headers={"x-corp-llm-call-id": "abc-123"},
    )

    sent = copilot_endpoint.calls.last.request.headers
    assert sent["editor-version"] == "vscode/1.95.0"
    assert sent["x-corp-llm-call-id"] == "abc-123"


@respx.mock
def test_caller_can_still_override_editor_version(copilot_endpoint, no_copilot_login):
    litellm.completion(
        model="github_copilot/claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        api_key="ghu_fake_token",
        api_base=API_BASE,
        extra_headers={"Editor-Version": "vscode/9.9.9"},
    )

    assert copilot_endpoint.calls.last.request.headers["editor-version"] == "vscode/9.9.9"


@patch("litellm.llms.openai.openai.OpenAIChatCompletion.make_sync_openai_chat_completion_request")
@patch("litellm.llms.openai.openai.OpenAIChatCompletion._get_openai_client")
def test_sdk_path_merges_instead_of_replacing(mock_get_client, mock_request):
    """Same bug at the unit level: headers= must not drop extra_headers=."""
    from litellm.llms.openai.openai import OpenAIChatCompletion

    parsed_response = MagicMock()
    parsed_response.model_dump.return_value = CHAT_COMPLETION_BODY
    mock_get_client.return_value = MagicMock()
    mock_request.return_value = ({}, parsed_response)

    OpenAIChatCompletion().completion(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
        model_response=litellm.ModelResponse(),
        timeout=60.0,
        optional_params={"extra_headers": dict(COPILOT_HEADERS)},
        litellm_params={},
        logging_obj=MagicMock(),
        custom_llm_provider="github_copilot",
        api_key="ghu_fake_token",
        headers={"x-forwarded-by-client": "1"},
        acompletion=False,
    )

    sent_data = mock_request.call_args.kwargs["data"]
    assert sent_data["extra_headers"] == {**COPILOT_HEADERS, "x-forwarded-by-client": "1"}


def test_http_handler_applies_extra_headers_from_optional_params():
    """The generic handler serves most non-OpenAI providers; it used to drop these."""
    from litellm.llms.base_llm.chat.transformation import BaseConfig
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    config = MagicMock(spec=BaseConfig)
    config.validate_environment.return_value = {"Authorization": "Bearer k"}
    config.get_complete_url.return_value = "https://example.com/v1/chat/completions"
    config.should_fake_stream.return_value = False
    config.max_retry_on_unprocessable_entity_error = 1

    response = httpx.Response(200, json=CHAT_COMPLETION_BODY)
    with patch.object(BaseLLMHTTPHandler, "_make_common_sync_call", return_value=response):
        BaseLLMHTTPHandler().completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://example.com/v1",
            custom_llm_provider="openai_like",
            model_response=litellm.ModelResponse(),
            encoding=None,
            logging_obj=MagicMock(),
            optional_params={"extra_headers": {"x-provider-header": "1"}},
            timeout=5.0,
            litellm_params={},
            acompletion=False,
            headers={"x-caller-header": "2"},
            provider_config=config,
        )

    assert config.validate_environment.call_args.kwargs["headers"] == {
        "x-provider-header": "1",
        "x-caller-header": "2",
    }
