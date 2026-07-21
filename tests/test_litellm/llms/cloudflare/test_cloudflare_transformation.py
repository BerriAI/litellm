import pytest

from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig


def test_supported_params_include_tools_and_tool_choice():
    config = CloudflareChatConfig()

    params = config.get_supported_openai_params(model="@cf/meta/llama-2-7b-chat-int8")

    assert "tools" in params
    assert "tool_choice" in params
    assert "stream" in params
    assert "max_tokens" in params


def test_get_complete_url_defaults_to_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base=None,
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    )
    assert "/ai/run/" not in url


def test_get_complete_url_appends_chat_completions_to_explicit_base():
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/v1",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    )
    assert "/ai/run/" not in url


def test_get_complete_url_is_idempotent_for_full_base():
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    )


def test_get_complete_url_falls_back_to_account_id_when_base_is_empty(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    )


def test_get_complete_url_raises_when_account_id_and_base_missing(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    config = CloudflareChatConfig()

    with pytest.raises(ValueError, match="Missing CLOUDFLARE_ACCOUNT_ID"):
        config.get_complete_url(
            api_base=None,
            api_key="cf-key",
            model="@cf/meta/llama-2-7b-chat-int8",
            optional_params={},
            litellm_params={},
        )


def test_get_complete_url_raises_when_account_id_is_empty(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "   ")
    config = CloudflareChatConfig()

    with pytest.raises(ValueError, match="Missing CLOUDFLARE_ACCOUNT_ID"):
        config.get_complete_url(
            api_base=None,
            api_key="cf-key",
            model="@cf/meta/llama-2-7b-chat-int8",
            optional_params={},
            litellm_params={},
        )


def test_get_complete_url_migrates_legacy_ai_run_base():
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    )
    assert "/ai/run" not in url


def test_transform_request_passes_tools_through_in_openai_format():
    config = CloudflareChatConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "weather in nyc?"}]

    body = config.transform_request(
        model="@cf/meta/llama-2-7b-chat-int8",
        messages=messages,
        optional_params={"tools": tools, "tool_choice": "auto"},
        litellm_params={},
        headers={},
    )

    assert body["messages"] == messages
    assert body["model"] == "@cf/meta/llama-2-7b-chat-int8"
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_transform_request_flattens_content_part_arrays_to_string():
    config = CloudflareChatConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "More text"},
            ],
        }
    ]

    body = config.transform_request(
        model="@cf/meta/llama-2-7b-chat-int8",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert body["messages"][0]["content"] == "HelloMore text"


def test_transform_request_leaves_string_content_untouched():
    config = CloudflareChatConfig()
    messages = [
        {"role": "system", "content": "you are terse"},
        {"role": "user", "content": "weather in nyc?"},
    ]

    body = config.transform_request(
        model="@cf/meta/llama-2-7b-chat-int8",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert [m["content"] for m in body["messages"]] == ["you are terse", "weather in nyc?"]


def test_validate_environment_requires_api_key():
    config = CloudflareChatConfig()

    with pytest.raises(ValueError, match="Missing Cloudflare API Key"):
        config.validate_environment(
            headers={},
            model="@cf/meta/llama-2-7b-chat-int8",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )


def test_validate_environment_sets_bearer_and_content_type():
    config = CloudflareChatConfig()

    headers = config.validate_environment(
        headers={},
        model="@cf/meta/llama-2-7b-chat-int8",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="cf-key",
    )

    assert headers["Authorization"] == "Bearer cf-key"
    assert headers["Content-Type"] == "application/json"
