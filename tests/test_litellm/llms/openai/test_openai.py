import pytest

from litellm.llms.openai.openai import OpenAIChatCompletion, OpenAIConfig


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.openai.com/v1",
        "https://api.openai.com:443/v1",
        "https://southcentralus.privatelink.api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        "https://us.api.openai.com/v1",
        "HTTPS://API.OPENAI.COM/v1/",
    ],
)
def test_get_stream_options_defaults_include_usage_on_every_openai_backed_host(api_base):
    """
    PrivateLink and regional hostnames reach the real OpenAI backend, so a stream with no caller
    stream_options must ask for the usage chunk exactly as the default base does. Regression guard
    for LIT-6875: spend for those deployments fell back to local token counting.
    """
    assert OpenAIChatCompletion().get_stream_options(stream_options=None, api_base=api_base) == {
        "stream_options": {"include_usage": True}
    }


@pytest.mark.parametrize(
    "api_base",
    [
        "https://my-gateway.example/v1",
        "https://api.openai.com.evil.example/v1",
        "https://notapi.openai.com/v1",
        "https://gateway.example/v1?upstream=api.openai.com",
        "https://openai.internal.example/api.openai.com/v1",
    ],
)
def test_get_stream_options_leaves_foreign_hosts_without_a_usage_default(api_base):
    """Only the host decides: an OpenAI-compatible backend elsewhere may not support stream_options at all."""
    assert OpenAIChatCompletion().get_stream_options(stream_options=None, api_base=api_base) == {}


@pytest.mark.parametrize(
    "api_base",
    ["https://southcentralus.privatelink.api.openai.com/v1", "https://my-gateway.example/v1"],
)
def test_get_stream_options_passes_caller_stream_options_through_on_any_host(api_base):
    caller_options = {"include_usage": False}
    assert OpenAIChatCompletion().get_stream_options(stream_options=caller_options, api_base=api_base) == {
        "stream_options": caller_options
    }


def _developer_message_after_a_user_turn():
    return [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "developer", "content": "Answer with exactly one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


class TestTranslateDeveloperRoleToSystemRole:
    def test_hosted_openai_keeps_the_developer_message_in_place(self):
        messages = OpenAIConfig().translate_developer_role_to_system_role(
            messages=_developer_message_after_a_user_turn(),
            custom_llm_provider="openai",
            api_base="https://api.openai.com/v1",
        )
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "system", "user"]
        assert messages[3]["content"] == "Answer with exactly one word."

    def test_openai_compatible_provider_hoists_the_developer_message(self):
        messages = OpenAIConfig().translate_developer_role_to_system_role(
            messages=_developer_message_after_a_user_turn(),
            custom_llm_provider="azure_ai",
            api_base="https://example.services.ai.azure.com/models",
        )
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
        assert messages[0]["content"] == "You are terse.\n\nAnswer with exactly one word."
