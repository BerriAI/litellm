from unittest.mock import MagicMock, patch

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


@patch("litellm.llms.chatgpt.chat.transformation.Authenticator")
def test_chatgpt_validate_environment_uses_request_auth_file_path(
    mock_authenticator_class,
):
    default_authenticator = MagicMock()
    default_authenticator.get_api_base.return_value = (
        "https://chatgpt.com/backend-api/codex"
    )

    request_authenticator = MagicMock()
    request_authenticator.get_access_token.return_value = "access-123"
    request_authenticator.get_account_id.return_value = "acct-123"
    request_authenticator.get_api_base.return_value = (
        "https://chatgpt.com/backend-api/codex"
    )

    mock_authenticator_class.side_effect = [
        default_authenticator,
        request_authenticator,
    ]

    config = ChatGPTConfig()
    headers = config.validate_environment(
        headers={},
        model="chatgpt/gpt-5.3-codex",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={
            "chatgpt_auth_file_path": "/tmp/chatgpt-account-a.json",
            "litellm_session_id": "session-123",
        },
    )

    mock_authenticator_class.assert_any_call(
        auth_file_path="/tmp/chatgpt-account-a.json",
        api_base=None,
    )
    assert headers["Authorization"] == "Bearer access-123"
    assert headers["ChatGPT-Account-Id"] == "acct-123"
    assert headers["session_id"] == "session-123"


@patch("litellm.llms.chatgpt.chat.transformation.Authenticator")
def test_chatgpt_provider_info_prefers_request_api_base(mock_authenticator_class):
    default_authenticator = MagicMock()
    default_authenticator.get_api_base.return_value = (
        "https://chatgpt.com/backend-api/codex"
    )
    mock_authenticator_class.return_value = default_authenticator

    config = ChatGPTConfig()
    dynamic_api_base, dynamic_api_key, custom_llm_provider = (
        config._get_openai_compatible_provider_info(
            model="gpt-5.3-codex",
            api_base="https://chatgpt.example.internal/backend-api/codex",
            api_key=None,
            custom_llm_provider="chatgpt",
        )
    )

    assert dynamic_api_base == "https://chatgpt.example.internal/backend-api/codex"
    assert dynamic_api_key is None
    assert custom_llm_provider == "chatgpt"
