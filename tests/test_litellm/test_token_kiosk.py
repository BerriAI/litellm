import litellm
from litellm.llms.token_kiosk.chat.transformation import TokenKioskConfig


def test_token_kiosk_config_get_complete_url():
    config = TokenKioskConfig()
    url = config.get_complete_url(
        api_base="https://agent-router.gaib.ai/v1",
        api_key="test-key",
        model="token_kiosk/claude-3-5-sonnet",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://agent-router.gaib.ai/v1/chat/completions"

    # Test trailing slash
    url_slash = config.get_complete_url(
        api_base="https://agent-router.gaib.ai/v1/",
        api_key="test-key",
        model="token_kiosk/claude-3-5-sonnet",
        optional_params={},
        litellm_params={},
    )
    assert url_slash == "https://agent-router.gaib.ai/v1/chat/completions"

    # Test full endpoint path already ending with /chat/completions
    url_full = config.get_complete_url(
        api_base="https://agent-router.gaib.ai/v1/chat/completions",
        api_key="test-key",
        model="token_kiosk/claude-3-5-sonnet",
        optional_params={},
        litellm_params={},
    )
    assert url_full == "https://agent-router.gaib.ai/v1/chat/completions"


def test_token_kiosk_llm_provider_enum():
    assert litellm.LlmProviders.TOKEN_KIOSK.value == "token_kiosk"
    assert "token_kiosk" in litellm.openai_compatible_providers


def test_token_kiosk_get_openai_compatible_provider_info():
    config = TokenKioskConfig()
    api_base, api_key = config.get_openai_compatible_provider_info(
        api_base=None, api_key="test-key"
    )
    assert api_base == "https://agent-router.gaib.ai/v1"
    assert api_key == "test-key"


def test_token_kiosk_lazy_import():
    assert litellm.TokenKioskConfig is TokenKioskConfig


def test_token_kiosk_get_llm_provider():
    model, custom_llm_provider, dynamic_api_key, api_base = (
        litellm.get_llm_provider(
            model="token_kiosk/claude-3-5-sonnet",
            api_key="test-token-kiosk-key",
        )
    )
    assert model == "claude-3-5-sonnet"
    assert custom_llm_provider == "token_kiosk"
    assert dynamic_api_key == "test-token-kiosk-key"
    assert api_base == "https://agent-router.gaib.ai/v1"
