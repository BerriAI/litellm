import pytest

from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig


def test_get_complete_url_encodes_model_path_segment():
    config = CloudflareChatConfig()

    assert (
        config.get_complete_url(
            api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
            api_key="cf-key",
            model="@cf/meta/llama?x=1#frag",
            optional_params={},
            litellm_params={},
        )
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/%40cf/meta/llama%3Fx%3D1%23frag"
    )

    with pytest.raises(ValueError, match="dot path segment"):
        config.get_complete_url(
            api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
            api_key="cf-key",
            model="../../accounts/other",
            optional_params={},
            litellm_params={},
        )


def test_validate_environment_content_type():
    """Test that the content-type header is correctly set to application/json."""
    config = CloudflareChatConfig()
    headers = config.validate_environment(
        headers={},
        model="@cf/meta/llama",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key="fake-key",
    )

    assert headers["content-type"] == "application/json"
    assert headers["Authorization"] == "Bearer fake-key"
