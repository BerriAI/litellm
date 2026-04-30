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
