from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig


def test_get_complete_url_encodes_model_path_segment():
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
        api_key="cf-key",
        model="../../accounts/other?x=1#frag",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/..%2F..%2Faccounts%2Fother%3Fx%3D1%23frag"
    )
