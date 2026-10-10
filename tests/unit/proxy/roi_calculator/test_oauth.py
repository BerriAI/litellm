from dataclasses import replace
from typing import Final

from pydantic import SecretStr

from litellm.proxy.roi_calculator.oauth import OAuthConfig


def test_app_urls_support_enterprise_and_a_gateway_path_prefix() -> None:
    cloud: Final = OAuthConfig(
        "github",
        "https://api.github.com",
        "https://github.com",
        "test-client",
        SecretStr("test-secret"),
        "https://gateway.example.test/proxy",
        "test-app",
    )
    enterprise: Final = replace(cloud, api_url="https://git.example.test/api/v3", base_url="https://git.example.test")
    assert cloud.installation_url == cloud.base_url + "/apps/test-app/installations/new"
    assert enterprise.installation_url == enterprise.base_url + "/github-apps/test-app/installations/new"
    assert enterprise.cookie_path == "/proxy/roi-calculator/observed/oauth"
    assert enterprise.redirect_uri.startswith(enterprise.proxy_url + "/roi-calculator/")
    assert replace(cloud, provider="gitlab").installation_url is None
    assert replace(cloud, app_slug="").installation_url is None
