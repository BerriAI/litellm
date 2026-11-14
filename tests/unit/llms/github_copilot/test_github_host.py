from typing import Final

import pytest

from litellm.llms.github_copilot.github_host import (
    github_host_endpoints,
    normalized_github_host,
    shared_login_github_host,
)

GHE_API_KEY_URL: Final = "https://api.octo.ghe.com/copilot_internal/v2/token"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "github.com"),
        ("", "github.com"),
        ("GitHub.com", "github.com"),
        ("OCTO.ghe.com", "octo.ghe.com"),
        ("octo-1.ghe.com", "octo-1.ghe.com"),
        ("https://octo.ghe.com", None),
        ("ghes.example.com", None),
        ("octo.ghe.com:443", None),
        ("a.b.ghe.com", None),
        ("octo.ghe.com.", None),
        (" api.octo.ghe.com", None),
        ("octo.ghe.com/path", None),
    ],
)
def test_normalized_github_host_accepts_only_bare_supported_hosts(raw: object, expected: str | None) -> None:
    assert normalized_github_host(raw) == expected


def test_github_com_endpoints_read_url_overrides_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_DEVICE_CODE_URL", "https://device.example/code")
    monkeypatch.setenv("GITHUB_COPILOT_ACCESS_TOKEN_URL", "https://device.example/token")
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "custom-client")

    endpoints: Final = github_host_endpoints("github.com")

    assert endpoints.device_code_url == "https://device.example/code"
    assert endpoints.access_token_url == "https://device.example/token"
    assert endpoints.copilot_user_url == "https://api.github.com/copilot_internal/user"
    assert endpoints.client_id == "custom-client"


def test_ghe_endpoints_ignore_github_com_url_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_DEVICE_CODE_URL", "https://device.example/code")
    monkeypatch.setenv("GITHUB_COPILOT_ACCESS_TOKEN_URL", "https://device.example/token")
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "custom-client")

    endpoints: Final = github_host_endpoints("octo.ghe.com")

    assert endpoints.device_code_url == "https://octo.ghe.com/login/device/code"
    assert endpoints.access_token_url == "https://octo.ghe.com/login/oauth/access_token"
    assert endpoints.copilot_user_url == "https://api.octo.ghe.com/copilot_internal/user"
    assert endpoints.user_url == "https://api.octo.ghe.com/user"
    assert endpoints.default_copilot_api_base == "https://copilot-api.octo.ghe.com"
    assert endpoints.client_id == "custom-client"


@pytest.mark.parametrize(
    ("access_token_url", "api_key_url", "expected"),
    [
        (None, None, "github.com"),
        ("https://Octo.ghe.com/login/oauth/access_token", None, "octo.ghe.com"),
        (None, GHE_API_KEY_URL, "octo.ghe.com"),
        (None, "https://octo.ghe.com/api/v3/copilot_internal/v2/token", "octo.ghe.com"),
        ("https://github.com/login/oauth/access_token", GHE_API_KEY_URL, "github.com"),
        ("http://127.0.0.1:8080/login/oauth/access_token", GHE_API_KEY_URL, "octo.ghe.com"),
        ("https://ghes.example.com/login/oauth/access_token", None, "github.com"),
        ("https://[octo.ghe.com/login/oauth/access_token", None, "github.com"),
    ],
)
def test_shared_login_github_host_reads_the_tenant_from_the_login_urls(
    monkeypatch: pytest.MonkeyPatch, access_token_url: str | None, api_key_url: str | None, expected: str
) -> None:
    for name, value in (
        ("GITHUB_COPILOT_ACCESS_TOKEN_URL", access_token_url),
        ("GITHUB_COPILOT_API_KEY_URL", api_key_url),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert shared_login_github_host() == expected
