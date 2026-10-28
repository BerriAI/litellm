from typing import Final

import pytest

from litellm.llms.github_copilot.github_host import (
    github_host_endpoints,
    normalized_github_host,
)


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
    monkeypatch.setenv("GITHUB_COPILOT_API_KEY_URL", "https://api.example/copilot")
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "custom-client")

    endpoints: Final = github_host_endpoints("github.com")

    assert endpoints.device_code_url == "https://device.example/code"
    assert endpoints.access_token_url == "https://device.example/token"
    assert endpoints.api_key_url == "https://api.example/copilot"
    assert endpoints.client_id == "custom-client"


def test_ghe_endpoints_ignore_github_com_url_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_DEVICE_CODE_URL", "https://device.example/code")
    monkeypatch.setenv("GITHUB_COPILOT_ACCESS_TOKEN_URL", "https://device.example/token")
    monkeypatch.setenv("GITHUB_COPILOT_API_KEY_URL", "https://api.example/copilot")
    monkeypatch.setenv("GITHUB_COPILOT_CLIENT_ID", "custom-client")

    endpoints: Final = github_host_endpoints("octo.ghe.com")

    assert endpoints.device_code_url == "https://octo.ghe.com/login/device/code"
    assert endpoints.access_token_url == "https://octo.ghe.com/login/oauth/access_token"
    assert endpoints.api_key_url == "https://api.octo.ghe.com/copilot_internal/v2/token"
    assert endpoints.user_url == "https://api.octo.ghe.com/user"
    assert endpoints.default_copilot_api_base == "https://copilot-api.octo.ghe.com"
    assert endpoints.client_id == "custom-client"
