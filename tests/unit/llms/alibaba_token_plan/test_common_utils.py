from collections.abc import Callable
from typing import Final

import pytest

import litellm
from litellm.llms.alibaba_token_plan.common_utils import (
    CHAT_ENDPOINT,
    DEFAULT_API_BASE,
    DEFAULT_CHAT_API_BASE,
    DEFAULT_MESSAGES_API_BASE,
    IMAGE_ENDPOINT,
    REALTIME_ENDPOINT,
    get_api_base,
    get_api_key,
    get_messages_api_base,
    get_native_api_url,
    validate_headers,
)


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, DEFAULT_CHAT_API_BASE),
        ("https://gateway.example/openai/v1", "https://gateway.example/openai/v1"),
        (
            "https://gateway.example/openai/v1/chat/completions",
            "https://gateway.example/openai/v1",
        ),
    ],
)
def test_chat_api_base_uses_openai_compatible_base(api_base: str | None, expected: str) -> None:
    assert get_api_base(api_base) == expected


def test_api_base_environment_and_explicit_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_BASE", "https://environment.example/openai/v1")
    assert get_api_base(None) == "https://environment.example/openai/v1"
    assert get_api_base("https://explicit.example/openai/v1") == "https://explicit.example/openai/v1"


@pytest.mark.parametrize("api_base", [None, DEFAULT_API_BASE, DEFAULT_CHAT_API_BASE])
def test_messages_uses_official_messages_base_for_official_defaults(api_base: str | None) -> None:
    assert get_messages_api_base(api_base) == DEFAULT_MESSAGES_API_BASE


def test_messages_preserves_explicit_gateway_base() -> None:
    assert get_messages_api_base("https://gateway.example/anthropic") == "https://gateway.example/anthropic"


@pytest.mark.parametrize("resolver", [get_api_base, get_messages_api_base])
@pytest.mark.parametrize("suffix", ["?tenant=test", "#fragment"])
def test_compatible_api_bases_reject_query_and_fragment(resolver: Callable[[str | None], str], suffix: str) -> None:
    with pytest.raises(ValueError, match="must not include a query or fragment"):
        resolver(f"https://gateway.example/api/v1{suffix}")


@pytest.mark.parametrize("api_base", [None, DEFAULT_API_BASE, DEFAULT_CHAT_API_BASE, DEFAULT_MESSAGES_API_BASE])
def test_native_operation_uses_official_endpoint_for_official_defaults(api_base: str | None) -> None:
    assert get_native_api_url(api_base, IMAGE_ENDPOINT) == f"{DEFAULT_API_BASE}/{IMAGE_ENDPOINT}"


def test_native_operation_preserves_explicit_full_gateway_url() -> None:
    gateway_url: Final = "https://gateway.example/native/images?tenant=test"
    assert get_native_api_url(gateway_url, IMAGE_ENDPOINT) == gateway_url


@pytest.mark.parametrize("prefix", ["", "/proxy/token-plan"])
@pytest.mark.parametrize(
    "suffix",
    ["/compatible-mode/v1", "/compatible-mode/v1/chat/completions", "/apps/anthropic", "/apps/anthropic/v1/messages"],
)
def test_shared_gateway_base_preserves_prefix_across_api_schemes(
    prefix: str, suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root: Final = f"https://gateway.example:8443{prefix}"
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_BASE", f"{root}{suffix}/")
    assert get_api_base(None) == f"{root}/compatible-mode/v1"
    assert get_messages_api_base(None) == f"{root}/apps/anthropic"
    assert get_native_api_url(None, IMAGE_ENDPOINT) == f"{root}/{IMAGE_ENDPOINT}"
    assert get_native_api_url(None, "") == root


def test_shared_gateway_origin_resolves_each_api_scheme() -> None:
    root: Final = "http://localhost:8080"
    assert get_api_base(root) == f"{root}/compatible-mode/v1"
    assert get_messages_api_base(root) == f"{root}/apps/anthropic"
    assert get_native_api_url(root, IMAGE_ENDPOINT) == f"{root}/{IMAGE_ENDPOINT}"


def test_default_base_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_BASE", raising=False)
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "environment-key")
    headers: Final = {"X-Request-ID": "request-id"}

    assert get_api_base(None) == DEFAULT_CHAT_API_BASE
    assert get_api_key(None) == "environment-key"
    assert validate_headers(headers, "explicit-key") == {
        "X-Request-ID": "request-id",
        "Authorization": "Bearer explicit-key",
        "Content-Type": "application/json",
    }
    assert validate_headers({}, None)["Authorization"] == "Bearer environment-key"
    assert headers == {"X-Request-ID": "request-id"}


def test_missing_provider_key_does_not_use_dashscope_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "pay-as-you-go-key")
    with pytest.raises(litellm.AuthenticationError, match="ALIBABA_TOKEN_PLAN_API_KEY"):
        validate_headers({}, None)


def test_explicit_credentials_replace_case_insensitive_auth_headers() -> None:
    headers: Final = validate_headers(
        {"authorization": "Bearer stale-key", "AUTHORIZATION": "Bearer other-key", "content-type": "text/plain"},
        "configured-key",
    )
    assert headers == {"Authorization": "Bearer configured-key", "Content-Type": "application/json"}


@pytest.mark.parametrize("scheme", ["http", "ws"])
@pytest.mark.parametrize("resolver", [get_api_base, get_messages_api_base])
def test_official_subscription_bases_require_tls(scheme: str, resolver: Callable[[str | None], str]) -> None:
    with pytest.raises(ValueError, match="require HTTPS or WSS"):
        resolver(DEFAULT_API_BASE.replace("https:", f"{scheme}:"))


@pytest.mark.parametrize("scheme", ["http", "ws"])
@pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, REALTIME_ENDPOINT])
def test_official_native_endpoints_require_tls(scheme: str, endpoint: str) -> None:
    with pytest.raises(ValueError, match="require HTTPS or WSS"):
        get_native_api_url(DEFAULT_API_BASE.replace("https:", f"{scheme}:"), endpoint)


@pytest.mark.parametrize("scheme", ["http", "ws"])
def test_explicit_local_gateways_preserve_plaintext_transport(scheme: str) -> None:
    api_base: Final = f"{scheme}://localhost:8080/token-plan"
    assert get_api_base(api_base) == api_base
    assert get_native_api_url(api_base, IMAGE_ENDPOINT) == api_base
