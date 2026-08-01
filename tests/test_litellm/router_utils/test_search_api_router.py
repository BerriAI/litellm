from litellm.router_utils.search_api_router import SearchAPIRouter


def test_resolve_search_provider_credentials_expands_os_environ_refs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("E2E_SEARCH_API_KEY", "secret-from-env")
    monkeypatch.setenv("E2E_SEARCH_API_BASE", "https://search.example")

    api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
        tool_litellm_params={
            "api_key": "os.environ/E2E_SEARCH_API_KEY",
            "api_base": "os.environ/E2E_SEARCH_API_BASE",
        }
    )

    assert api_key == "secret-from-env"
    assert api_base == "https://search.example"


def test_resolve_search_provider_credentials_passes_through_plain_values() -> None:
    api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
        tool_litellm_params={
            "api_key": "sk-literal",
            "api_base": "https://literal.example",
        }
    )

    assert api_key == "sk-literal"
    assert api_base == "https://literal.example"


def test_resolve_search_provider_credentials_missing_env_returns_none(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MISSING_SEARCH_KEY_XYZ", raising=False)

    api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
        tool_litellm_params={"api_key": "os.environ/MISSING_SEARCH_KEY_XYZ"}
    )

    assert api_key is None
    assert api_base is None
