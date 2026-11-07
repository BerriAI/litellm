import json
from typing import Final

import pytest
import respx

import litellm

PROXY_RESPONSE: Final = {
    "object": "search",
    "results": [
        {
            "title": "LiteLLM",
            "url": "https://www.litellm.ai",
            "snippet": "LiteLLM is an open-source AI gateway",
            "date": None,
            "last_updated": None,
            "score": 0.95,
            "raw_content": None,
        }
    ],
    "provider_metadata": {"request_id": "req-1", "response_time": 0.8},
}


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        ("https://proxy.example.com", "https://proxy.example.com/search/web-search"),
        ("https://proxy.example.com/", "https://proxy.example.com/search/web-search"),
        ("https://proxy.example.com/v1", "https://proxy.example.com/v1/search/web-search"),
        ("https://gateway.example.com/litellm", "https://gateway.example.com/litellm/search/web-search"),
    ],
)
def test_search_calls_the_named_proxy_tool_and_returns_its_response_unchanged(
    respx_mock: respx.MockRouter, api_base: str, expected_url: str
) -> None:
    route: Final = respx_mock.post(expected_url).respond(json=PROXY_RESPONSE)

    response: Final = litellm.search(
        query="What is the LiteLLM proxy?",
        search_provider="litellm_proxy",
        search_tool_name="web-search",
        api_base=api_base,
        api_key="sk-virtual-key",
        max_results=3,
        search_domain_filter=["litellm.ai"],
        topic="news",
    )

    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-virtual-key"
    assert json.loads(request.content) == {
        "query": "What is the LiteLLM proxy?",
        "max_results": 3,
        "search_domain_filter": ["litellm.ai"],
        "topic": "news",
    }
    assert response.model_dump(mode="json") == PROXY_RESPONSE


@pytest.mark.asyncio
async def test_asearch_calls_the_named_proxy_tool_and_returns_its_response_unchanged(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://proxy.example.com/v1/search/web-search").respond(json=PROXY_RESPONSE)

    response: Final = await litellm.asearch(
        query=["LiteLLM", "proxy"],
        search_provider="litellm_proxy",
        search_tool_name="web-search",
        api_base="https://proxy.example.com/v1",
        api_key="sk-virtual-key",
    )

    assert json.loads(route.calls.last.request.content) == {"query": ["LiteLLM", "proxy"]}
    assert response.model_dump(mode="json") == PROXY_RESPONSE


def test_a_tool_name_with_a_slash_stays_one_path_segment(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(url__regex=r"https://proxy\.example\.com/.*").respond(json=PROXY_RESPONSE)

    litellm.search(
        query="q",
        search_provider="litellm_proxy",
        search_tool_name="team/web search",
        api_base="https://proxy.example.com",
        api_key="sk-virtual-key",
    )

    assert route.calls.last.request.url.raw_path == b"/search/team%2Fweb%20search"


@pytest.mark.parametrize("tool_name", ["..", ".", ""])
def test_a_dot_tool_name_is_refused_before_any_request(respx_mock: respx.MockRouter, tool_name: str) -> None:
    with pytest.raises(litellm.BadRequestError):
        litellm.search(
            query="q",
            search_provider="litellm_proxy",
            search_tool_name=tool_name,
            api_base="https://proxy.example.com",
            api_key="sk-virtual-key",
        )

    assert not respx_mock.calls


def test_search_without_a_tool_name_is_a_bad_request_before_any_request(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="search_tool_name"):
        litellm.search(
            query="q",
            search_provider="litellm_proxy",
            api_base="https://proxy.example.com",
            api_key="sk-virtual-key",
        )

    assert not respx_mock.calls


def test_search_without_a_proxy_url_is_a_bad_request() -> None:
    with pytest.raises(litellm.BadRequestError, match="LITELLM_PROXY_API_BASE"):
        litellm.search(query="q", search_provider="litellm_proxy", search_tool_name="web-search")


def test_search_reads_the_proxy_url_and_key_from_the_environment(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://proxy.example.com")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-env-key")
    route: Final = respx_mock.post("https://proxy.example.com/search/web-search").respond(json=PROXY_RESPONSE)

    litellm.search(query="q", search_provider="litellm_proxy", search_tool_name="web-search")

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-env-key"


def test_the_environment_key_is_not_sent_to_a_caller_chosen_host(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://proxy.example.com")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-env-key")

    with pytest.raises(litellm.APIConnectionError, match="LITELLM_PROXY_API_KEY"):
        litellm.search(
            query="q",
            search_provider="litellm_proxy",
            search_tool_name="web-search",
            api_base="https://attacker.example.com",
        )

    assert not respx_mock.calls


def test_search_against_a_proxy_without_auth_sends_no_authorization_header(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://proxy.example.com/search/web-search").respond(json=PROXY_RESPONSE)

    litellm.search(
        query="q",
        search_provider="litellm_proxy",
        search_tool_name="web-search",
        api_base="https://proxy.example.com",
    )

    assert "Authorization" not in route.calls.last.request.headers


@pytest.mark.asyncio
async def test_a_router_proxy_tool_calls_the_downstream_tool_its_params_name(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://proxy.example.com/search/remote-tavily").respond(json=PROXY_RESPONSE)
    router: Final = litellm.Router(
        model_list=[],
        search_tools=[
            {
                "search_tool_name": "local-alias",
                "litellm_params": {
                    "search_provider": "litellm_proxy",
                    "api_base": "https://proxy.example.com",
                    "api_key": "sk-downstream-key",
                    "search_tool_name": "remote-tavily",
                },
            }
        ],
    )

    response: Final = await router.asearch(
        model="local-alias", search_tool_name="local-alias", query="What is the LiteLLM proxy?", max_results=2
    )

    assert route.call_count == 1
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-downstream-key"
    assert json.loads(route.calls.last.request.content) == {"query": "What is the LiteLLM proxy?", "max_results": 2}
    assert response.model_dump(mode="json") == PROXY_RESPONSE


@pytest.mark.asyncio
async def test_a_router_proxy_tool_without_a_downstream_name_is_a_bad_request_before_any_request(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(url__regex=r"https://proxy\.example\.com/.*").respond(json=PROXY_RESPONSE)
    router: Final = litellm.Router(
        model_list=[],
        search_tools=[
            {
                "search_tool_name": "local-alias",
                "litellm_params": {
                    "search_provider": "litellm_proxy",
                    "api_base": "https://proxy.example.com",
                    "api_key": "sk-downstream-key",
                },
            }
        ],
    )

    with pytest.raises(litellm.BadRequestError, match="search_tool_name"):
        await router.asearch(model="local-alias", search_tool_name="local-alias", query="What is the LiteLLM proxy?")

    assert route.call_count == 0

