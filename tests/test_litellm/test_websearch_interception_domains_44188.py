from copy import deepcopy
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.integrations.websearch_interception.tools import get_litellm_web_search_tool
from litellm.integrations.websearch_interception.transformation import WebSearchTransformation
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.types.integrations.websearch_interception import SearchSucceeded
from litellm.types.utils import CallTypes


@pytest.fixture
def search_response() -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(title=f"Result {index}", url=url, snippet=f"Snippet {index}")
            for index, url in enumerate(
                (
                    "https://example.org/article",
                    "https://docs.example.org/guide",
                    "https://example.com/article",
                    "https://example.net/article",
                    "https://docs.example.net/guide",
                    "https://notexample.org/article",
                    "https://example.org.evil.test/article",
                )
            )
        ]
    )


@pytest.fixture
def search_mock(monkeypatch: pytest.MonkeyPatch, search_response: SearchResponse) -> AsyncMock:
    from litellm.proxy import proxy_server

    mock: Final = AsyncMock(return_value=search_response)
    monkeypatch.setattr(litellm, "asearch", mock)
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        SimpleNamespace(search_tools=[{"litellm_params": {"search_provider": "firecrawl"}}]),
    )
    return mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filters", "expected_indices"),
    [
        ({"allowed_domains": ["example.org"]}, (0, 1)),
        ({"blocked_domains": ["example.net"]}, (0, 1, 2, 5, 6)),
        ({"allowed_domains": ["example.org"], "blocked_domains": ["docs.example.org"]}, (0,)),
        ({"allowed_domains": ["example.org"], "blocked_domains": ["example.org"]}, ()),
        ({"allowed_domains": []}, ()),
        ({}, (0, 1, 2, 3, 4, 5, 6)),
    ],
)
async def test_short_circuit_filters_citations_and_text(
    filters: dict[str, list[str]],
    expected_indices: tuple[int, ...],
    search_response: SearchResponse,
    search_mock: AsyncMock,
) -> None:
    logger: Final = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])
    response: Final = await logger.try_short_circuit_search(
        model="interception-test-model",
        messages=[{"role": "user", "content": "Find domain documentation"}],
        tools=[{"type": "web_search_20250305", "name": "web_search", **filters}],
        custom_llm_provider="github_copilot",
    )

    assert response is not None
    expected: Final = [search_response.results[index] for index in expected_indices]
    content: Final = response["content"]
    assert [item["url"] for item in content[1]["content"]] == [item.url for item in expected]
    assert content[2]["text"] == (
        "\n\n".join(f"Title: {item.title}\nURL: {item.url}\nSnippet: {item.snippet}" for item in expected)
        if expected
        else "No search results found."
    )
    assert response["stop_reason"] == "end_turn"
    search_mock.assert_awaited_once_with(
        query="Find domain documentation",
        search_provider="firecrawl",
        **({"search_domain_filter": filters["allowed_domains"]} if "allowed_domains" in filters else {}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["pre_request", "deployment", "both_hooks", "chat_completion", "responses"])
@pytest.mark.parametrize("source", ["definition", "arguments", "attempted_override"])
async def test_agentic_paths_preserve_domain_constraints(
    path: str,
    source: str,
    search_response: SearchResponse,
    search_mock: AsyncMock,
) -> None:
    logger: Final = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
    filters: Final = {"allowed_domains": ["example.org"], "blocked_domains": ["docs.example.org"]}
    kwargs: Final = {
        "custom_llm_provider": "bedrock",
        "litellm_params": {"custom_llm_provider": "bedrock"},
        "tools": [{"type": "web_search_20250305", "name": "web_search", **(filters if source != "arguments" else {})}],
    }
    if path in ("deployment", "both_hooks", "chat_completion"):
        await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    if path in ("pre_request", "both_hooks", "responses"):
        await logger.async_pre_request_hook(model="interception-test-model", messages=[], kwargs=kwargs)
    tool_input: Final = {
        "query": "Find domain documentation",
        **(filters if source == "arguments" else {}),
        **({"allowed_domains": ["example.com"], "blocked_domains": []} if source == "attempted_override" else {}),
    }
    tool_calls: Final = [{"id": "toolu_test", "name": "litellm_web_search", "input": tool_input}]
    optional_params: Final = {
        "max_tokens": 1024,
        **({"tools": kwargs["tools"]} if path != "pre_request" else {}),
    }
    expected_text: Final = "Title: Result 0\nURL: https://example.org/article\nSnippet: Snippet 0"

    if path == "chat_completion":
        chat_patch: Final = await logger._build_chat_completion_request_patch(
            model="interception-test-model",
            messages=[],
            tool_calls=tool_calls,
            optional_params=optional_params,
            kwargs={},
        )
        assert chat_patch.messages[-1]["content"] == expected_text
    elif path == "responses":
        responses_patch: Final = await logger._build_responses_request_patch(
            model="interception-test-model",
            messages=[],
            tool_calls=tool_calls,
            optional_params=optional_params,
            kwargs={},
        )
        assert responses_patch.messages[-1]["output"] == expected_text
    else:
        patch, outcomes = await logger._build_anthropic_request_patch(
            model="interception-test-model",
            messages=[],
            tool_calls=tool_calls,
            thinking_blocks=[],
            anthropic_messages_optional_request_params=optional_params,
            logging_obj=None,
            kwargs={"tools": kwargs["tools"]},
        )
        assert patch.messages[-1]["content"][0]["content"] == expected_text
        assert isinstance(outcomes[0], SearchSucceeded)
        assert outcomes[0].response.results == [search_response.results[0]]
    search_mock.assert_awaited_once_with(
        query="Find domain documentation", search_provider="firecrawl", search_domain_filter=["example.org"]
    )


@pytest.mark.parametrize(
    ("url", "retained"),
    [
        ("https://EXAMPLE.ORG:8443/article", True),
        ("https://docs.EXAMPLE.ORG./article", True),
        ("//docs.example.org/article", True),
        ("https://example.org@evil.test/article", False),
        ("https://evil.test/example.org", False),
        ("https://evil.test/?url=https://example.org", False),
        ("https://notexample.org/article", False),
        ("https://example.org.evil.test/article", False),
        ("https://[invalid/article", False),
        ("/relative/article", False),
    ],
)
def test_domain_matching_uses_normalized_hostname(url: str, retained: bool) -> None:
    item: Final = SearchResult(title="Title", url=url, snippet="Snippet")
    response: Final = SearchResponse(results=[item])
    filtered: Final = WebSearchTransformation.filter_search_response(
        response, allowed_domains=["https://EXAMPLE.ORG:443"]
    )
    assert filtered.results == ([item] if retained else [])
    assert response.results == [item]


@pytest.mark.asyncio
async def test_tool_builder_preserves_filters_for_execution(search_mock: AsyncMock) -> None:
    tool: Final = get_litellm_web_search_tool(allowed_domains=["example.org"], blocked_domains=["docs.example.org"])
    logger: Final = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])
    response: Final = await logger.try_short_circuit_search(
        model="interception-test-model",
        messages=[{"role": "user", "content": "Find domain documentation"}],
        tools=[tool],
        custom_llm_provider="github_copilot",
    )
    assert response["content"] == [
        {"type": "text", "text": "Title: Result 0\nURL: https://example.org/article\nSnippet: Snippet 0"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["request", "deployment"])
async def test_disabled_interception_leaves_native_domains_untouched(hook: str) -> None:
    logger: Final = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
    tool: Final = {
        "type": "web_search_20250305",
        "name": "web_search",
        "allowed_domains": ["example.org"],
        "blocked_domains": ["docs.example.org"],
    }
    kwargs: Final = {
        "tools": [tool],
        "custom_llm_provider": "anthropic",
        "litellm_params": {"custom_llm_provider": "anthropic"},
    }
    original: Final = deepcopy(kwargs)
    result: Final = (
        await logger.async_pre_request_hook(model="interception-test-model", messages=[], kwargs=kwargs)
        if hook == "request"
        else await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    )
    assert result is None
    assert kwargs == original


@pytest.mark.parametrize("blocked_domains", [None, []])
def test_unconstrained_search_preserves_response(
    search_response: SearchResponse, blocked_domains: list[str] | None
) -> None:
    assert (
        WebSearchTransformation.filter_search_response(search_response, blocked_domains=blocked_domains)
        is search_response
    )
