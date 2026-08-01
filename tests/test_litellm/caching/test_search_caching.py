"""
Regression tests for caching of the Search API (`litellm.search` / `litellm.asearch`).

Search call types were wired into the cache write/gate path (CallTypes.search /
CallTypes.asearch are valid `supported_call_types`, and both functions are
`@client`-decorated), but the cache *read* path was incomplete:

  1. On a cache hit, the wrapper unconditionally called `litellm.get_llm_provider(model=...)`.
     For a search the `model` is a search-tool name (e.g. "tavily-search"), which has no
     LLM provider, so the second (cached) call raised
     "LLM Provider NOT provided ... Received Model Group=tavily-search".
  2. `_convert_cached_result_to_model_response` had no `search`/`asearch` branch, so a
     cached search dict was never rebuilt into a `SearchResponse`.

These tests confirm a second, identical search is served from cache as a proper
`SearchResponse` without re-hitting the provider.
"""

import asyncio
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler


def _make_search_response() -> SearchResponse:
    resp = SearchResponse(
        results=[
            SearchResult(
                title="LiteLLM Caching",
                url="https://docs.litellm.ai/docs/proxy/caching",
                snippet="How to cache responses in LiteLLM.",
            )
        ],
        object="search",
    )
    resp._hidden_params = {"response_cost": 0.0}
    return resp


def _patch_provider_search():
    return patch.object(
        BaseLLMHTTPHandler,
        "search",
        return_value=_make_search_response(),
    )


@pytest.fixture(autouse=True)
def _search_cache():
    os.environ["TAVILY_API_KEY"] = "sk-test-tavily"
    # Search is not in the default supported_call_types, so opt it in (this is
    # exactly what a user must do in cache_params to cache search). Completion is
    # included too so the non-search cache-hit path (provider resolution) is
    # exercised alongside it.
    litellm.cache = Cache(
        type=LiteLLMCacheType.LOCAL,
        supported_call_types=["search", "asearch", "completion", "acompletion"],
    )
    yield
    litellm.cache = None


@pytest.mark.asyncio
async def test_asearch_second_call_served_from_cache():
    """A repeated async search must hit the cache and not re-call the provider."""
    with _patch_provider_search() as mock_search:
        first = await litellm.asearch(query="litellm cache test", search_provider="tavily")
        second = await litellm.asearch(query="litellm cache test", search_provider="tavily")

    # Provider hit exactly once: the second call came from cache.
    assert mock_search.call_count == 1
    # Regression: the second call previously raised get_llm_provider() on "tavily-search".
    assert isinstance(second, SearchResponse)
    assert second.object == "search"
    assert [r.url for r in second.results] == [r.url for r in first.results]


def test_search_second_call_served_from_cache():
    """Same guarantee for the synchronous entrypoint."""
    with _patch_provider_search() as mock_search:
        first = litellm.search(query="litellm cache test", search_provider="tavily")
        second = litellm.search(query="litellm cache test", search_provider="tavily")

    assert mock_search.call_count == 1
    assert isinstance(second, SearchResponse)
    assert [r.url for r in second.results] == [r.url for r in first.results]


# The search guard added an `else` branch around the existing get_llm_provider()
# call on the cache-hit path. These confirm a normal (non-search) completion
# cache hit still resolves its provider through that branch and is unaffected.
_COMPLETION_KWARGS = dict(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "hi"}],
    mock_response="hello",
    caching=True,
)


@pytest.mark.asyncio
async def test_acompletion_cache_hit_still_resolves_provider():
    await litellm.acompletion(**_COMPLETION_KWARGS)
    # async set_cache runs as a background task; let it settle before re-reading.
    await asyncio.sleep(0.5)
    second = await litellm.acompletion(**_COMPLETION_KWARGS)
    assert second._hidden_params.get("cache_hit") is True


def test_completion_cache_hit_still_resolves_provider():
    litellm.completion(**_COMPLETION_KWARGS)
    second = litellm.completion(**_COMPLETION_KWARGS)
    assert second._hidden_params.get("cache_hit") is True


def test_search_cache_key_distinguishes_provider_and_options():
    """Different search provider/options must not collide on one cached result.

    `query` is already part of the key, but the provider and search options were
    omitted, so e.g. tavily vs perplexity, different max_results/country, or an
    arbitrary provider-specific param like safesearch=off vs strict for the same
    query would have shared a single cached SearchResponse.
    """
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    base = dict(query="x", search_provider="tavily", model="tavily-search")
    key = cache.get_cache_key(**base)
    assert key != cache.get_cache_key(**{**base, "search_provider": "perplexity"})
    assert key != cache.get_cache_key(**{**base, "max_results": 5})
    assert key != cache.get_cache_key(**{**base, "country": "DE"})
    # a provider-specific optional param (not in any fixed search param set) must
    # still affect the key — search keys all non-litellm kwargs, not a fixed list.
    assert key != cache.get_cache_key(**{**base, "safesearch": "strict"})


def test_search_cache_key_excludes_transient_litellm_params():
    """Per-request litellm params must NOT enter the search key, or every search
    would be a cache miss."""
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    base = dict(query="x", search_provider="tavily", model="tavily-search")
    key = cache.get_cache_key(**base)
    assert key == cache.get_cache_key(**{**base, "litellm_call_id": "abc-123"})
    assert key == cache.get_cache_key(**{**base, "litellm_call_id": "different-id"})
