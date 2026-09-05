import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.serpapi.search.transformation import SerpApiSearchConfig
from litellm.types.utils import SearchProviders
from litellm.utils import ProviderConfigManager, get_model_info


@pytest.mark.asyncio
async def test_serpapi_asearch_uses_get_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_KEY", "test-api-key")
    local_model_cost = TypeAdapter(dict[str, object]).validate_json(
        (Path(__file__).parents[4] / "model_prices_and_context_window.json").read_text()
    )
    monkeypatch.setattr(litellm, "model_cost", local_model_cost)
    mock_response = httpx.Response(
        status_code=200,
        json={
            "organic_results": [
                {
                    "title": "Test Result",
                    "link": "https://example.com/result",
                    "snippet": "Test snippet",
                }
            ]
        },
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_get,
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        search = cast(
            Callable[..., Awaitable[SearchResponse]],
            litellm.asearch,
        )
        response = await search(
            query="latest AI developments",
            search_provider="serpapi",
            engine="google_light",
            max_results=5,
            hl="en",
        )

    mock_get.assert_awaited_once()
    mock_post.assert_not_awaited()
    request_args = mock_get.await_args
    assert request_args is not None
    request_url = cast(str, request_args.kwargs["url"])
    query_params = parse_qs(urlparse(request_url).query)
    assert query_params["q"] == ["latest AI developments"]
    assert query_params["api_key"] == ["test-api-key"]
    assert query_params["engine"] == ["google_light"]
    assert query_params["num"] == ["5"]
    assert query_params["hl"] == ["en"]
    assert response.results[0].title == "Test Result"
    assert response.results[0].url == "https://example.com/result"
    hidden_params = TypeAdapter(dict[str, object]).validate_python(getattr(response, "_hidden_params"))
    assert hidden_params["response_cost"] == 0.025


def test_serpapi_search_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_KEY", "test-api-key")
    config = ProviderConfigManager.get_provider_search_config(SearchProviders.SERPAPI)
    assert isinstance(config, SerpApiSearchConfig)

    headers = config.validate_environment(headers={})
    data = config.transform_search_request(
        query="latest AI developments",
        optional_params={
            "engine": "google_light",
            "max_results": 5,
            "search_domain_filter": ["arxiv.org", "nature.com"],
            "country": "US",
            "max_tokens_per_page": 1024,
            "hl": "en",
        },
    )
    url = config.get_complete_url(
        api_base=None,
        optional_params={},
        data=data,
    )
    mock_response = httpx.Response(
        status_code=200,
        json={
            "organic_results": [
                {
                    "title": "Test Result",
                    "link": "https://example.com/result",
                    "snippet": "Test snippet",
                    "date": "Jul 23, 2026",
                }
            ]
        },
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )
    response = config.transform_search_response(
        raw_response=mock_response,
        logging_obj=None,
    )

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    assert headers == {"Content-Type": "application/json"}
    assert parsed_url.scheme == "https"
    assert parsed_url.netloc == "serpapi.com"
    assert parsed_url.path == "/search.json"
    assert query_params["api_key"] == ["test-api-key"]
    assert query_params["engine"] == ["google_light"]
    assert query_params["num"] == ["5"]
    assert query_params["gl"] == ["us"]
    assert query_params["hl"] == ["en"]
    assert "max_tokens_per_page" not in query_params
    assert "site:arxiv.org" in query_params["q"][0]
    assert "site:nature.com" in query_params["q"][0]
    assert json.loads(json.dumps(data)) == data
    assert response.object == "search"
    assert len(response.results) == 1
    assert response.results[0].title == "Test Result"
    assert response.results[0].url == "https://example.com/result"
    assert response.results[0].snippet == "Test snippet"
    assert response.results[0].date == "Jul 23, 2026"


def test_serpapi_api_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "alias-api-key")
    config = SerpApiSearchConfig()
    data = config.transform_search_request(query=["test", "query"], optional_params={})

    url = config.get_complete_url(
        api_base=None,
        optional_params={},
        data=data,
    )

    query_params = parse_qs(urlparse(url).query)
    assert query_params["api_key"] == ["alias-api-key"]
    assert query_params["engine"] == ["google"]
    assert query_params["q"] == ["test query"]


def test_serpapi_ui_friendly_name() -> None:
    assert SerpApiSearchConfig.ui_friendly_name() == "SerpApi"


def test_serpapi_caller_api_key_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERPAPI_KEY", "environment-api-key")
    config = SerpApiSearchConfig()
    data = config.transform_search_request(query="test query", optional_params={})

    url = config.get_complete_url(
        api_base=None,
        optional_params={},
        data=data,
        api_key="caller-api-key",
    )

    query_params = parse_qs(urlparse(url).query)
    assert query_params["api_key"] == ["caller-api-key"]


def test_serpapi_rejects_unencodable_passthrough_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERPAPI_KEY", "test-api-key")
    config = SerpApiSearchConfig()
    data = config.transform_search_request(
        query="test query",
        optional_params={"nested": {"x": 1}},
    )

    with pytest.raises(ValueError) as exc_info:
        config.get_complete_url(
            api_base=None,
            optional_params={},
            data=data,
        )

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == "Invalid SerpApi URL parameter value for: nested"


def test_serpapi_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SERPAPI_KEY is not set"):
        SerpApiSearchConfig().validate_environment(headers={})


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"organic_results": []},
        {
            "search_metadata": {"status": "Success"},
            "error": "Google Light hasn't returned any results for this query.",
        },
    ],
)
def test_serpapi_empty_results(payload: dict[str, object]) -> None:
    mock_response = httpx.Response(
        status_code=200,
        json=payload,
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    response = SerpApiSearchConfig().transform_search_response(
        raw_response=mock_response,
        logging_obj=None,
    )

    assert response.results == []


def test_serpapi_preserves_result_extra_fields() -> None:
    mock_response = httpx.Response(
        status_code=200,
        json={
            "organic_results": [
                {
                    "title": "Test Result",
                    "link": "https://example.com/result",
                    "snippet": "Test snippet",
                    "position": 1,
                    "displayed_link": "example.com",
                    "rich_snippet": {"top": {"extensions": ["Extra detail"]}},
                    "url": "https://wrong.example/result",
                    "last_updated": "Yesterday",
                }
            ]
        },
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    response = SerpApiSearchConfig().transform_search_response(
        raw_response=mock_response,
        logging_obj=None,
    )

    result = response.results[0]
    assert getattr(result, "position") == 1
    assert getattr(result, "displayed_link") == "example.com"
    assert getattr(result, "rich_snippet") == {"top": {"extensions": ["Extra detail"]}}
    assert result.url == "https://example.com/result"
    assert result.last_updated is None


def test_serpapi_max_results_truncates_response() -> None:
    config = SerpApiSearchConfig()
    config.transform_search_request(
        query="test query",
        optional_params={"max_results": 3},
    )
    mock_response = httpx.Response(
        status_code=200,
        json={
            "organic_results": [
                {
                    "title": f"Result {index}",
                    "link": f"https://example.com/{index}",
                    "snippet": f"Snippet {index}",
                }
                for index in range(5)
            ]
        },
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    response = config.transform_search_response(
        raw_response=mock_response,
        logging_obj=None,
    )

    assert len(response.results) == 3
    assert response.results[-1].title == "Result 2"


def test_serpapi_non_success_response_raises() -> None:
    mock_response = httpx.Response(
        status_code=429,
        text="rate limited",
        headers={"Retry-After": "60"},
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    with pytest.raises(BaseLLMException) as exc_info:
        SerpApiSearchConfig().transform_search_response(
            raw_response=mock_response,
            logging_obj=None,
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.message == "rate limited"


def test_serpapi_non_json_response_raises() -> None:
    mock_response = httpx.Response(
        status_code=200,
        text="<html>Bad Gateway</html>",
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    with pytest.raises(BaseLLMException, match="Expected a JSON body from SerpApi") as exc_info:
        SerpApiSearchConfig().transform_search_response(
            raw_response=mock_response,
            logging_obj=None,
        )

    assert exc_info.value.status_code == 200
    assert "Bad Gateway" in exc_info.value.message


def test_serpapi_error_envelope_raises() -> None:
    mock_response = httpx.Response(
        status_code=200,
        json={
            "search_metadata": {"status": "Error"},
            "error": "Invalid API key",
        },
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    with pytest.raises(BaseLLMException, match="Invalid API key") as exc_info:
        SerpApiSearchConfig().transform_search_response(
            raw_response=mock_response,
            logging_obj=None,
        )

    assert exc_info.value.status_code == 200


def test_serpapi_unrecognized_response_shape_raises() -> None:
    mock_response = httpx.Response(
        status_code=200,
        json=[],
        request=httpx.Request("GET", "https://serpapi.com/search.json"),
    )

    with pytest.raises(BaseLLMException, match="Unrecognized SerpApi response shape") as exc_info:
        SerpApiSearchConfig().transform_search_response(
            raw_response=mock_response,
            logging_obj=None,
        )

    assert exc_info.value.status_code == 200


def test_serpapi_search_cost_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    local_model_cost = TypeAdapter(dict[str, object]).validate_json(
        (Path(__file__).parents[4] / "model_prices_and_context_window.json").read_text()
    )
    monkeypatch.setattr(litellm, "model_cost", local_model_cost)
    model_info = get_model_info(
        model="serpapi/search",
        custom_llm_provider="serpapi",
        api_key="test-api-key",
    )

    assert model_info.get("input_cost_per_query") == 0.025
    assert model_info["litellm_provider"] == "serpapi"
    assert model_info["mode"] == "search"
