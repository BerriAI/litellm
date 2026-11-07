from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.serpapi.search.transformation import SerpApiSearchConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


_MINIMAL_ORGANIC: Final = [
    {
        "position": 1,
        "title": "Coffee - Wikipedia",
        "link": "https://en.wikipedia.org/wiki/Coffee",
        "snippet": "Coffee is a beverage prepared from roasted coffee beans.",
    }
]


class TestSerpApiSearchConfig:
    def test_ui_friendly_name(self):
        assert SerpApiSearchConfig().ui_friendly_name() == "SerpApi"

    def test_get_http_method(self):
        assert SerpApiSearchConfig().get_http_method() == "GET"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()
        result = config.validate_environment({}, api_key="test_key")
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_without_api_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        config = SerpApiSearchConfig()
        with pytest.raises(ValueError, match="SERPAPI_API_KEY is not set"):
            config.validate_environment({})

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_basic(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="python programming",
            optional_params={},
            api_key="test_key",
        )

        assert "_serpapi_params" in result
        params = result["_serpapi_params"]
        assert params["q"] == "python programming"
        assert params["engine"] == "google"
        assert params["api_key"] == "test_key"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_default_engine_is_google(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="test",
            optional_params={},
            api_key="test_key",
        )

        assert result["_serpapi_params"]["engine"] == "google"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_explicit_engine_google_news(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="AI news",
            optional_params={"engine": "google_news"},
            api_key="test_key",
        )

        assert result["_serpapi_params"]["engine"] == "google_news"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_explicit_engine_google_shopping(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="laptop",
            optional_params={"engine": "google_shopping"},
            api_key="test_key",
        )

        assert result["_serpapi_params"]["engine"] == "google_shopping"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_max_results(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="test",
            optional_params={"max_results": 5},
            api_key="test_key",
        )

        assert result["_serpapi_params"]["num"] == 5

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_country(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="test",
            optional_params={"country": "US"},
            api_key="test_key",
        )

        assert result["_serpapi_params"]["gl"] == "us"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_domain_filter(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="machine learning",
            optional_params={"search_domain_filter": ["arxiv.org", "nature.com"]},
            api_key="test_key",
        )

        q = result["_serpapi_params"]["q"]
        assert "site:arxiv.org" in q
        assert "site:nature.com" in q
        assert "machine learning" in q

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_list_query(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query=["python", "programming"],
            optional_params={},
            api_key="test_key",
        )

        assert result["_serpapi_params"]["q"] == "python programming"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_transform_search_request_passthrough_extra_params(self, mock_secret):
        mock_secret.return_value = "test_key"
        config = SerpApiSearchConfig()

        result = config.transform_search_request(
            query="test",
            optional_params={"hl": "fr", "device": "mobile"},
            api_key="test_key",
        )

        params = result["_serpapi_params"]
        assert params["hl"] == "fr"
        assert params["device"] == "mobile"

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_get_complete_url_builds_query_string(self, mock_secret):
        mock_secret.return_value = None
        config = SerpApiSearchConfig()

        data: Final = {
            "_serpapi_params": {
                "engine": "google",
                "q": "test query",
                "api_key": "test_key",
            }
        }

        url = config.get_complete_url(api_base=None, optional_params={}, data=data)

        assert "https://serpapi.com/search.json?" in url
        assert "engine=google" in url
        assert "q=test+query" in url
        assert "api_key=test_key" in url

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_get_complete_url_without_data(self, mock_secret):
        mock_secret.return_value = None
        config = SerpApiSearchConfig()

        url = config.get_complete_url(api_base=None, optional_params={}, data=None)

        assert url == "https://serpapi.com/search.json"

    def test_transform_search_response_normalizes_organic_results(self):
        config = SerpApiSearchConfig()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "organic_results": [
                {
                    "position": 1,
                    "title": "Test Result 1",
                    "link": "https://example.com/1",
                    "snippet": "Snippet for result 1",
                    "date": "Jan 15, 2025",
                },
                {
                    "position": 2,
                    "title": "Test Result 2",
                    "link": "https://example.com/2",
                    "snippet": "Snippet for result 2",
                },
            ]
        }

        result = config.transform_search_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, SearchResponse)
        assert result.object == "search"
        assert len(result.results) == 2

        assert result.results[0].title == "Test Result 1"
        assert result.results[0].url == "https://example.com/1"
        assert result.results[0].snippet == "Snippet for result 1"
        assert result.results[0].date == "Jan 15, 2025"
        assert result.results[0].last_updated is None

        assert result.results[1].title == "Test Result 2"
        assert result.results[1].url == "https://example.com/2"
        assert result.results[1].date is None

    def test_transform_search_response_shopping_results(self):
        config = SerpApiSearchConfig()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "shopping_results": [
                {
                    "title": "Shopping Result 1",
                    "link": "https://example.com/shop",
                    "snippet": "Buy this thing",
                }
            ]
        }

        result = config.transform_search_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 1
        assert result.results[0].title == "Shopping Result 1"

    def test_transform_search_response_empty_results(self):
        config = SerpApiSearchConfig()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"organic_results": []}

        result = config.transform_search_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 0

    def test_transform_search_response_missing_organic_results_key(self):
        config = SerpApiSearchConfig()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"search_metadata": {"status": "Success"}}

        result = config.transform_search_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 0

    def test_transform_search_response_missing_optional_fields(self):
        config = SerpApiSearchConfig()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "organic_results": [
                {
                    "link": "https://example.com",
                }
            ]
        }

        result = config.transform_search_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert result.results[0].title == ""
        assert result.results[0].snippet == ""
        assert result.results[0].url == "https://example.com"
        assert result.results[0].date is None

    @patch("litellm.llms.serpapi.search.transformation.get_secret_str")
    def test_missing_api_key_in_transform_raises(self, mock_secret):
        mock_secret.return_value = None
        config = SerpApiSearchConfig()

        with pytest.raises(ValueError, match="SERPAPI_API_KEY is not set"):
            config.transform_search_request(
                query="test",
                optional_params={},
                api_key=None,
            )


class TestSerpApiIntegration:
    @pytest.mark.asyncio
    async def test_asearch_endpoint_and_auth(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic_results": _MINIMAL_ORGANIC,
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            response = await litellm.asearch(
                query="coffee",
                search_provider="serpapi",
                max_results=1,
            )

            assert mock_get.call_count == 1
            call_args = mock_get.call_args
            url: str = call_args.kwargs.get("url") or call_args.args[0]
            assert "serpapi.com/search.json" in url
            assert "api_key=test-api-key" in url
            assert "q=coffee" in url
            assert "engine=google" in url
            assert "num=1" in url

            assert response.object == "search"
            assert len(response.results) == 1
            assert response.results[0].title == "Coffee - Wikipedia"

    @pytest.mark.asyncio
    async def test_asearch_explicit_engine_selection(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"organic_results": []}

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            await litellm.asearch(
                query="AI news",
                search_provider="serpapi",
                engine="google_news",
            )

            call_args = mock_get.call_args
            url: str = call_args.kwargs.get("url") or call_args.args[0]
            assert "engine=google_news" in url

    @pytest.mark.asyncio
    async def test_asearch_google_shopping_engine(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"organic_results": []}

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            await litellm.asearch(
                query="laptop",
                search_provider="serpapi",
                engine="google_shopping",
            )

            call_args = mock_get.call_args
            url: str = call_args.kwargs.get("url") or call_args.args[0]
            assert "engine=google_shopping" in url

    @pytest.mark.asyncio
    async def test_asearch_empty_results(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            response = await litellm.asearch(
                query="xyznonexistent12345",
                search_provider="serpapi",
            )

            assert response.object == "search"
            assert len(response.results) == 0


class TestSerperRegressionWithSharedCode:
    @pytest.mark.asyncio
    async def test_serper_unaffected_by_serpapi_registration(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Serper Result",
                    "link": "https://example.com",
                    "snippet": "Snippet",
                }
            ]
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            response = await litellm.asearch(
                query="test",
                search_provider="serper",
            )

            assert mock_post.call_count == 1
            call_args = mock_post.call_args
            url: str = call_args.kwargs.get("url") or call_args.args[0]
            assert "google.serper.dev/search" in url

            headers = call_args.kwargs.get("headers", {})
            assert "X-API-KEY" in headers
            assert headers["X-API-KEY"] == "test-serper-key"

            assert len(response.results) == 1
            assert response.results[0].title == "Serper Result"


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)
