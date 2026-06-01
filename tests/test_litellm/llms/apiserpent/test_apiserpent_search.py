"""
Tests for APISerpent search API integration (quick + deep search).
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

import litellm
from litellm.llms.apiserpent.search.defaults import APISerpentSearchParams
from litellm.llms.apiserpent.search.transformation import APISerpentSearchConfig
from litellm.llms.base_llm.search.transformation import SearchResponse


def _params(config, query, optional_params):
    return config.transform_search_request(
        query=query, optional_params=optional_params
    )["_apiserpent_params"]


class TestAPISerpentDefaults:
    def test_defaults_applied(self):
        params = APISerpentSearchParams().to_request_params()
        assert params["engine"] == "google"
        assert params["country"] == "us"
        assert params["num"] == 10
        assert params["format"] == "full"
        assert "freshness" not in params
        assert "pixel_position" not in params

    def test_bool_lowercased(self):
        params = APISerpentSearchParams(pixel_position=True).to_request_params()
        assert params["pixel_position"] == "true"

    @pytest.mark.parametrize("num", [0, 101, 500])
    def test_num_out_of_range_raises(self, num):
        with pytest.raises(ValueError, match="num must be between 1 and 100"):
            APISerpentSearchParams(num=num)

    @pytest.mark.parametrize("pages", [0, 11, 50])
    def test_pages_out_of_range_raises(self, pages):
        with pytest.raises(ValueError, match="pages must be between 1 and 10"):
            APISerpentSearchParams(pages=pages)

    def test_valid_bounds_accepted(self):
        params = APISerpentSearchParams(num=100, pages=10).to_request_params()
        assert params["num"] == 100
        assert params["pages"] == 10


class TestAPISerpentConfig:
    def test_ui_friendly_name(self):
        assert APISerpentSearchConfig().ui_friendly_name() == "APISerpent"

    def test_get_http_method(self):
        assert APISerpentSearchConfig().get_http_method() == "GET"

    @patch("litellm.llms.apiserpent.search.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_get_secret):
        mock_get_secret.return_value = None
        headers = APISerpentSearchConfig().validate_environment(
            {}, api_key="test-api-key"
        )
        assert headers["X-API-Key"] == "test-api-key"
        assert headers["Content-Type"] == "application/json"

    @patch("litellm.llms.apiserpent.search.transformation.get_secret_str")
    def test_validate_environment_without_api_key(self, mock_get_secret):
        mock_get_secret.return_value = None
        with pytest.raises(ValueError, match="APISERPENT_API_KEY is not set"):
            APISerpentSearchConfig().validate_environment({})

    def test_transform_request_basic_applies_defaults(self):
        params = _params(APISerpentSearchConfig(), "test query", {})
        assert params["q"] == "test query"
        assert params["engine"] == "google"
        assert params["num"] == 10

    def test_transform_request_list_query_joined(self):
        assert _params(APISerpentSearchConfig(), ["foo", "bar"], {})["q"] == "foo bar"

    def test_quick_num_clamped(self):
        config = APISerpentSearchConfig()
        assert _params(config, "q", {"max_results": 250})["num"] == 100
        assert _params(config, "q", {"max_results": 0})["num"] == 1

    def test_deep_num_floor_is_10(self):
        config = APISerpentSearchConfig()
        params = _params(config, "q", {"deep": True, "max_results": 5})
        assert params["num"] == 10

    def test_country_lowercased(self):
        assert (
            _params(APISerpentSearchConfig(), "q", {"country": "US"})["country"] == "us"
        )

    def test_engine_and_optional_passthrough(self):
        params = _params(
            APISerpentSearchConfig(),
            "q",
            {"engine": "bing", "language": "es", "freshness": "d", "safe": "strict"},
        )
        assert params["engine"] == "bing"
        assert params["language"] == "es"
        assert params["freshness"] == "d"
        assert params["safe"] == "strict"

    def test_pixel_position_passthrough_lowercased(self):
        params = _params(APISerpentSearchConfig(), "q", {"pixel_position": True})
        assert params["pixel_position"] == "true"

    def test_domain_filter(self):
        params = _params(
            APISerpentSearchConfig(),
            "machine learning",
            {"search_domain_filter": ["arxiv.org", "nature.com"]},
        )
        assert "site:arxiv.org" in params["q"]
        assert "site:nature.com" in params["q"]
        assert "machine learning" in params["q"]

    def test_get_complete_url_quick_path(self):
        config = APISerpentSearchConfig()
        data = {"_apiserpent_params": {"q": "test", "num": 5}}
        url = config.get_complete_url(api_base=None, optional_params={}, data=data)
        parsed = urlparse(url)
        assert (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            == "https://apiserpent.com/api/search/quick"
        )
        assert parse_qs(parsed.query)["q"] == ["test"]

    def test_get_complete_url_deep_path(self):
        config = APISerpentSearchConfig()
        data = {"_apiserpent_params": {"q": "test"}}
        url = config.get_complete_url(
            api_base=None, optional_params={"deep": True}, data=data
        )
        parsed = urlparse(url)
        assert (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            == "https://apiserpent.com/api/search"
        )

    def test_explicit_api_base_swaps_host_and_keeps_routing(self):
        config = APISerpentSearchConfig()
        url = config.get_complete_url(
            api_base="https://staging.apiserpent.com",
            optional_params={"deep": True},
            data={"_apiserpent_params": {"q": "x"}},
        )
        parsed = urlparse(url)
        assert (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            == "https://staging.apiserpent.com/api/search"
        )

    def test_get_complete_url_is_idempotent(self):
        """The handler re-invokes get_complete_url with the resolved URL as api_base."""
        config = APISerpentSearchConfig()
        resolved = config.get_complete_url(
            api_base=None, optional_params={"deep": True}, data=None
        )
        again = config.get_complete_url(
            api_base=resolved,
            optional_params={"deep": True},
            data={"_apiserpent_params": {"q": "x"}},
        )
        assert again == "https://apiserpent.com/api/search?q=x"
        assert "/api/search/api/search" not in again

    def test_transform_response_full_format(self):
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "success": True,
            "results": {
                "organic": [
                    {"title": "R1", "url": "https://example.com/1", "snippet": "S1"},
                    {"title": "R2", "url": "https://example.com/2", "snippet": "S2"},
                ]
            },
        }
        response = APISerpentSearchConfig().transform_search_response(
            raw_response=raw_response, logging_obj=None
        )
        assert isinstance(response, SearchResponse)
        assert len(response.results) == 2
        assert response.results[0].title == "R1"
        assert response.results[0].url == "https://example.com/1"

    def test_transform_response_simple_format(self):
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "success": True,
            "results": [{"position": 1, "title": "R1", "url": "https://example.com/1"}],
        }
        response = APISerpentSearchConfig().transform_search_response(
            raw_response=raw_response, logging_obj=None
        )
        assert len(response.results) == 1
        assert response.results[0].title == "R1"

    def test_transform_response_empty(self):
        raw_response = MagicMock()
        raw_response.json.return_value = {"success": True, "results": {}}
        response = APISerpentSearchConfig().transform_search_response(
            raw_response=raw_response, logging_obj=None
        )
        assert len(response.results) == 0

    def test_transform_response_null_results(self):
        """An error response with `results: null` must not raise."""
        raw_response = MagicMock()
        raw_response.json.return_value = {"success": False, "results": None}
        response = APISerpentSearchConfig().transform_search_response(
            raw_response=raw_response, logging_obj=None
        )
        assert response.results == []


class TestAPISerpentSearchIntegration:
    @staticmethod
    def _mock_response():
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "results": {
                "organic": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "snippet": "A snippet",
                    }
                ]
            },
        }
        return mock_response

    @pytest.mark.asyncio
    async def test_asearch_quick_default(self):
        os.environ["APISERPENT_API_KEY"] = "test-api-key"
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = self._mock_response()

            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider="apiserpent",
                max_results=5,
                country="US",
            )

            parsed = urlparse(mock_get.call_args.kwargs["url"])
            assert (
                f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                == "https://apiserpent.com/api/search/quick"
            )
            qs = parse_qs(parsed.query)
            assert qs["q"] == ["latest developments in AI"]
            assert qs["num"] == ["5"]
            assert qs["country"] == ["us"]
            assert mock_get.call_args.kwargs["headers"]["X-API-Key"] == "test-api-key"

            assert response.object == "search"
            assert response.results[0].title == "Test Result"

    @pytest.mark.asyncio
    async def test_asearch_deep(self):
        os.environ["APISERPENT_API_KEY"] = "test-api-key"
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = self._mock_response()

            await litellm.asearch(
                query="climate research",
                search_provider="apiserpent",
                deep=True,
                max_results=40,
            )

            parsed = urlparse(mock_get.call_args.kwargs["url"])
            assert (
                f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                == "https://apiserpent.com/api/search"
            )
            assert parse_qs(parsed.query)["num"] == ["40"]
