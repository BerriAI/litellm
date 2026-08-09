"""
Tests for Aliyun IQS UnifiedSearch API integration.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.abspath("../.."))

import litellm


class TestAliyunIQSSearch:
    """
    Tests for Aliyun IQS UnifiedSearch functionality with mocked network responses.
    """

    @pytest.mark.asyncio
    async def test_aliyun_iqs_search_request_payload(self):
        """
        Test that validates the IQS search request payload structure without making real API calls.
        """
        # Set environment variable for API key
        os.environ["ALIYUN_IQS_API_KEY"] = "test-api-key"

        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requestId": "test-request-id",
            "pageItems": [
                {
                    "title": "Test Result 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a test snippet for result 1",
                    "publishedTime": "2026-07-01",
                },
                {
                    "title": "Test Result 2",
                    "link": "https://example.com/2",
                    "snippet": "",
                    "summary": "Summary fallback for result 2",
                },
            ],
        }

        # Mock the httpx AsyncClient post method
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            # Make the search call
            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider="aliyun_iqs",
                max_results=5,
            )

            # Verify the post method was called once
            assert mock_post.call_count == 1

            # Get the actual call arguments
            call_args = mock_post.call_args

            # Verify URL
            assert call_args.kwargs["url"] == "https://cloud-iqs.aliyuncs.com/search/unified"

            # Verify headers contain Authorization
            headers = call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"

            # Verify request payload
            json_data = call_args.kwargs.get("json")
            assert json_data is not None
            assert json_data["query"] == "latest developments in AI"
            assert json_data["engineType"] == "Generic"
            assert json_data["advancedParams"] == {"numResults": 5}

            # Verify response structure
            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) == 2

            # Verify first result
            first_result = response.results[0]
            assert first_result.title == "Test Result 1"
            assert first_result.url == "https://example.com/1"
            assert first_result.snippet == "This is a test snippet for result 1"
            assert first_result.date == "2026-07-01"

            # Verify second result falls back to summary when snippet is empty
            second_result = response.results[1]
            assert second_result.snippet == "Summary fallback for result 2"


class TestAliyunIQSSearchTransformations:
    """Unit tests for AliyunIQSSearchConfig transformations (no network)."""

    def setup_method(self):
        from litellm.llms.aliyun_iqs.search.transformation import AliyunIQSSearchConfig

        self.config = AliyunIQSSearchConfig()

    def test_transform_search_request_list_query_and_max_results(self):
        data = self.config.transform_search_request(
            query=["latest", "AI", "news"],
            optional_params={"max_results": 10, "engineType": "GenericAdvanced", "timeRange": "OneWeek"},
        )
        assert data["query"] == "latest AI news"
        assert data["engineType"] == "GenericAdvanced"
        assert data["advancedParams"] == {"numResults": 10}
        assert data["timeRange"] == "OneWeek"

    def test_get_complete_url_appends_path(self):
        assert (
            self.config.get_complete_url(api_base=None, optional_params={})
            == "https://cloud-iqs.aliyuncs.com/search/unified"
        )
        assert (
            self.config.get_complete_url(api_base="https://cloud-iqs.aliyuncs.com/search/unified", optional_params={})
            == "https://cloud-iqs.aliyuncs.com/search/unified"
        )

    def test_validate_environment_missing_key_raises(self, monkeypatch):
        import pytest as _pytest

        monkeypatch.delenv("ALIYUN_IQS_API_KEY", raising=False)
        with _pytest.raises(ValueError, match="ALIYUN_IQS_API_KEY"):
            self.config.validate_environment(headers={})

    def test_transform_search_response_empty(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"requestId": "r1"}
        response = self.config.transform_search_response(raw_response=mock_resp, logging_obj=None)
        assert response.results == []
        assert response.object == "search"

    def test_max_results_merges_native_advanced_params(self):
        data = self.config.transform_search_request(
            query="q",
            optional_params={
                "max_results": 5,
                "advancedParams": {"startPublishedDate": "2026-01-01"},
            },
        )
        assert data["advancedParams"] == {
            "startPublishedDate": "2026-01-01",
            "numResults": 5,
        }

    def test_get_complete_url_normalizes_trailing_slash(self):
        assert (
            self.config.get_complete_url(api_base="https://cloud-iqs.aliyuncs.com/", optional_params={})
            == "https://cloud-iqs.aliyuncs.com/search/unified"
        )

    def test_error_body_raises_instead_of_empty_results(self):
        import pytest as _pytest

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errorCode": "Retrieval.TestUserPeriodExpired",
            "errorMessage": "The test period has expired",
        }
        with _pytest.raises(ValueError, match="TestUserPeriodExpired"):
            self.config.transform_search_response(raw_response=mock_resp, logging_obj=None)
