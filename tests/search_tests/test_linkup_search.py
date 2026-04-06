"""
Tests for Linkup Search API integration.
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest


@pytest.mark.skip(reason="Local only tested search providers")
class TestLinkupSearch(BaseSearchTest):
    """
    E2E tests for Linkup Search functionality that make real API calls.
    Inherits from BaseSearchTest to run standard search tests.
    """

    def get_search_provider(self) -> str:
        """
        Return search_provider for Linkup Search.
        """
        return "linkup"


class TestLinkupSearchTransformation:
    """
    Unit tests for Linkup Search request/response transformation with mocked responses.
    """

    def test_linkup_search_request_transformation(self):
        """
        Test that validates the Linkup search request is correctly transformed from
        unified params to Linkup API format.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "type": "text",
                    "name": "Test Title",
                    "url": "https://example.com",
                    "content": "Test content",
                }
            ]
        }

        with patch.dict(os.environ, {"LINKUP_API_KEY": "test-api-key"}):
            with patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
                return_value=mock_response,
            ) as mock_post:
                litellm.search(
                    query="test query",
                    search_provider="linkup",
                    max_results=10,
                    search_domain_filter=["arxiv.org", "nature.com"],
                )

                assert mock_post.called
                call_kwargs = mock_post.call_args.kwargs
                request_body = call_kwargs.get("json")

                # Verify request transformation
                assert request_body is not None
                assert request_body["q"] == "test query"
                assert request_body["maxResults"] == 10
                assert request_body["depth"] == "standard"
                assert request_body["outputType"] == "searchResults"
                assert request_body["includeDomains"] == ["arxiv.org", "nature.com"]

    def test_linkup_search_response_transformation(self):
        """
        Test that validates the Linkup API response is correctly transformed to
        the unified SearchResponse format.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "type": "text",
                    "name": "Microsoft 2024 Annual Report",
                    "url": "https://www.microsoft.com/investor/reports/ar24/index.html",
                    "content": "Highlights from fiscal year 2024: Microsoft Cloud revenue increased 23% to $137.4 billion.",
                },
                {
                    "type": "text",
                    "name": "Another Result",
                    "url": "https://example.com/page",
                    "content": "Some other content",
                },
            ]
        }

        with patch.dict(os.environ, {"LINKUP_API_KEY": "test-api-key"}):
            with patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
                return_value=mock_response,
            ):
                response = litellm.search(
                    query="Microsoft revenue", search_provider="linkup"
                )

                # Verify response transformation
                assert response.object == "search"
                assert len(response.results) == 2

                first_result = response.results[0]
                assert first_result.title == "Microsoft 2024 Annual Report"
                assert (
                    first_result.url
                    == "https://www.microsoft.com/investor/reports/ar24/index.html"
                )
                assert "Microsoft Cloud revenue" in first_result.snippet
