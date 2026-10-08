"""
Tests for Linkup Search API integration.
"""

import datetime
import os
from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.linkup.search.transformation import LinkupSearchConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


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
                response = litellm.search(query="Microsoft revenue", search_provider="linkup")

                # Verify response transformation
                assert response.object == "search"
                assert len(response.results) == 2

                first_result = response.results[0]
                assert first_result.title == "Microsoft 2024 Annual Report"
                assert first_result.url == "https://www.microsoft.com/investor/reports/ar24/index.html"
                assert "Microsoft Cloud revenue" in first_result.snippet


def _transform(payload: object) -> SearchResponse:
    logging_obj = Logging(
        model="linkup/search",
        messages=[],
        stream=False,
        call_type="search",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    return LinkupSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload), logging_obj=logging_obj
    )


def test_linkup_text_and_image_results_keep_provider_order_and_fields():
    response = _transform(
        {
            "results": [
                {"type": "text", "name": "Annual Report", "url": "https://example.com/report", "content": "Revenue up"},
                {"type": "image", "name": "Chart", "url": "https://example.com/chart.png", "content": "A chart"},
                {"type": "image", "url": "https://example.com/logo.png"},
                {"name": "Untyped", "url": "https://example.com/untyped", "content": "Defaults to text"},
            ]
        }
    )

    assert response.results == [
        SearchResult(title="Annual Report", url="https://example.com/report", snippet="Revenue up"),
        SearchResult(title="Chart", url="https://example.com/chart.png", snippet="A chart"),
        SearchResult(title="https://example.com/logo.png", url="https://example.com/logo.png", snippet=""),
        SearchResult(title="Untyped", url="https://example.com/untyped", snippet="Defaults to text"),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answer": "A sourced answer without a results array"},
        {"results": []},
        {"results": ""},
        {"results": {}},
        {"results": [{"type": "video", "name": "Clip", "url": "https://example.com/clip"}]},
    ],
)
def test_linkup_response_without_text_or_image_results_is_empty(payload):
    response = _transform(payload)

    assert response.object == "search"
    assert response.results == []


def test_linkup_result_missing_optional_fields_defaults_to_empty_strings():
    response = _transform(
        {"results": [{"type": "text", "favicon": "https://example.com/favicon.ico"}, {"type": "image"}]}
    )

    assert response.results == [
        SearchResult(title="", url="", snippet=""),
        SearchResult(title="", url="", snippet=""),
    ]


@pytest.mark.parametrize(
    "result",
    [
        {"type": "text", "name": None, "url": "https://example.com", "content": "c"},
        {"type": "text", "name": "Title", "url": 7, "content": "c"},
        {"type": "image", "name": "Title", "url": "https://example.com", "content": None},
    ],
)
def test_linkup_result_with_non_string_field_is_rejected(result):
    with pytest.raises(ValidationError):
        _transform({"results": [result]})


@pytest.mark.parametrize("result_type", ["text", "image"])
def test_linkup_result_with_several_non_string_fields_reports_every_field(result_type):
    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": [{"type": result_type, "name": 7, "url": None, "content": ["content"]}]})

    assert [error["loc"] for error in exc_info.value.errors()] == [("title",), ("url",), ("snippet",)]


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)
