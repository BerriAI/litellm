"""
Tests for Exa AI search response transformation.

Regression coverage for https://github.com/BerriAI/litellm/issues/36905:
Exa returns each requested content mode in its own response field and omits
`text` entirely when only `highlights` or `summary` were requested, so reading
only `text` silently dropped billed content and returned an empty snippet.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig


def _raw_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode(),
        request=httpx.Request("POST", "https://api.exa.ai/search"),
    )


@pytest.fixture
def config() -> ExaAISearchConfig:
    return ExaAISearchConfig()


class TestExaAISnippetFallback:
    def test_text_is_used_when_present(self, config):
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": "t", "url": "https://example.com", "text": "full page text"}]}
            ),
            logging_obj=MagicMock(),
        )

        assert response.results[0].snippet == "full page text"

    def test_highlights_fill_snippet_when_text_absent(self, config):
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "highlights": ["first highlight", "second highlight"],
                        }
                    ]
                }
            ),
            logging_obj=MagicMock(),
        )

        assert response.results[0].snippet == "first highlight\n\nsecond highlight"
        assert response.results[0].highlights == ["first highlight", "second highlight"]

    def test_summary_fills_snippet_when_text_and_highlights_absent(self, config):
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": "t", "url": "https://example.com", "summary": "a short summary"}]}
            ),
            logging_obj=MagicMock(),
        )

        assert response.results[0].snippet == "a short summary"
        assert response.results[0].summary == "a short summary"

    def test_text_wins_over_highlights_and_summary(self, config):
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "text": "full page text",
                            "highlights": ["a highlight"],
                            "summary": "a summary",
                        }
                    ]
                }
            ),
            logging_obj=MagicMock(),
        )

        result = response.results[0]
        assert result.snippet == "full page text"
        assert result.highlights == ["a highlight"]
        assert result.summary == "a summary"

    def test_empty_highlights_list_falls_through_to_summary(self, config):
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": "t", "url": "https://example.com", "highlights": [], "summary": "a summary"}]}
            ),
            logging_obj=MagicMock(),
        )

        assert response.results[0].snippet == "a summary"

    def test_no_content_modes_yields_empty_snippet_and_no_extra_fields(self, config):
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": "t", "url": "https://example.com", "publishedDate": "2026-08-14T00:00:00Z"}]}
            ),
            logging_obj=MagicMock(),
        )

        result = response.results[0]
        assert result.snippet == ""
        assert result.date == "2026-08-14T00:00:00Z"
        assert not hasattr(result, "highlights")
        assert not hasattr(result, "summary")
