"""
Tests for Exa AI search response transformation.

Regression coverage for https://github.com/BerriAI/litellm/issues/37502 and
https://github.com/BerriAI/litellm/issues/36905: Exa returns each requested
content mode (`text`, `highlights`, `summary`) in its own response field and
omits `text` entirely when only `highlights` or `summary` were requested, so
reading only `text` silently dropped billed content and returned an empty
snippet. `highlightScores` and `score` were dropped the same way.
"""

import json
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig


def _raw_response(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode(),
        request=httpx.Request("POST", "https://api.exa.ai/search"),
    )


@pytest.fixture
def config() -> ExaAISearchConfig:
    return ExaAISearchConfig()


class TestExaAISnippetFallback:
    def test_text_only_produces_original_five_key_result(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "text": "full page text",
                            "publishedDate": "2026-01-01T00:00:00.000Z",
                        }
                    ]
                }
            ),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.model_dump(exclude_none=True) == {
            "title": "t",
            "url": "https://example.com",
            "snippet": "full page text",
            "date": "2026-01-01T00:00:00.000Z",
        }

    def test_highlights_fill_snippet_and_are_attached_when_text_absent(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "highlights": ["first highlight", "second highlight"],
                            "highlightScores": [0.9, 0.7],
                        }
                    ]
                }
            ),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == "first highlight\n\nsecond highlight"
        assert result.highlights == ("first highlight", "second highlight")
        assert result.highlight_scores == (0.9, 0.7)

    def test_summary_fills_snippet_when_text_and_highlights_absent(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "summary": "a short summary"}]}),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == "a short summary"
        assert result.summary == "a short summary"

    def test_text_wins_over_highlights_and_summary(self, config: ExaAISearchConfig) -> None:
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
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == "full page text"
        assert result.highlights == ("a highlight",)
        assert result.summary == "a summary"

    def test_highlights_win_over_summary_for_snippet(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "highlights": ["a highlight"], "summary": "a summary"}]}),
            logging_obj=Mock(),
        )

        assert response.results[0].snippet == "a highlight"

    def test_empty_highlights_list_falls_through_to_summary(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "highlights": [], "summary": "a summary"}]}),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == "a summary"
        assert result.highlights == ()

    def test_highlights_of_only_empty_strings_falls_through_to_summary(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": "t", "url": "https://example.com", "highlights": ["", ""], "summary": "a real summary"}]}
            ),
            logging_obj=Mock(),
        )

        assert response.results[0].snippet == "a real summary"

    def test_empty_string_when_no_content_fields_present(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com"}]}),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == ""
        assert result.model_dump(exclude_none=True) == {
            "title": "t",
            "url": "https://example.com",
            "snippet": "",
        }

    def test_explicit_null_highlights_and_highlight_scores_do_not_crash(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "text": "full page text",
                            "highlights": None,
                            "highlightScores": None,
                        }
                    ]
                }
            ),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == "full page text"
        assert "highlights" not in result.model_dump(exclude_none=True)
        assert "highlight_scores" not in result.model_dump(exclude_none=True)

    def test_highlights_as_a_bare_string_is_not_iterated_character_by_character(
        self, config: ExaAISearchConfig
    ) -> None:
        """A malformed response sending `highlights` as a string, not a list, must not be
        silently treated as an iterable of characters."""
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "highlights": "abc"}]}),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == ""
        assert "highlights" not in result.model_dump(exclude_none=True)

    def test_highlights_list_with_non_string_items_does_not_crash(self, config: ExaAISearchConfig) -> None:
        """Non-string items in `highlights` fall out of the joined snippet (since a snippet
        must be a string) but are preserved as-is on the raw `highlights` field, not
        dropped, so as not to desync `highlightScores`' positional correspondence."""
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": "t", "url": "https://example.com", "highlights": [1, 2, 3], "summary": "a summary"}]}
            ),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == "a summary"
        assert result.highlights == (1, 2, 3)

    def test_highlights_and_highlight_scores_stay_positionally_paired_when_one_has_a_bad_item(
        self, config: ExaAISearchConfig
    ) -> None:
        """highlightScores[i] is Exa's relevance score for highlights[i]; a malformed entry
        in only one of the two arrays must not silently desync that pairing by dropping an
        entry from one array but not the other."""
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "highlights": ["a", 42, "b"],
                            "highlightScores": [0.9, 0.8, 0.7],
                        }
                    ]
                }
            ),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.highlights == ("a", 42, "b")
        assert result.highlight_scores == (0.9, 0.8, 0.7)

    def test_non_string_text_and_summary_do_not_crash(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "text": 12345}]}),
            logging_obj=Mock(),
        )
        assert response.results[0].snippet == ""

        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "summary": {"nested": "x"}}]}),
            logging_obj=Mock(),
        )
        assert response.results[0].snippet == ""

    def test_non_string_title_url_and_date_do_not_crash(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": 123, "url": ["not", "a", "url"], "text": "hi", "publishedDate": 20260101}]}
            ),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.title == ""
        assert result.url == ""
        assert result.date is None
        assert result.snippet == "hi"

    def test_explicit_null_results_does_not_crash(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": None}),
            logging_obj=Mock(),
        )

        assert response.results == []

    def test_null_entry_within_results_list_does_not_crash(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [None, {"title": "t", "url": "https://example.com", "text": "hi"}]}),
            logging_obj=Mock(),
        )

        assert len(response.results) == 1
        assert response.results[0].snippet == "hi"

    def test_whitespace_only_highlights_fall_through_to_summary(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {"results": [{"title": "t", "url": "https://example.com", "highlights": [" ", "\t"], "summary": "a real summary"}]}
            ),
            logging_obj=Mock(),
        )

        assert response.results[0].snippet == "a real summary"


class TestExaAIScore:
    def test_neural_search_score_is_attached(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "text": "full page text",
                            "score": 0.9438,
                        }
                    ]
                }
            ),
            logging_obj=Mock(),
        )

        assert response.results[0].score == 0.9438

    def test_score_absent_when_not_returned(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "text": "full page text"}]}),
            logging_obj=Mock(),
        )

        assert "score" not in response.results[0].model_dump(exclude_none=True)

    def test_zero_score_is_attached_not_treated_as_absent(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response({"results": [{"title": "t", "url": "https://example.com", "text": "full page text", "score": 0.0}]}),
            logging_obj=Mock(),
        )

        assert response.results[0].score == 0.0

    def test_score_highlights_and_highlight_scores_all_attached_together(self, config: ExaAISearchConfig) -> None:
        response = config.transform_search_response(
            raw_response=_raw_response(
                {
                    "results": [
                        {
                            "title": "t",
                            "url": "https://example.com",
                            "highlights": ["a highlight", "another highlight"],
                            "highlightScores": [0.95, 0.42],
                            "score": 0.87,
                            "publishedDate": "2026-01-01T00:00:00.000Z",
                        }
                    ]
                }
            ),
            logging_obj=Mock(),
        )

        result = response.results[0]
        assert result.snippet == "a highlight\n\nanother highlight"
        assert result.highlights == ("a highlight", "another highlight")
        assert result.highlight_scores == (0.95, 0.42)
        assert result.score == 0.87
        assert result.date == "2026-01-01T00:00:00.000Z"
