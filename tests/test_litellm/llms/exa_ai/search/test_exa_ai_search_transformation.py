from typing import Final, TypedDict
from unittest.mock import Mock

from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig


class _ExaSearchResult(TypedDict, total=False):
    title: str
    url: str
    publishedDate: str
    text: str
    highlights: list[str]
    summary: str


class _ExaSearchPayload(TypedDict):
    results: list[_ExaSearchResult]


def _config() -> ExaAISearchConfig:
    return ExaAISearchConfig()


def _resp(payload: _ExaSearchPayload) -> Mock:
    r: Final = Mock()
    r.json.return_value = payload
    return r


def _result(**overrides: str | list[str]) -> _ExaSearchResult:
    base: Final[_ExaSearchResult] = {
        "title": "Lion Finance Group (LON:BGEO) Q1 2026 Earnings Call Transcript & Audio",
        "url": "https://stockanalysis.com/quote/lon/BGEO/transcripts/557900-q1-2026/",
        "publishedDate": "2026-05-01T00:00:00.000Z",
    }
    return {**base, **overrides}  # type: ignore[return-value]


def test_transform_search_response_uses_text_when_present():
    resp = _config().transform_search_response(
        _resp({"results": [_result(text="net interest margin is expected to stabilize")]}),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == "net interest margin is expected to stabilize"


def test_transform_search_response_falls_back_to_highlights_when_text_missing():
    """Exa returns no "text" field when only contents.highlights is requested."""
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [
                    _result(
                        highlights=[
                            "Net interest margin outlook remains stable for the quarter.",
                            "Management expects continued deposit growth.",
                        ]
                    )
                ]
            }
        ),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == (
        "Net interest margin outlook remains stable for the quarter.\n\n"
        "Management expects continued deposit growth."
    )


def test_transform_search_response_falls_back_to_summary_when_text_and_highlights_missing():
    """Exa returns no "text" field when only contents.summary is requested."""
    resp = _config().transform_search_response(
        _resp({"results": [_result(summary="The outlook for net interest margin is positive.")]}),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == "The outlook for net interest margin is positive."


def test_transform_search_response_text_takes_priority_over_highlights_and_summary():
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [
                    _result(
                        text="full text content",
                        highlights=["a highlight"],
                        summary="a summary",
                    )
                ]
            }
        ),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == "full text content"


def test_transform_search_response_highlights_take_priority_over_summary():
    resp = _config().transform_search_response(
        _resp({"results": [_result(highlights=["a highlight"], summary="a summary")]}),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == "a highlight"


def test_transform_search_response_empty_string_when_no_content_fields_present():
    resp = _config().transform_search_response(_resp({"results": [_result()]}), logging_obj=Mock())
    assert resp.results[0].snippet == ""


def test_transform_search_response_empty_highlights_list_falls_back_to_summary():
    resp = _config().transform_search_response(
        _resp({"results": [_result(highlights=[], summary="a summary")]}),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == "a summary"
