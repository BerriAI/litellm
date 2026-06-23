from unittest.mock import Mock

from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig


def _config() -> ExaAISearchConfig:
    return ExaAISearchConfig()


def _resp(payload):
    r = Mock()
    r.json.return_value = payload
    return r


def test_transform_response_maps_canonical_fields():
    """text -> snippet, publishedDate -> date (existing behavior preserved)."""
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [
                    {
                        "title": "T",
                        "url": "https://e.com",
                        "text": "body text",
                        "publishedDate": "2026-06-17T17:06:28.000Z",
                    }
                ],
            }
        ),
        logging_obj=Mock(),
    )
    assert len(resp.results) == 1
    result = resp.results[0]
    assert result.title == "T"
    assert result.url == "https://e.com"
    assert result.snippet == "body text"
    assert result.date == "2026-06-17T17:06:28.000Z"
    assert result.last_updated is None


def test_snippet_falls_back_to_highlights_when_text_absent():
    """When the request asks for `contents.highlights` (no `text`), snippet must
    not be empty — it should fall back to the highlights content."""
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [
                    {
                        "title": "T",
                        "url": "https://e.com",
                        "highlights": ["first highlight", "second highlight"],
                    }
                ],
            }
        ),
        logging_obj=Mock(),
    )
    snippet = resp.results[0].snippet
    assert "first highlight" in snippet
    assert "second highlight" in snippet


def test_text_preferred_over_highlights_for_snippet():
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [
                    {
                        "title": "T",
                        "url": "https://e.com",
                        "text": "full text",
                        "highlights": ["a highlight"],
                    }
                ],
            }
        ),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == "full text"


def test_preserves_per_result_extra_fields():
    """highlights / image / author / favicon / score must survive on each result."""
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [
                    {
                        "title": "T",
                        "url": "https://e.com",
                        "highlights": ["h1"],
                        "image": "https://e.com/img.jpg",
                        "author": "Hisse.net",
                        "favicon": "https://e.com/fav.png",
                        "score": 0.42,
                    }
                ],
            }
        ),
        logging_obj=Mock(),
    )
    dumped = resp.results[0].model_dump()
    assert dumped["highlights"] == ["h1"]
    assert dumped["image"] == "https://e.com/img.jpg"
    assert dumped["author"] == "Hisse.net"
    assert dumped["favicon"] == "https://e.com/fav.png"
    assert dumped["score"] == 0.42


def test_preserves_top_level_output_summary_and_grounding():
    """The deep-search synthesized summary (`output.content`) and its grounding
    citations must survive at the top level of the response."""
    output = {
        "content": "Latest news items found for GESAN ...",
        "grounding": [
            {
                "field": "content",
                "citations": [{"url": "https://e.com", "title": "T"}],
                "confidence": "high",
            }
        ],
    }
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [{"title": "T", "url": "https://e.com", "text": "x"}],
                "output": output,
            }
        ),
        logging_obj=Mock(),
    )
    dumped = resp.model_dump()
    assert dumped["output"] == output


def test_preserves_top_level_cost_and_request_metadata():
    resp = _config().transform_search_response(
        _resp(
            {
                "results": [{"title": "T", "url": "https://e.com", "text": "x"}],
                "costDollars": {"total": 0.012},
                "requestId": "ea4710a59e72ae0ed80521584c7a4814",
                "resolvedSearchType": "deep",
                "searchTime": 6987,
            }
        ),
        logging_obj=Mock(),
    )
    dumped = resp.model_dump()
    assert dumped["costDollars"] == {"total": 0.012}
    assert dumped["requestId"] == "ea4710a59e72ae0ed80521584c7a4814"
    assert dumped["resolvedSearchType"] == "deep"
    assert dumped["searchTime"] == 6987


def test_object_field_is_search():
    resp = _config().transform_search_response(
        _resp({"results": [{"title": "T", "url": "https://e.com", "text": "x"}]}),
        logging_obj=Mock(),
    )
    assert resp.object == "search"


def test_empty_results():
    resp = _config().transform_search_response(
        _resp({"results": []}), logging_obj=Mock()
    )
    assert resp.results == []


def test_missing_results_key():
    resp = _config().transform_search_response(_resp({}), logging_obj=Mock())
    assert resp.results == []


def test_non_list_results_is_safe():
    resp = _config().transform_search_response(
        _resp({"results": {"unexpected": "shape"}}), logging_obj=Mock()
    )
    assert resp.results == []
