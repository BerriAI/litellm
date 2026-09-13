from unittest.mock import Mock

import pytest

from litellm.llms.linkup.search.transformation import LinkupSearchConfig


def _config() -> LinkupSearchConfig:
    return LinkupSearchConfig()


def _resp(payload: dict) -> Mock:
    r = Mock()
    r.json.return_value = payload
    return r


def _result(**overrides):
    base = {
        "type": "text",
        "name": "Test Title",
        "url": "https://example.com",
        "content": "Test content",
    }
    return {**base, **overrides}




# --- transform_search_request: date filtering ---


def test_transform_search_request_maps_start_date_to_from_date():
    data = _config().transform_search_request("q", {"start_date": "1999-03-20"})
    assert data["fromDate"] == "1999-03-20"
    assert "start_date" not in data


def test_transform_search_request_maps_end_date_to_to_date():
    data = _config().transform_search_request("q", {"end_date": "1999-04-20"})
    assert data["toDate"] == "1999-04-20"
    assert "end_date" not in data


def test_transform_search_request_date_range_together():
    data = _config().transform_search_request(
        "q", {"start_date": "1999-03-20", "end_date": "1999-04-20"}
    )
    assert data["fromDate"] == "1999-03-20"
    assert data["toDate"] == "1999-04-20"


# --- transform_search_request: passthrough ---


def test_transform_search_request_does_not_duplicate_consumed_params():
    """Regression test: a translated param (start_date -> fromDate) must not
    also leak through raw via the passthrough path."""
    data = _config().transform_search_request(
        "q", {"start_date": "1999-03-20", "max_results": 5, "includeImages": True}
    )
    assert "start_date" not in data
    assert "max_results" not in data
    assert data["fromDate"] == "1999-03-20"
    assert data["maxResults"] == 5
    assert data["includeImages"] is True


# --- transform_search_response ---


def test_transform_search_response_image_result_falls_back_to_url_for_title():
    resp = _resp({"results": [_result(type="image", name="", url="https://example.com/img.png")]})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.title == "https://example.com/img.png"


def test_transform_search_response_image_result_uses_name_when_present():
    resp = _resp({"results": [_result(type="image", name="Alt text")]})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.title == "Alt text"
