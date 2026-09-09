import json
from unittest.mock import Mock

import pytest

from litellm.llms.linkup.search.transformation import LinkupSearchConfig


def _config() -> LinkupSearchConfig:
    return LinkupSearchConfig()


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


def test_transform_search_request_without_dates_omits_both():
    data = _config().transform_search_request("q", {"max_results": 5})
    assert "fromDate" not in data
    assert "toDate" not in data


def test_transform_search_request_passes_through_unhandled_kwargs():
    """A param this function doesn't explicitly handle should still reach Linkup unchanged."""
    data = _config().transform_search_request("q", {"includeImages": True})
    assert data["includeImages"] is True


def test_transform_search_request_joins_list_query():
    assert _config().transform_search_request(["foo", "bar"], {})["q"] == "foo bar"


def test_transform_search_request_defaults_depth_and_output_type():
    data = _config().transform_search_request("q", {})
    assert data["depth"] == "standard"
    assert data["outputType"] == "searchResults"