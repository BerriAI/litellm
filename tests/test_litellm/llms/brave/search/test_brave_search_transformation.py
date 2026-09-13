from unittest.mock import Mock

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.brave.search.transformation import BraveSearchConfig, to_yyyy_mm_dd


def _config() -> BraveSearchConfig:
    return BraveSearchConfig()


def _resp(payload: dict, params: dict | None = None) -> Mock:
    """Mock httpx.Response with the .json() and .request.url.params shape
    Brave's transform_search_response actually reads."""
    r = Mock()
    r.json.return_value = payload
    req = Mock()
    req.url = Mock()
    req.url.params = params or {}
    r.request = req
    return r


def _result(**overrides):
    base = {
        "title": "Test Title",
        "url": "https://example.com",
        "description": "Test description",
        "page_age": None,
        "age": None,
        "fetched_content_timestamp": None,
    }
    return {**base, **overrides}


# --- transform_search_request ---


def test_transform_search_request_country_is_forwarded():
    """Regression test: the docstring always claimed country -> country, but
    there was previously no code path that actually implemented it — it
    relied on shared-list passthrough, which excluded it."""
    data = _config().transform_search_request("q", {"country": "US"})
    assert data["_brave_params"]["country"] == "US"


def test_transform_search_request_date_range_maps_to_freshness():
    data = _config().transform_search_request("q", {"start_date": "2022-04-01", "end_date": "2022-07-30"})
    assert data["_brave_params"]["freshness"] == "2022-04-01to2022-07-30"



def test_transform_search_request_no_dates_omits_freshness():
    data = _config().transform_search_request("q", {"max_results": 5})
    assert "freshness" not in data["_brave_params"]


def test_transform_search_request_explicit_freshness_not_clobbered_by_absent_dates():
    """A caller-supplied native `freshness` preset (e.g. 'pw') should survive
    when start_date/end_date aren't given at all."""
    data = _config().transform_search_request("q", {"freshness": "pw"})
    assert data["_brave_params"]["freshness"] == "pw"


def test_both_dates_combined_into_freshness():
    result = _config().transform_search_request(
        query="test query",
        optional_params={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    )
    params = result["_brave_params"]
    assert params["freshness"] == "2024-01-01to2024-12-31"

def test_both_dates_override_native_freshness():
    result = _config().transform_search_request(
        query="test query",
        optional_params={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "freshness": "pw",
        },
    )
    params = result["_brave_params"]
    assert params["freshness"] == "2024-01-01to2024-12-31"

def test_one_sided_date_with_drop_params_warns_and_omits_freshness(monkeypatch, caplog):
    import litellm

    monkeypatch.setattr(litellm, "drop_params", True)
    result =_config().transform_search_request(
        query="test query",
        optional_params={"start_date": "2024-01-01"},
    )
    params = result["_brave_params"]
    assert "freshness" not in params
    assert "one-sided date ranges" in caplog.text

def test_one_sided_date_with_drop_params_preserves_native_freshness(monkeypatch, caplog
):
    import litellm

    monkeypatch.setattr(litellm, "drop_params", True)
    result = _config().transform_search_request(
        query="test query",
        optional_params={
            "end_date": "2024-12-31",
            "freshness": "pm",
        },
    )
    params = result["_brave_params"]
    assert params["freshness"] == "pm"
    assert "one-sided date ranges" in caplog.text

def test_one_sided_date_without_drop_params_raises_even_with_native_fallback(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(UnsupportedParamsError):
        _config().transform_search_request(
            query="test query",
            optional_params={
                "start_date": "2024-01-01",
                "freshness": "pm",
            },
        )


def test_max_results_overrides_native_count():
    result = _config().transform_search_request(
        query="test query",
        optional_params={
            "max_results": 5,
            "count": 15,
        },
    )
    params = result["_brave_params"]
    assert params["count"] == 5

# --- to_yyyy_mm_dd helper ---


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("not-a-date", None),
        ("2024-01-15", "2024-01-15"),
        ("2024/01/15", "2024-01-15"),
        (1705334400, "2024-01-15"),  # unix seconds
        (1705334400000, "2024-01-15"),  # unix milliseconds
    ],
)
def test_to_yyyy_mm_dd(value, expected):
    assert to_yyyy_mm_dd(value) == expected