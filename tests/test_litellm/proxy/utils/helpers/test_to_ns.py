from datetime import datetime, timezone

import pytest

from litellm.proxy.utils import _to_ns


def normalize(value):
    return value


def test_to_ns_happy_path_utc_epoch():
    dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    expected = int(dt.timestamp() * 1e9)
    summary = {
        "input_iso": dt.isoformat(),
        "result": _to_ns(dt),
        "expected": expected,
    }
    assert summary == {
        "input_iso": "2024-01-01T00:00:00+00:00",
        "result": expected,
        "expected": expected,
    }


def test_to_ns_happy_path_microsecond_precision():
    dt = datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
    expected = int(dt.timestamp() * 1e9)
    summary = {
        "input_iso": dt.isoformat(),
        "result": _to_ns(dt),
        "expected": expected,
    }
    assert summary == {
        "input_iso": "2024-06-15T12:30:45.123456+00:00",
        "result": expected,
        "expected": expected,
    }


def test_to_ns_result_is_int():
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result = _to_ns(dt)
    summary = {
        "type": type(result).__name__,
        "is_positive": result > 0,
        "result": result,
    }
    assert summary == {
        "type": "int",
        "is_positive": True,
        "result": int(dt.timestamp() * 1e9),
    }


def test_to_ns_raises_on_invalid_input():
    with pytest.raises(AttributeError):
        _to_ns("2024-01-01T00:00:00")
