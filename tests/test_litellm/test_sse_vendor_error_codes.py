import pytest

from litellm.proxy.common_request_processing import _parse_event_data_for_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, 502),
        (99, 502),
        (100, 100),
        (400, 400),
        (599, 599),
        (600, 502),
        (1302, 502),
        ("4001", 502),
    ],
)
async def test_parse_event_maps_numeric_vendor_error_codes(code, expected):
    event = f'data: {{"error": {{"code": {code!r}, "message": "provider error"}}}}'
    if isinstance(code, str):
        event = event.replace("'", '"')

    assert await _parse_event_data_for_error(event) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        'data: {"error": {"code": "not-a-number"}}',
        'data: {"error": {"message": "missing code"}}',
        "data: [DONE]",
    ],
)
async def test_parse_event_ignores_missing_or_non_numeric_error_codes(event):
    assert await _parse_event_data_for_error(event) is None
