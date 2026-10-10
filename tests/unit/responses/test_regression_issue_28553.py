import pytest

from litellm.proxy.common_request_processing import _stream_usage_tracking_updates


@pytest.mark.parametrize(
    "route_type,expect_injected",
    [
        ("acompletion", True),
        ("atext_completion", False),
        ("aresponses", False),
        ("aget_responses", False),
        ("arealtime", False),
    ],
)
def test_stream_usage_only_injected_for_whitelisted_route_types(route_type, expect_injected):
    updates = _stream_usage_tracking_updates(
        data={"stream": True, "model": "gpt-4o"},
        general_settings={},
        route_type=route_type,
        supports_stream_options=lambda: True,
    )
    assert ("stream_options" in updates) is expect_injected
