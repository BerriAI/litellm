import pytest


@pytest.mark.parametrize(
    "finish_reason, expected",
    [
        ("stop_sequence", "stop"),
        ("content_filtered", "content_filter"),
        ("ERROR_TOXIC", "content_filter"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
    ],
)
def test_map_finish_reason_maps_provider_values(finish_reason, expected):
    from litellm.litellm_core_utils.core_helpers import map_finish_reason

    assert map_finish_reason(finish_reason) == expected

