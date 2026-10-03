from typing import Final

import pytest
from pydantic import JsonValue, ValidationError

from litellm.rust_bridge.trace_queries import SPAN_DETAIL, SPAN_ERROR, SpanDetailParams
from litellm.rust_bridge.trace_query_responses import TraceSQLResponse


@pytest.mark.parametrize("offset", (-1, 2**64))
def test_named_query_rejects_offsets_outside_the_native_integer_range(offset: int) -> None:
    with pytest.raises(ValidationError) as error:
        SPAN_ERROR.parameters.model_validate(
            {
                "all_teams": 0,
                "user_id": "",
                "team_ids": ["team"],
                "trace_id": "trace",
                "trace_ref": "ref",
                "span_id": "span",
                "error_offset": offset,
                "error_version": "",
            }
        )
    assert error.value.error_count() == 1


def test_named_query_rejects_parameters_for_a_different_query() -> None:
    detail: Final = SpanDetailParams(
        all_teams=0,
        user_id="",
        team_ids=("team",),
        trace_id="trace",
        trace_ref="ref",
        span_id="span",
    )
    with pytest.raises(ValidationError) as error:
        SPAN_ERROR.parameters.model_validate(detail)
    assert error.value.error_count() == 1


def test_named_query_rejects_rows_missing_required_result_fields() -> None:
    with pytest.raises(ValidationError) as error:
        SPAN_DETAIL.response.validate_json('{"data":[{"span_id":"span","input":"input","output":"output"}]}')
    assert error.value.error_count() == 1


def test_sql_envelope_preserves_nested_data_large_integer_strings_and_extra_fields() -> None:
    envelope: Final[dict[str, JsonValue]] = {
        "meta": [{"name": "count", "type": "UInt64", "comment": "label"}],
        "data": [{"count": "9007199254740993", "nested": [True, None, {"value": 2}]}],
        "rows": "1",
        "statistics": {"elapsed": 0.01, "rows_read": "1", "bytes_read": "8", "extra_stat": 4},
        "totals": {"count": "9007199254740993"},
    }
    result: Final = TraceSQLResponse.model_validate(envelope)
    assert result.model_dump(mode="json", exclude_unset=True) == envelope
