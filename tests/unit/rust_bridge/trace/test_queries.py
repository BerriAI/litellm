from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue, ValidationError

from litellm.rust_bridge.trace.generated.models import LensContentParams
from litellm.rust_bridge.trace.queries import LENS_CONTENT, LENS_EVIDENCE, TraceSQLResponse


@pytest.mark.parametrize("offset", (-1, 2**32))
def test_named_query_rejects_offsets_outside_the_native_integer_range(offset: int) -> None:
    with pytest.raises(ValidationError) as error:
        LENS_CONTENT.parameters.model_validate(
            {
                "all_teams": 0,
                "team": "team",
                "key_hash": "",
                "source": "traces",
                "id": "trace",
                "record_team": "team",
                "trace_ref": "ref",
                "cursor": "",
                "offset": offset,
            }
        )
    assert error.value.error_count() == 1


def test_named_query_rejects_parameters_for_a_different_query() -> None:
    detail: Final = LensContentParams(
        all_teams=0,
        team="team",
        key_hash="",
        source="traces",
        id="trace",
        record_team="team",
        trace_ref="ref",
        cursor="",
        offset=0,
    )
    with pytest.raises(ValidationError) as error:
        LENS_EVIDENCE.parameters.model_validate(detail)
    assert error.value.error_count() == 1


def test_named_query_rejects_rows_missing_required_result_fields() -> None:
    with pytest.raises(ValidationError) as error:
        LENS_CONTENT.response.validate_json('{"data":[{"span_id":"span","name":"name"}]}')
    assert error.value.error_count() == 4


def test_sql_envelope_preserves_nested_data_large_integer_strings_and_extra_fields() -> None:
    envelope: Final[Mapping[str, JsonValue]] = {
        "meta": [{"name": "count", "type": "UInt64", "comment": "label"}],
        "data": [{"count": "9007199254740993", "nested": [True, None, {"value": 2}]}],
        "rows": "1",
        "statistics": {"elapsed": 0.01, "rows_read": "1", "bytes_read": "8", "extra_stat": 4},
        "totals": {"count": "9007199254740993"},
    }
    result: Final = TraceSQLResponse.model_validate(envelope)
    assert result.model_dump(mode="json", exclude_unset=True) == envelope


@pytest.mark.parametrize("count", (0, "9007199254740993", 2**64 - 1))
def test_clickhouse_rows_normalize_numbers_and_preserve_tuples(count: int | str) -> None:
    from litellm.rust_bridge.trace.queries import LENS_SAMPLE

    result: Final = LENS_SAMPLE.response.validate_json(
        '{"data":[{"source":"traces","trace_id":"trace","team_id":"team","name":"run",'
        '"start_time":"time","span_count":'
        + (f'"{count}"' if isinstance(count, str) else str(count))
        + ',"root_seen":"1","eligible":"2","selected":2.0,"attributes":[["key","value"]]}]}'
    )
    row: Final = result.data[0]
    assert row.span_count == int(count)
    assert row.root_seen == 1
    assert row.selected == 2
    assert row.attributes == (("key", "value"),)
    assert row.service == ""
    assert row.trace_ref == ""
    assert row.selection_key == ""
    with pytest.raises(ValidationError):
        row.name = "changed"


def test_response_defaults_remain_normalized_when_omitted() -> None:
    from litellm.rust_bridge.trace.generated.models import ActivityAvailability
    from litellm.rust_bridge.trace.queries import LENS_SAMPLE

    row: Final = LENS_SAMPLE.response.validate_json(
        '{"data":[{"source":"requests","trace_id":"trace","team_id":"team","name":"run",'
        '"start_time":"time","span_count":"1","root_seen":1,"eligible":"2"}]}'
    ).data[0]
    assert row.attributes == ()
    assert row.selected == 0
    assert ActivityAvailability().traces is False
    assert ActivityAvailability().requests is False


@pytest.mark.parametrize("count", (-1, "18446744073709551616", "1.5"))
def test_clickhouse_count_rejects_invalid_quoted_and_unquoted_numbers(count: int | str) -> None:
    from litellm.rust_bridge.trace.queries import LENS_EVIDENCE

    with pytest.raises(ValidationError):
        LENS_EVIDENCE.response.validate_python({"data": [{"count": count}]})


def test_dictionary_validation_keeps_required_nullable_and_optional_fields_distinct() -> None:
    from pydantic import TypeAdapter

    from litellm.rust_bridge.trace.generated.types import SpanDetail, SpanErrorPage

    result: Final = TypeAdapter(SpanDetail).validate_python(
        {
            "span_id": "span",
            "input": "",
            "output": "",
            "attributes": {"key": "value"},
            "input_ui": {"kind": "messages", "messages": [{"role": "user", "content": "hello"}]},
            "output_ui": {"kind": "text", "text": "answer"},
        }
    )
    assert result["input_ui"] == {"kind": "messages", "messages": ({"role": "user", "content": "hello"},)}
    assert result["attributes"] == {"key": "value"}
    assert (
        TypeAdapter(SpanErrorPage).validate_python(
            {
                "span_id": "span",
                "message": "error",
                "total_chars": 5,
                "next_cursor": None,
            }
        )["next_cursor"]
        is None
    )
    with pytest.raises(ValidationError):
        TypeAdapter(SpanErrorPage).validate_python({"span_id": "span", "message": "error", "total_chars": 5})


def test_invalid_native_response_preserves_validation_error_as_cause() -> None:
    from litellm.rust_bridge.trace.storage import _decode_query_response

    with pytest.raises(RuntimeError, match="Native trace query returned an invalid response") as error:
        _decode_query_response(LENS_EVIDENCE.response, '{"data":[{"count":-1}]}')
    assert isinstance(error.value.__cause__, ValidationError)


@pytest.mark.parametrize("flag", (0, 1, "0", "1"))
def test_clickhouse_availability_normalizes_numeric_boolean_flags(flag: int | str) -> None:
    from litellm.rust_bridge.trace.generated.models import ActivityAvailability

    result: Final = ActivityAvailability.model_validate({"traces": flag, "requests": flag})
    assert result.traces is (str(flag) == "1")
    assert result.requests is result.traces


def test_response_flags_reject_values_outside_the_boolean_range() -> None:
    from litellm.rust_bridge.trace.generated.models import ActivityAvailability

    with pytest.raises(ValidationError):
        ActivityAvailability.model_validate({"traces": 2})
    with pytest.raises(ValidationError):
        LENS_CONTENT.response.validate_python(
            {
                "data": [
                    {
                        "span_id": "s",
                        "parent_span_id": "",
                        "name": "n",
                        "kind": "agent",
                        "content": "",
                        "truncated": "2",
                    }
                ]
            }
        )
