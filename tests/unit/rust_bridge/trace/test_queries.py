from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.rust_bridge.trace.queries import TraceSQLResponse


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
        _decode_query_response(TypeAdapter(TraceSQLResponse), '{"meta":[],"data":[],"rows":-1}')
    assert isinstance(error.value.__cause__, ValidationError)
