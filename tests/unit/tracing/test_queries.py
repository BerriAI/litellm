from typing import Final

import pytest
from pydantic import ValidationError

from litellm.tracing.queries import TRACE_AGENTS


@pytest.mark.parametrize("count", (0, "9007199254740993", 2**64 - 1))
def test_agent_rows_normalize_counts_without_precision_loss(count: int | str) -> None:
    result: Final = TRACE_AGENTS.response.validate_python(
        {"data": [{"agent_name": "agent", "runs": count, "failed_runs": 0, "last_seen_ms": 1, "frameworks": ["otel"]}]}
    )
    row: Final = result.data[0]
    assert row.runs == int(count)
    assert row.frameworks == ("otel",)
    with pytest.raises(ValidationError):
        row.agent_name = "changed"


@pytest.mark.parametrize("count", (-1, "18446744073709551616", "1.5"))
def test_agent_rows_reject_invalid_counts(count: int | str) -> None:
    with pytest.raises(ValidationError):
        TRACE_AGENTS.response.validate_python(
            {"data": [{"agent_name": "agent", "runs": count, "failed_runs": 0, "last_seen_ms": 1}]}
        )


def test_dictionary_validation_keeps_required_nullable_and_optional_fields_distinct() -> None:
    from pydantic import TypeAdapter

    from litellm.tracing.generated.types import SpanDetail, SpanErrorPage

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
