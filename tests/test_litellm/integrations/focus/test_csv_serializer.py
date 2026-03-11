"""Tests for FocusCsvSerializer."""

from __future__ import annotations

import polars as pl

from litellm.integrations.focus.serializers.csv import FocusCsvSerializer


def test_should_serialize_dataframe_to_csv():
    frame = pl.DataFrame({"BilledCost": [1.5, 2.0], "ServiceName": ["openai", "anthropic"]})
    serializer = FocusCsvSerializer()
    result = serializer.serialize(frame)

    assert isinstance(result, bytes)
    lines = result.decode("utf-8").strip().split("\n")
    assert lines[0] == "BilledCost,ServiceName"
    assert len(lines) == 3  # header + 2 data rows


def test_should_return_header_only_for_empty_frame():
    frame = pl.DataFrame(
        schema={"BilledCost": pl.Float64, "ServiceName": pl.Utf8}
    )
    serializer = FocusCsvSerializer()
    result = serializer.serialize(frame)

    lines = result.decode("utf-8").strip().split("\n")
    assert lines[0] == "BilledCost,ServiceName"
    assert len(lines) == 1  # header only


def test_extension_should_be_csv():
    assert FocusCsvSerializer.extension == "csv"
