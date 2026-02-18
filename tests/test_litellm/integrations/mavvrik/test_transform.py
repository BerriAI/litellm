"""Unit tests for the Mavvrik transform layer (CSV output)."""

import io
import os
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.transform import MavvrikTransformer


def _make_df(**kwargs) -> pl.DataFrame:
    """Helper: build a minimal spend DataFrame, overriding defaults with kwargs."""
    defaults = {
        "date": ["2025-01-19"],
        "user_id": ["user-1"],
        "api_key": ["sk-abc"],
        "model": ["gpt-4o"],
        "model_group": ["gpt-4o-group"],
        "custom_llm_provider": ["openai"],
        "prompt_tokens": [100],
        "completion_tokens": [50],
        "spend": [1.5],
        "api_requests": [5],
        "successful_requests": [4],
        "failed_requests": [1],
        "team_id": ["team-1"],
        "api_key_alias": ["prod-key"],
        "team_alias": ["Alpha"],
        "user_email": ["alice@example.com"],
    }
    defaults.update(kwargs)
    return pl.DataFrame(defaults)


class TestMavvrikTransformerToCsv:
    def test_empty_dataframe_returns_empty_string(self):
        transformer = MavvrikTransformer()
        result = transformer.to_csv(pl.DataFrame())
        assert result == ""

    def test_filters_zero_successful_requests(self):
        transformer = MavvrikTransformer()
        df = _make_df(successful_requests=[0])
        result = transformer.to_csv(df)
        assert result == ""

    def test_keeps_nonzero_successful_requests(self):
        transformer = MavvrikTransformer()
        df = _make_df(successful_requests=[3])
        result = transformer.to_csv(df)
        assert result != ""

    def test_all_zero_requests_returns_empty(self):
        transformer = MavvrikTransformer()
        df = pl.DataFrame(
            {
                "date": ["2025-01-19", "2025-01-20"],
                "successful_requests": [0, 0],
                "spend": [1.0, 2.0],
                "model": ["gpt-4", "gpt-4"],
            }
        )
        result = transformer.to_csv(df)
        assert result == ""

    def test_mixed_requests_filters_correctly(self):
        transformer = MavvrikTransformer()
        df = pl.DataFrame(
            {
                "date": ["2025-01-19", "2025-01-20", "2025-01-21"],
                "successful_requests": [5, 0, 3],
                "spend": [1.0, 2.0, 3.0],
                "model": ["gpt-4", "gpt-4", "gpt-4"],
            }
        )
        result = transformer.to_csv(df)
        lines = [l for l in result.strip().split("\n") if l]
        # 1 header + 2 data rows
        assert len(lines) == 3

    def test_output_has_header_row(self):
        transformer = MavvrikTransformer()
        df = _make_df()
        result = transformer.to_csv(df)
        header = result.split("\n")[0]
        assert "model" in header
        assert "spend" in header

    def test_all_db_columns_present_in_header(self):
        transformer = MavvrikTransformer()
        df = _make_df()
        header = transformer.to_csv(df).split("\n")[0]
        for col in [
            "date", "user_id", "api_key", "model", "model_group",
            "custom_llm_provider", "prompt_tokens", "completion_tokens",
            "spend", "api_requests", "successful_requests", "failed_requests",
            "team_id", "api_key_alias", "team_alias", "user_email",
        ]:
            assert col in header, f"Expected column '{col}' in CSV header"

    def test_spend_value_in_output(self):
        transformer = MavvrikTransformer()
        df = _make_df(spend=[42.5])
        result = transformer.to_csv(df)
        assert "42.5" in result

    def test_model_value_in_output(self):
        transformer = MavvrikTransformer()
        df = _make_df(model=["claude-3-5-sonnet"])
        result = transformer.to_csv(df)
        assert "claude-3-5-sonnet" in result

    def test_no_successful_requests_column_no_filter_applied(self):
        """If column is absent, all rows are included (no filter)."""
        transformer = MavvrikTransformer()
        df = pl.DataFrame({"date": ["2025-01-19"], "spend": [5.0], "model": ["gpt-4"]})
        result = transformer.to_csv(df)
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 2  # header + 1 data row

    def test_multiple_rows_all_in_output(self):
        transformer = MavvrikTransformer()
        df = pl.DataFrame(
            {
                "date": ["2025-01-19", "2025-01-20"],
                "successful_requests": [2, 3],
                "spend": [1.0, 2.0],
                "model": ["gpt-4", "claude-3"],
            }
        )
        result = transformer.to_csv(df)
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 3  # header + 2 data rows

    def test_output_is_valid_csv(self):
        transformer = MavvrikTransformer()
        df = _make_df()
        result = transformer.to_csv(df)
        # Polars can re-read its own CSV output
        reloaded = pl.read_csv(io.StringIO(result))
        assert len(reloaded) == 1
        assert "model" in reloaded.columns

    def test_return_type_is_str(self):
        transformer = MavvrikTransformer()
        df = _make_df()
        result = transformer.to_csv(df)
        assert isinstance(result, str)
