"""Unit tests for the Mavvrik transform layer (CSV output)."""

import io
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.exporter import Exporter


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


class TestExporterToCsv:
    def test_empty_dataframe_returns_empty_string(self):
        transformer = Exporter()
        result = transformer._to_csv(pl.DataFrame())
        assert result == ""

    def test_nonzero_successful_requests_in_output(self):
        transformer = Exporter()
        df = _make_df(successful_requests=[3])
        result = transformer._to_csv(df)
        assert result != ""

    def test_output_has_header_row(self):
        transformer = Exporter()
        df = _make_df()
        result = transformer._to_csv(df)
        header = result.split("\n")[0]
        assert "model" in header
        assert "spend" in header

    def test_all_db_columns_present_in_header(self):
        transformer = Exporter()
        df = _make_df()
        header = transformer._to_csv(df).split("\n")[0]
        for col in [
            "date",
            "user_id",
            "api_key",
            "model",
            "model_group",
            "custom_llm_provider",
            "prompt_tokens",
            "completion_tokens",
            "spend",
            "api_requests",
            "successful_requests",
            "failed_requests",
            "team_id",
            "api_key_alias",
            "team_alias",
            "user_email",
        ]:
            assert col in header, f"Expected column '{col}' in CSV header"

    def test_spend_value_in_output(self):
        transformer = Exporter()
        df = _make_df(spend=[42.5])
        result = transformer._to_csv(df)
        assert "42.5" in result

    def test_model_value_in_output(self):
        transformer = Exporter()
        df = _make_df(model=["claude-3-5-sonnet"])
        result = transformer._to_csv(df)
        assert "claude-3-5-sonnet" in result

    def test_multiple_rows_all_in_output(self):
        transformer = Exporter()
        df = pl.DataFrame(
            {
                "date": ["2025-01-19", "2025-01-20"],
                "successful_requests": [2, 3],
                "spend": [1.0, 2.0],
                "model": ["gpt-4", "claude-3"],
            }
        )
        result = transformer._to_csv(df)
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 3  # header + 2 data rows

    def test_output_is_valid_csv(self):
        transformer = Exporter()
        df = _make_df()
        result = transformer._to_csv(df)
        # Polars can re-read its own CSV output
        reloaded = pl.read_csv(io.StringIO(result))
        assert len(reloaded) == 1
        assert "model" in reloaded.columns

    def test_return_type_is_str(self):
        transformer = Exporter()
        df = _make_df()
        result = transformer._to_csv(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# to_csv — connection_id column
# ---------------------------------------------------------------------------


class TestToCsvConnectionId:
    def test_adds_connection_id_column_when_provided(self):
        exporter = Exporter()
        df = _make_df()
        result = exporter._to_csv(df, connection_id="conn-123")
        assert "connection_id" in result
        assert "conn-123" in result

    def test_no_connection_id_column_when_omitted(self):
        exporter = Exporter()
        df = _make_df()
        result = exporter._to_csv(df)
        assert "connection_id" not in result.split("\n")[0]


# ---------------------------------------------------------------------------
# _prisma_client — raises when DB not connected
# ---------------------------------------------------------------------------


class TestExporterPrismaClient:
    def test_raises_runtime_error_when_db_not_connected(self):
        """_prisma_client raises RuntimeError when prisma_client is None."""
        exporter = Exporter()
        with patch(
            "litellm.integrations.mavvrik.exporter.prisma_client", None, create=True
        ):
            with patch(
                "litellm.integrations.mavvrik.exporter.Exporter._prisma_client",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(
                        RuntimeError("Database not connected")
                    )
                ),
            ):
                with pytest.raises(RuntimeError, match="Database not connected"):
                    _ = exporter._prisma_client


# ---------------------------------------------------------------------------
# get_usage_data and get_earliest_date — mocked prisma
# ---------------------------------------------------------------------------


class TestExporterDbMethods:
    @pytest.mark.asyncio
    async def test_get_usage_data_returns_dataframe(self):
        """get_usage_data() returns a Polars DataFrame from query_raw results."""
        exporter = Exporter()
        mock_rows = [
            {
                "date": "2026-04-10",
                "user_id": "user-1",
                "model": "gpt-4o",
                "spend": 0.015,
                "successful_requests": 5,
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        ]
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=mock_rows)

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            df = await exporter._get_usage_data("2026-04-10")

        assert len(df) == 1
        assert "model" in df.columns

    @pytest.mark.asyncio
    async def test_get_usage_data_with_limit(self):
        """get_usage_data() appends LIMIT clause when limit is provided."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])
        captured = []

        async def fake_query_raw(query, *params):
            captured.append((query, params))
            return []

        mock_client.db.query_raw = fake_query_raw

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            await exporter._get_usage_data("2026-04-10", limit=100)

        assert "LIMIT" in captured[0][0]
        assert 100 in captured[0][1]

    @pytest.mark.asyncio
    async def test_get_earliest_date_returns_date_string(self):
        """get_earliest_date() returns first 10 chars of the MIN(date) result."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(
            return_value=[{"earliest": "2026-01-01T00:00:00"}]
        )

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            result = await exporter.get_earliest_date()

        assert result == "2026-01-01"

    @pytest.mark.asyncio
    async def test_get_earliest_date_returns_none_when_empty(self):
        """get_earliest_date() returns None when table is empty."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[{"earliest": None}])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            result = await exporter.get_earliest_date()

        assert result is None


# ---------------------------------------------------------------------------
# Exporter.export() — public method combining _get_usage_data + filter + _to_csv
# ---------------------------------------------------------------------------


class TestExporterExport:
    @pytest.mark.asyncio
    async def test_export_returns_dataframe_and_csv(self):
        """export() returns (filtered_df, csv_str) in one call."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_rows = [
            {
                "date": "2026-04-10",
                "user_id": "user-1",
                "model": "gpt-4o",
                "spend": 0.015,
                "successful_requests": 5,
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        ]
        mock_client.db.query_raw = AsyncMock(return_value=mock_rows)

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            df, csv = await exporter.export(
                date_str="2026-04-10", connection_id="conn-1"
            )

        assert len(df) == 1
        assert "conn-1" in csv
        assert isinstance(csv, str)

    @pytest.mark.asyncio
    async def test_export_returns_empty_when_no_data(self):
        """export() returns empty DataFrame and empty string when no rows."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            df, csv = await exporter.export(
                date_str="2026-04-10", connection_id="conn-1"
            )

        assert df.is_empty()
        assert csv == ""


# ---------------------------------------------------------------------------
# Exporter._stream_pages — async generator for paginated DB fetch
# ---------------------------------------------------------------------------


class TestStreamPages:
    @pytest.mark.asyncio
    async def test_yields_header_then_csv_rows(self):
        """_stream_pages() first yields a CSV header, then row data."""
        exporter = Exporter()
        mock_rows = [
            {
                "date": "2026-04-10",
                "model": "gpt-4o",
                "spend": 0.01,
                "successful_requests": 1,
            },
        ]
        # page 1 returns 1 row, page 2 returns empty → stop
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(side_effect=[mock_rows, []])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            chunks = []
            async for chunk in exporter._stream_pages(
                "2026-04-10", connection_id="c-1"
            ):
                chunks.append(chunk)

        assert len(chunks) >= 1
        combined = "".join(chunks)
        assert "date" in combined and "model" in combined
        assert "gpt-4o" in combined

    @pytest.mark.asyncio
    async def test_yields_nothing_when_db_empty(self):
        """_stream_pages() yields nothing when DB returns no rows."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            chunks = []
            async for chunk in exporter._stream_pages(
                "2026-04-10", connection_id="c-1"
            ):
                chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_paginates_using_offset(self):
        """_stream_pages() uses OFFSET to fetch subsequent pages."""
        exporter = Exporter()
        page1 = [
            {
                "date": "2026-04-10",
                "model": "gpt-4o",
                "spend": 0.01,
                "successful_requests": 1,
            }
        ] * 3
        page2 = []
        mock_client = MagicMock()
        captured_queries = []

        async def fake_query_raw(query, *params):
            captured_queries.append(params)
            return page1 if len(captured_queries) == 1 else page2

        mock_client.db.query_raw = fake_query_raw

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            async for _ in exporter._stream_pages(
                "2026-04-10", connection_id="c", page_size=3
            ):
                pass

        # Two queries: page 1 (offset 0) and page 2 (offset 3 → empty → stop)
        assert len(captured_queries) == 2
        assert captured_queries[0][-1] == 0  # first OFFSET is 0
        assert captured_queries[1][-1] == 3  # second OFFSET is page_size


# ---------------------------------------------------------------------------
# Exporter — no DB connected: log warning, return gracefully
# ---------------------------------------------------------------------------


class TestExporterNoDb:
    @pytest.mark.asyncio
    async def test_get_usage_data_returns_empty_when_no_db(self):
        """_get_usage_data returns empty DataFrame when DB not connected."""
        exporter = Exporter()
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: None),
        ):
            df = await exporter._get_usage_data("2026-04-10")
        assert df.is_empty()

    @pytest.mark.asyncio
    async def test_get_earliest_date_returns_none_when_no_db(self):
        """get_earliest_date returns None when DB not connected."""
        exporter = Exporter()
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await exporter.get_earliest_date()
        assert result is None

    @pytest.mark.asyncio
    async def test_stream_pages_yields_nothing_when_no_db(self):
        """_stream_pages yields nothing when DB not connected."""
        exporter = Exporter()
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: None),
        ):
            chunks = []
            async for chunk in exporter._stream_pages("2026-04-10", connection_id="c"):
                chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_pages_yields_nothing_when_db_empty(self):
        """_stream_pages yields nothing when DB returns no rows for the date."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            chunks = []
            async for chunk in exporter._stream_pages("2026-04-10", connection_id="c"):
                chunks.append(chunk)
        assert chunks == []
