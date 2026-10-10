from datetime import datetime, timezone

import polars as pl

from litellm.integrations.focus.database import FocusLiteLLMDatabase
from litellm.integrations.focus.destinations import FocusTimeWindow
from litellm.integrations.focus.destinations.factory import FocusDestinationFactory
from litellm.integrations.focus.export_engine import FocusExportEngine
from litellm.integrations.focus.serializers import FocusParquetSerializer
from litellm.integrations.focus.transformer import FocusTransformer


def _vantage_csv_engine() -> FocusExportEngine:
    return FocusExportEngine(
        provider="vantage",
        export_format="csv",
        prefix="exports",
        destination_config={"api_key": "test-key", "integration_token": "test-token"},
    )


def test_legacy_engine_properties_forward_to_public_storage():
    engine = _vantage_csv_engine()

    replacement_destination = FocusDestinationFactory.create(
        provider="vantage",
        prefix="replacement",
        config={"api_key": "test-key", "integration_token": "test-token"},
    )
    engine._destination = replacement_destination
    assert engine._destination is engine.destination is replacement_destination

    replacement_serializer = FocusParquetSerializer()
    engine._serializer = replacement_serializer
    assert engine._serializer is engine.serializer is replacement_serializer

    replacement_transformer = FocusTransformer()
    engine._transformer = replacement_transformer
    assert engine._transformer is engine.transformer is replacement_transformer

    replacement_database = FocusLiteLLMDatabase()
    engine._database = replacement_database
    assert engine._database is engine.database is replacement_database


def test_build_filename_uses_the_window_and_serializer_format():
    engine = _vantage_csv_engine()
    window = FocusTimeWindow(
        start_time=datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, 4, 5, 6, tzinfo=timezone.utc),
        frequency="hourly",
    )

    assert engine.build_filename(window) == "usage_20240102T030405Z_20240102T040506Z.csv"


def test_export_aggregates_compute_values_and_handle_missing_columns():
    frame = pl.DataFrame(
        {
            "spend": [1.5, 2.5, 1.0],
            "team_id": ["team-a", "team-a", "team-b"],
        }
    )

    assert FocusExportEngine.sum_column(frame, "spend") == 5.0
    assert FocusExportEngine.count_unique(frame, "team_id") == 2
    assert FocusExportEngine.sum_column(frame, "missing") == 0.0
    assert FocusExportEngine.count_unique(frame, "missing") == 0
    assert FocusExportEngine.sum_column(pl.DataFrame(), "spend") == 0.0
    assert FocusExportEngine.count_unique(pl.DataFrame(), "team_id") == 0
