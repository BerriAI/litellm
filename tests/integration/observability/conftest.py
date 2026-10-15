from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.otlp_sink import (
    AuditConfigWriter,
    SpanSinks,
    audit_configuration_writer,
    langfuse_callback_variables,
    owned_sinks,
)
from integration._support.prometheus_series import CapRig, series_cap_rig, spend_rows
from pydantic import JsonValue


@pytest.fixture(scope="module")
def audit_sinks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SpanSinks]:
    directory: Final = tmp_path_factory.mktemp("otel-audit-sinks")
    with owned_sinks(directory) as sinks:
        yield sinks


@pytest.fixture(scope="module")
def otel_audit_config(audit_sinks: SpanSinks) -> AuditConfigWriter:
    return audit_configuration_writer(audit_sinks)


@pytest.fixture(scope="module")
def langfuse_vars(audit_sinks: SpanSinks) -> dict[str, JsonValue]:
    return langfuse_callback_variables(audit_sinks)


@pytest.fixture(scope="session")
def capped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapRig]:
    """A two-worker proxy capped at three series per metric, with three keys already holding a series each."""
    with series_cap_rig(
        tmp_path_factory.mktemp("series-cap"),
        {"prometheus_metrics_max_series_per_metric": 3},
        workers=2,
        warm_keys=3,
    ) as rig:
        eventually(
            lambda: tuple(len(spend_rows(key.alias)) for key in rig.warm),
            lambda counts: all(count == 1 for count in counts),
            seconds=70,
        )
        yield rig
