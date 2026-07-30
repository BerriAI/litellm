"""MCP suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness handling, and the
`e2e`/`covers` markers live in the parent tests/e2e/conftest.py. McpClient holds
the shared ProxyClient, so the `resources` fixture tears down whatever this suite
creates (keys via the ProxyClient, MCP servers via the deferred cleanups).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast

import pytest

from mcp_client import McpClient, build_client
from proxy_client import ProxyClient


class DdLogsReader(Protocol):
    def poll_events_for_marker(self, marker: str) -> list[object]: ...


class _DdLogsReaderBuilder(Protocol):
    def __call__(self) -> DdLogsReader: ...


def _build_dd_logs_reader() -> DdLogsReader:
    # Load logging/datadog_reader.py by path so basedpyright does not require a
    # package layout. Register the module in sys.modules before exec so
    # dataclasses inside it can resolve cls.__module__ (otherwise Python 3.12
    # raises AttributeError: 'NoneType' object has no attribute '__dict__').
    path = Path(__file__).resolve().parent.parent / "logging" / "datadog_reader.py"
    name = "e2e_logging_datadog_reader"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    builder = cast(_DdLogsReaderBuilder, getattr(module, "build_dd_logs_reader"))
    return builder()


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> McpClient:
    return build_client(proxy)


@pytest.fixture(scope="session")
def dd_logs() -> DdLogsReader:
    return _build_dd_logs_reader()
