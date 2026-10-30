from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from e2e_config import PROXY_BASE_URL
from proxy_client import ProxyClient

from client_apps.codex.cli_driver import Gateway, resolve_binary, timeout_seconds
from client_apps.codex.columns import column_lock, load_columns
from client_apps.codex.features import CodexCell
from client_apps.codex.registry import ColumnRegistry
from client_apps.compat_collector import CompatRecorder


@dataclass(frozen=True, slots=True)
class CodexSuiteClient:
    proxy: ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> CodexSuiteClient:
    return CodexSuiteClient(proxy=proxy)


@pytest.fixture(scope="session")
def codex_binary() -> Path:
    binary: Final = resolve_binary()
    if binary is None:
        pytest.fail("No codex binary: run `npm ci` in tests/e2e/client_apps or put codex on PATH")
    return binary


@pytest.fixture(scope="session")
def column_registry(proxy: ProxyClient) -> Iterator[ColumnRegistry]:
    registry: Final = ColumnRegistry(proxy, load_columns())
    yield registry
    registry.release()


@pytest.fixture
def codex_cell(
    compat_result: CompatRecorder,
    column_registry: ColumnRegistry,
    codex_binary: Path,
    scoped_key: str,
    tmp_path: Path,
) -> Iterator[CodexCell]:
    provider: Final = compat_result.address.provider
    errors: Final = column_registry.ensure(provider)
    if errors:
        pytest.fail("; ".join(f"could not register {error.model_name}: {error.error}" for error in errors))
    with column_lock(provider):
        yield CodexCell(
            binary=codex_binary,
            gateway=Gateway(base_url=PROXY_BASE_URL, api_key=scoped_key),
            seconds=timeout_seconds(os.environ),
            sessions_root=tmp_path,
        )
