from __future__ import annotations

import asyncio
import errno
import importlib
import socket
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Final

import httpx
import pytest
import yaml
from pydantic import TypeAdapter

from tests.integration._support.client import Gateway
from tests.integration._support.process import worker_configuration

TESTS_DIR: Final = Path(__file__).resolve().parents[2]
UNRELATED_CRASH: Final = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'litellm'\n"


def test_worker_configuration_routes_stock_edges_to_its_own_servers_and_preserves_other_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    monkeypatch.setenv("INTEGRATION_PROVIDER_PORT", "8203")
    monkeypatch.setenv("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8202")
    source: Final = tmp_path / "source.yaml"
    original: Final = (
        'api_base: http://127.0.0.1:8191/v1\ntoken_url: "http://127.0.0.1:8190/_oauth/token"\n'
        "peer: http://127.0.0.1:81910/v1\ncustom: http://127.0.0.1:9000/v1\n"
    )
    source.write_text(original)
    configured: Final = worker_configuration(source, tmp_path / "worker")
    assert yaml.safe_load(configured.read_text()) == {
        "api_base": "http://127.0.0.1:8203/v1",
        "token_url": "http://127.0.0.1:8202/_oauth/token",
        "peer": "http://127.0.0.1:81910/v1",
        "custom": "http://127.0.0.1:9000/v1",
        "general_settings": {"database_connection_pool_limit": 2},
    }
    assert source.read_text() == original


@pytest.mark.parametrize("limit", (1, 5, 0, None, "", "bad"))
def test_worker_configuration_preserves_explicit_pool_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int | str | None
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    source: Final = tmp_path / "source.yaml"
    original: Final = {
        "general_settings": {"database_connection_pool_limit": limit, "database_connection_timeout": 3},
        "model_list": [{"model_name": "synthetic", "litellm_params": {"model": "openai/synthetic"}}],
    }
    source.write_text(yaml.safe_dump(original))
    configured: Final = worker_configuration(source, tmp_path / "worker")
    assert yaml.safe_load(configured.read_text()) == original
    assert yaml.safe_load(source.read_text()) == original


def test_worker_configuration_budgets_implicit_pools_without_changing_connection_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INTEGRATION_WORKER_ISOLATION", "1")
    source: Final = tmp_path / "source.yaml"
    original: Final = {"general_settings": {"database_extra_connection_params": {"connection_limit": "1"}}}
    source.write_text(yaml.safe_dump(original))
    configured: Final = worker_configuration(source, tmp_path / "worker")
    decoded: Final = TypeAdapter(dict[str, object]).validate_python(yaml.safe_load(configured.read_text()))
    assert decoded == {
        "general_settings": {
            "database_connection_pool_limit": 2,
            "database_extra_connection_params": {"connection_limit": "1"},
        }
    }
    assert yaml.safe_load(source.read_text()) == original


def test_serial_proxy_keeps_the_requested_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("INTEGRATION_WORKER_ISOLATION", raising=False)
    source: Final = tmp_path / "source.yaml"
    source.write_text("api_base: http://127.0.0.1:8191/v1\n")
    assert worker_configuration(source, tmp_path / "unused") == source
    assert source.read_text() == "api_base: http://127.0.0.1:8191/v1\n"


@pytest.fixture
def process_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    return importlib.import_module("integration._support.process")


async def _refused_bind(port: int) -> OSError | None:
    try:
        await asyncio.get_running_loop().create_server(asyncio.Protocol, "127.0.0.1", port)
    except OSError as refused:
        return refused
    return None


@pytest.fixture
def bind_error_line() -> Iterator[str]:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        refused: Final = asyncio.run(_refused_bind(held.getsockname()[1]))
        assert refused is not None and refused.errno == errno.EADDRINUSE
        yield f"ERROR:    {refused}\n"


def _written_log(directory: Path, text: str) -> Path:
    log: Final = directory / "owned-proxy.log"
    log.write_text(text)
    return log


def test_lost_port_race_matches_the_bind_error_the_server_logs(
    process_module: ModuleType, bind_error_line: str, tmp_path: Path
) -> None:
    assert process_module._lost_port_race(1, _written_log(tmp_path, bind_error_line))


def test_lost_port_race_ignores_an_exit_for_another_reason(process_module: ModuleType, tmp_path: Path) -> None:
    assert not process_module._lost_port_race(1, _written_log(tmp_path, UNRELATED_CRASH))


def test_lost_port_race_needs_the_process_to_have_exited(
    process_module: ModuleType, bind_error_line: str, tmp_path: Path
) -> None:
    assert not process_module._lost_port_race(None, _written_log(tmp_path, bind_error_line))


@pytest.mark.parametrize(
    ("writer", "explicit_reader", "expected_reader"),
    (
        ("postgresql://writer@127.0.0.1/shared", None, "postgresql://reader@127.0.0.1/shared"),
        ("postgresql://writer@127.0.0.1/scratch", None, None),
        (
            "postgresql://writer@127.0.0.1/scratch",
            "postgresql://reader@127.0.0.1/scratch",
            "postgresql://reader@127.0.0.1/scratch",
        ),
    ),
)
def test_owned_writer_keeps_only_a_reader_paired_with_its_database(
    process_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    explicit_reader: str | None,
    expected_reader: str | None,
) -> None:
    monkeypatch.setenv("INTEGRATION_PROXY_READ_REPLICA_URL", "postgresql://reader@127.0.0.1/shared")
    overrides: Final = {
        "DATABASE_URL": writer,
        **({"DATABASE_URL_READ_REPLICA": explicit_reader} if explicit_reader else {}),
    }
    with httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client:
        environment: Final = process_module._proxy_environment(
            Gateway(client, "owned-test-key", "http://127.0.0.1:1"), overrides, ()
        )
    assert environment["DATABASE_URL"] == writer
    assert environment.get("DATABASE_URL_READ_REPLICA") == expected_reader


@pytest.mark.parametrize("explicit", (False, True))
@pytest.mark.parametrize("remove", (False, True))
def test_owned_runtime_configuration_preserves_custom_worker_startup_hooks(
    process_module: ModuleType, monkeypatch: pytest.MonkeyPatch, explicit: bool, remove: bool
) -> None:
    monkeypatch.setenv("LITELLM_WORKER_STARTUP_HOOKS", "ambient:configure")
    with httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client:
        environment: Final = process_module._proxy_environment(
            Gateway(client, "owned-test-key", "http://127.0.0.1:1"),
            {"LITELLM_WORKER_STARTUP_HOOKS": "custom:configure"} if explicit else {},
            ("LITELLM_WORKER_STARTUP_HOOKS",) if remove else (),
        )
    expected: Final = "custom:configure" if explicit else ("" if remove else "ambient:configure")
    assert environment["LITELLM_WORKER_STARTUP_HOOKS"] == ",".join(
        hook
        for hook in (
            "integration._support.runtime:configure_executor",
            "integration._support.runtime:configure_readiness",
            expected,
        )
        if hook
    )
