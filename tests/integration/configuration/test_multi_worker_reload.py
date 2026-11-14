import json
import os
import re
from pathlib import Path
from typing import Final

import psutil
import pytest

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process

STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


def _worker_pids(owned: OwnedProxy) -> frozenset[int]:
    return frozenset(int(match.group(1)) for match in STARTED_WORKER.finditer(owned.log.read_text()))


def _launched_config(owned: OwnedProxy) -> Path:
    arguments: Final = psutil.Process(owned.process.pid).cmdline()
    return Path(arguments[arguments.index("--config") + 1])


def _config(model: str) -> str:
    return json.dumps(
        {
            "model_list": [{"model_name": model, "litellm_params": {"model": "openai/reload-test"}}],
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
            "router_settings": {"redis_host": "os.environ/REDIS_HOST", "redis_port": "os.environ/REDIS_PORT"},
        }
    )


def _assert_replaced(owned: OwnedProxy, previous: frozenset[int], workers: int) -> frozenset[int]:
    started: Final = eventually(
        lambda: _worker_pids(owned),
        lambda pids: len(pids - previous) == workers,
        seconds=graceful_stop_seconds(),
    )
    assert all(not psutil.pid_exists(pid) for pid in previous), owned.log.read_text()
    assert owned.process.poll() is None, owned.log.read_text()
    assert len(started) == len(previous) + workers, owned.log.read_text()
    return started


def _assert_models(gateway: Gateway) -> None:
    data: Final = gateway.get("/v1/models")["data"]
    assert isinstance(data, list), data
    assert tuple(object_value(model)["id"] for model in data) == ("after-reload",)


@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.timeout(6 * graceful_stop_seconds())
def test_reload_replaces_every_worker_and_shutdown_leaves_none(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    root: Final = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("INTEGRATION_PROXY_ROOT", str(tmp_path))
    config: Final = tmp_path / "config.yaml"
    source: Final = tmp_path / "reload_source.py"
    env: Final = tmp_path / ".env"
    config.write_text(_config("before-reload"))
    source.write_text("VALUE = 1\n")
    env.write_text("RELOAD_TEST_VALUE=before\n")
    with owned_proxy_process(
        gateway,
        tmp_path,
        {
            "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"), os.environ.get("PYTHONPATH", ""))),
            "STORE_MODEL_IN_DB": "False",
        },
        config=config,
        remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "PYTHON_DOTENV_DISABLED"),
        workers=workers,
        database_setup=(),
        extra_arguments=("--reload",),
    ) as owned:
        initial: Final = eventually(
            lambda: _worker_pids(owned), lambda pids: len(pids) == workers, seconds=graceful_stop_seconds()
        )
        _launched_config(owned).write_text(_config("after-reload"))
        after_config: Final = _assert_replaced(owned, initial, workers)
        for _ in range(2 * workers):
            _assert_models(owned.gateway)
        source.write_text("VALUE = 2\n")
        after_source: Final = _assert_replaced(owned, after_config, workers)
        env.write_text("RELOAD_TEST_VALUE=after\n")
        after_env: Final = _assert_replaced(owned, after_source, workers)
        response: Final = owned.gateway.request("GET", "/health/readiness")
        assert response.status_code == 200, response.text
        owned.begin_shutdown()
        owned.process.wait(timeout=graceful_stop_seconds())
        assert all(not psutil.pid_exists(pid) for pid in after_env), owned.log.read_text()
        assert owned.process.returncode == 0, owned.log.read_text()
