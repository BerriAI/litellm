from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Callable, Iterator
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils import get_model_cost_map as loader
from litellm.litellm_core_utils import model_cost_api
from litellm.litellm_core_utils.get_model_cost_map import (
    _MODEL_COST_LAYERS,
    GetModelCostMap,
    get_model_cost_map_source_info,
    load_model_cost_map_at_import,
)
from litellm.litellm_core_utils.model_cost_revalidate import (
    REVALIDATOR,
    Revalidator,
    _revalidation_allowed,
)
from litellm.rust_bridge._native import ModelRegistry
from litellm.utils import get_model_cost_mutation_generation
from tests.unit.litellm_core_utils.test_model_cost_api import (
    _isolated_model_cost_state,
    _local_responses,
    _LocalResponse,
)


@pytest.fixture(autouse=True)
def isolated_catalog() -> Iterator[None]:
    with _isolated_model_cost_state():
        yield


def _bundled_payload() -> bytes:
    entries: Final = copy.deepcopy(ModelRegistry.bundled().served())
    entries["swr/test-only"] = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
    }
    return json.dumps(entries).encode()


def _install_bundled(url: str) -> object:
    rows: Final = load_model_cost_map_at_import(url)
    litellm.model_cost = rows
    return rows


def _run_revalidation(
    url: str,
    *,
    sleep: Callable[[float], None] = lambda _seconds: None,
    max_attempts: int = 3,
) -> None:
    Revalidator(allowed=lambda: True, sleep=sleep, max_attempts=max_attempts)._run(url)


def test_successful_revalidation_adopts_remote_catalog_and_provenance() -> None:
    with _local_responses((_LocalResponse(200, _bundled_payload(), headers=(("ETag", '"remote-v1"'),)),)) as responses:
        live_map: Final = _install_bundled(responses.url)
        _run_revalidation(responses.url)

    assert litellm.model_cost is live_map
    assert "swr/test-only" in litellm.model_cost
    source: Final = get_model_cost_map_source_info()
    assert source["source"] == "remote"
    assert source["url"] == responses.url
    assert source["etag"] == '"remote-v1"'
    assert source["fallback_reason"] is None
    assert source["checked_at"] is not None


def test_not_modified_updates_checked_at_without_adoption() -> None:
    etag: Final = '"remote-v1"'
    with _local_responses(
        (
            _LocalResponse(200, _bundled_payload(), headers=(("ETag", etag),)),
            _LocalResponse(304, b"", headers=(("ETag", etag),)),
        )
    ) as responses:
        live_map: Final = _install_bundled(responses.url)
        _run_revalidation(responses.url)
        remote_registry: Final = GetModelCostMap.loaded_registry()
        before_layers: Final = _MODEL_COST_LAYERS.compose(remote_registry)
        before_source: Final = get_model_cost_map_source_info()
        before_generation: Final = get_model_cost_mutation_generation()
        _run_revalidation(responses.url)

        assert litellm.model_cost is live_map
        assert GetModelCostMap.loaded_registry() is remote_registry
        assert _MODEL_COST_LAYERS.compose(remote_registry) == before_layers
        assert get_model_cost_mutation_generation() == before_generation
        after_source: Final = get_model_cost_map_source_info()
        assert after_source["source"] == before_source["source"] == "remote"
        assert after_source["source_revision"] == before_source["source_revision"]
        assert after_source["etag"] == etag
        assert after_source["loaded_at"] == before_source["loaded_at"]
        assert after_source["checked_at"] != before_source["checked_at"]
        assert responses.request_headers.get(timeout=10).get("if-none-match") is None
        assert responses.request_headers.get(timeout=10).get("if-none-match") == etag


def test_retry_after_is_driven_by_native_sleep_step() -> None:
    sleeps: list[float] = []
    with _local_responses(
        (
            _LocalResponse(503, b"temporary", headers=(("Retry-After", "2"),)),
            _LocalResponse(200, _bundled_payload(), headers=(("ETag", '"remote-v2"'),)),
        )
    ) as responses:
        _install_bundled(responses.url)
        _run_revalidation(responses.url, sleep=sleeps.append)

    assert sleeps == [2.0]
    assert responses.requests.qsize() == 2
    assert get_model_cost_map_source_info()["source"] == "remote"


def test_exhausted_failure_retains_remote_catalog_and_updates_check_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _local_responses(
        (
            _LocalResponse(200, _bundled_payload(), headers=(("ETag", '"remote-v1"'),)),
            (503, b"temporary"),
            (503, b"temporary"),
        )
    ) as responses:
        _install_bundled(responses.url)
        _run_revalidation(responses.url)
        before: Final = get_model_cost_map_source_info()
        _run_revalidation(responses.url, max_attempts=2)

    after: Final = get_model_cost_map_source_info()
    assert after["source"] == "remote"
    assert after["source_revision"] == before["source_revision"]
    assert after["etag"] == before["etag"] == '"remote-v1"'
    assert after["checked_at"] != before["checked_at"]
    assert after["fallback_reason"] is not None
    assert "swr/test-only" in litellm.model_cost
    assert sum("model cost map revalidation failed" in record.getMessage() for record in caplog.records) == 1


def test_start_is_single_flight_while_response_is_held() -> None:
    started = threading.Event()
    release = threading.Event()
    with _local_responses((_LocalResponse(200, _bundled_payload(), started=started, release=release),)) as responses:
        _install_bundled(responses.url)
        revalidator: Final = Revalidator(allowed=lambda: True)
        revalidator.start(responses.url)
        assert started.wait(timeout=10)
        revalidator.start(responses.url)
        release.set()
        worker: Final = revalidator._thread
        assert worker is not None
        worker.join(timeout=10)
        assert not worker.is_alive()

    assert responses.requests.qsize() == 1
    assert "swr/test-only" in litellm.model_cost


def test_start_discards_scheduled_revalidation_after_explicit_reload() -> None:
    sleep_started = threading.Event()
    release_sleep = threading.Event()

    def gated_sleep(_seconds: float) -> None:
        sleep_started.set()
        release_sleep.wait(timeout=10)

    with _local_responses((_LocalResponse(200, _bundled_payload()),)) as responses:
        live_map: Final = _install_bundled(responses.url)
        revalidator: Final = Revalidator(allowed=lambda: True, sleep=gated_sleep)
        revalidator.start(responses.url, jitter=True)
        try:
            assert sleep_started.wait(timeout=10)
            explicit: Final = model_cost_api.reload_model_cost_map(
                model_cost_api.Entries({"explicit/scheduled-reload": {"litellm_provider": "openai", "mode": "chat"}})
            )
            assert isinstance(explicit, model_cost_api.ModelCostReloaded)
            explicit_map: Final = litellm.model_cost
        finally:
            release_sleep.set()
            worker: Final = revalidator._thread
            assert worker is not None
            worker.join(timeout=10)

    assert not worker.is_alive()
    assert responses.requests.qsize() == 0
    assert litellm.model_cost is explicit_map
    assert litellm.model_cost is live_map
    assert "explicit/scheduled-reload" in litellm.model_cost
    assert "swr/test-only" not in litellm.model_cost


def test_stale_remote_adoption_is_discarded_after_explicit_reload() -> None:
    started = threading.Event()
    release = threading.Event()
    with _local_responses((_LocalResponse(200, _bundled_payload(), started=started, release=release),)) as responses:
        live_map: Final = _install_bundled(responses.url)
        revalidator: Final = Revalidator(allowed=lambda: True)
        revalidator.start(responses.url)
        assert started.wait(timeout=10)
        explicit: Final = model_cost_api.reload_model_cost_map(
            model_cost_api.Entries({"explicit/reload": {"litellm_provider": "openai", "mode": "chat"}})
        )
        assert isinstance(explicit, model_cost_api.ModelCostReloaded)
        release.set()
        worker: Final = revalidator._thread
        assert worker is not None
        worker.join(timeout=10)

    assert litellm.model_cost is live_map
    assert "explicit/reload" in litellm.model_cost
    assert "swr/test-only" not in litellm.model_cost


@pytest.mark.parametrize(
    ("variable", "value"),
    [("LITELLM_LOCAL_MODEL_COST_MAP", "True"), ("LITELLM_OFFLINE_MODE", "True")],
)
def test_revalidation_gate_blocks_environment_modes(monkeypatch: pytest.MonkeyPatch, variable: str, value: str) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    monkeypatch.setenv(variable, value)
    assert _revalidation_allowed() is False


def test_revalidation_gate_blocks_cli_and_reserved_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    monkeypatch.setattr(loader, "_is_cli_process", lambda: True)
    assert _revalidation_allowed() is False
    monkeypatch.setattr(loader, "_is_cli_process", lambda: False)
    monkeypatch.setattr(
        "litellm.litellm_core_utils.model_cost_revalidate.is_process_reserved_for_forking",
        lambda: True,
    )
    assert _revalidation_allowed() is False


def test_revalidation_gate_distinguishes_gunicorn_master_and_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    monkeypatch.setattr(loader, "_is_cli_process", lambda: False)
    monkeypatch.setattr(
        "litellm.litellm_core_utils.model_cost_revalidate.is_process_reserved_for_forking",
        lambda: False,
    )
    module = __import__("litellm.litellm_core_utils.model_cost_revalidate", fromlist=["_PRELOAD_MASTER_PID"])
    monkeypatch.setattr(module, "_PRELOAD_MASTER_PID", os.getpid())
    assert _revalidation_allowed() is False
    monkeypatch.setattr(module, "_PRELOAD_MASTER_PID", os.getpid() + 1)
    assert _revalidation_allowed() is True


def test_fork_child_revalidates_local_state_and_skips_inherited_remote_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_child() -> bytes:
        reader, writer = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(reader)
                worker: Final = REVALIDATOR._thread
                if worker is not None:
                    worker.join(timeout=10)
                os.write(writer, b"child")
                os.close(writer)
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(writer)
        _, status = os.waitpid(pid, 0)
        result: Final = os.read(reader, 32)
        os.close(reader)
        assert os.waitstatus_to_exitcode(status) == 0
        return result

    with _local_responses((_LocalResponse(200, _bundled_payload()),)) as local_responses:
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        _install_bundled(local_responses.url)
        REVALIDATOR._url = local_responses.url
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "False")
        assert run_child() == b"child"
        assert local_responses.requests.qsize() == 1

    with _local_responses((_LocalResponse(200, _bundled_payload()),)) as remote_responses:
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        _install_bundled(remote_responses.url)
        _run_revalidation(remote_responses.url)
        REVALIDATOR._url = remote_responses.url
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "False")
        assert run_child() == b"child"
        assert remote_responses.requests.qsize() == 1
