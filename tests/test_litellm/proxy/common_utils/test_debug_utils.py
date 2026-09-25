import json
import os
import socket
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.bug_report_config import build_proxy_bug_report
from litellm.proxy.common_utils.debug_utils import (
    PSUTIL_MISSING_ERROR,
    _ProcFilesystemProcess,
    _summary_process_memory,
    get_memory_summary,
)
from litellm.proxy.common_utils.debug_utils import router as debug_router

PAGE_SIZE = 4096
STATM_SIZE_PAGES = 100_000
STATM_RESIDENT_PAGES = 30_000
MEMINFO_TOTAL_KB = 1_000_000


@pytest.fixture
def proc_process(tmp_path: Path) -> _ProcFilesystemProcess:
    statm = tmp_path / "statm"
    statm.write_text(f"{STATM_SIZE_PAGES} {STATM_RESIDENT_PAGES} 5000 1 0 20000 0\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal:       {MEMINFO_TOTAL_KB} kB\nMemFree:         400000 kB\nMemAvailable:    600000 kB\n"
    )
    return _ProcFilesystemProcess(statm_path=str(statm), meminfo_path=str(meminfo), page_size=PAGE_SIZE)


def test_proc_filesystem_process_reads_resident_and_virtual_bytes_from_statm(
    proc_process: _ProcFilesystemProcess,
) -> None:
    memory_info = proc_process.memory_info()

    assert memory_info.rss == STATM_RESIDENT_PAGES * PAGE_SIZE
    assert memory_info.vms == STATM_SIZE_PAGES * PAGE_SIZE


def test_proc_filesystem_process_reports_share_of_meminfo_total(proc_process: _ProcFilesystemProcess) -> None:
    expected_percent = STATM_RESIDENT_PAGES * PAGE_SIZE / (MEMINFO_TOTAL_KB * 1024) * 100

    assert proc_process.memory_percent() == pytest.approx(expected_percent)


def test_summary_reports_rss_from_the_proc_filesystem(proc_process: _ProcFilesystemProcess) -> None:
    memory, health_status = _summary_process_memory(proc_process)

    assert memory["ram_usage_mb"] == round(STATM_RESIDENT_PAGES * PAGE_SIZE / (1024 * 1024), 2)
    assert memory["system_memory_percent"] == pytest.approx(12.0)
    assert health_status == "healthy"
    assert "error" not in memory


def test_summary_without_any_memory_source_names_psutil_and_reports_no_rss() -> None:
    memory, health_status = _summary_process_memory(None)

    assert memory == {"error": PSUTIL_MISSING_ERROR}
    assert health_status == "healthy"


@pytest.mark.asyncio
async def test_memory_summary_names_the_host_and_worker_that_answered() -> None:
    summary = await get_memory_summary(UserAPIKeyAuth())

    assert summary["hostname"] == socket.gethostname()
    assert summary["worker_pid"] == os.getpid()
    assert summary["memory"]["ram_usage_mb"] > 0


HOSTILE_CONFIG: Mapping[str, object] = {
    "model_list": [
        {
            "model_name": "acme-prod-gpt4",
            "litellm_params": {
                "model": "azure/acme-gpt4o-deployment",
                "api_base": "https://acme-eastus.openai.azure.com",
                "api_key": "sk-live-secret-1",
            },
        }
    ],
    "litellm_settings": {"drop_params": True, "callbacks": ["langfuse", "acme_hooks.audit_logger"]},
}

HOSTILE_GENERAL_SETTINGS: Mapping[str, object] = {
    "master_key": "sk-live-secret-master",
    "database_url": "postgres://user:hunter2@10.0.0.7/litellm",
    "store_model_in_db": True,
}

HOSTILE_STRINGS = ("acme", "sk-live-secret", "hunter2", "10.0.0.7", "azure.com")


@pytest.fixture
def hostile_proxy_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous_config = proxy_server.proxy_config.get_config_state()
    proxy_server.proxy_config.update_config_state(config=HOSTILE_CONFIG)
    monkeypatch.setattr(proxy_server, "general_settings", dict(HOSTILE_GENERAL_SETTINGS))
    yield
    proxy_server.proxy_config.update_config_state(config=previous_config)


def _debug_client(caller: UserAPIKeyAuth) -> TestClient:
    app = FastAPI()
    app.include_router(debug_router)
    app.dependency_overrides[user_api_key_auth] = lambda: caller
    return TestClient(app)


@pytest.mark.parametrize(
    "caller",
    [
        UserAPIKeyAuth(),
        UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER),
        UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
    ],
)
@pytest.mark.usefixtures("hostile_proxy_config")
def test_debug_report_refuses_everyone_but_proxy_admins(caller: UserAPIKeyAuth) -> None:
    response = _debug_client(caller).get("/debug/report")

    assert response.status_code == 403, response.text
    assert "litellm_version" not in response.text


@pytest.mark.usefixtures("hostile_proxy_config")
def test_debug_report_returns_what_the_bug_report_link_carries_and_nothing_from_the_operator() -> None:
    response = _debug_client(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)).get("/debug/report")

    assert response.status_code == 200, response.text
    assert response.json() == json.loads(json.dumps(asdict(build_proxy_bug_report(RuntimeError("boom")).environment)))
    assert response.json()["config_lines"] == [
        "general_settings.store_model_in_db = true",
        "litellm_settings.drop_params = true",
        "litellm_settings.callbacks = [langfuse]",
        "model_list[*].provider = [azure]",
    ]
    assert not any(hostile in response.text for hostile in HOSTILE_STRINGS), response.text


def test_cache_stats_report_the_redis_pool_through_the_cache_object() -> None:
    from types import SimpleNamespace

    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.common_utils.debug_utils import _get_cache_memory_stats
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    class _PooledRedisCache:
        def connection_pool_status(self) -> Mapping[str, object]:
            return {"max_connections": 7, "connection_class": "Connection"}

    stats = _get_cache_memory_stats(
        user_api_key_cache=UserApiKeyCache(),
        llm_router=None,
        proxy_logging_obj=SimpleNamespace(internal_usage_cache=SimpleNamespace(dual_cache=DualCache())),
        redis_usage_cache=_PooledRedisCache(),
    )

    assert stats["redis_usage_cache"] == {
        "enabled": True,
        "cache_type": "_PooledRedisCache",
        "connection_pool": {"max_connections": 7, "connection_class": "Connection"},
    }
