import pytest

from litellm.integrations.vantage.vantage_logger import VantageLogger
from litellm.proxy import proxy_server

FOCUS_LOCK = "cronjob_lock:focus_export_usage_data"
VANTAGE_LOCK = "cronjob_lock:vantage_export_usage_data"


class _InMemoryRedis:
    def __init__(self, locks: dict[str, str]) -> None:
        self.locks = locks
        self.granted: list[tuple[str, str]] = []

    async def async_set_cache(self, key: str, value: str, nx: bool, ttl: int) -> bool:
        if nx and key in self.locks:
            return False
        self.locks[key] = value
        self.granted.append((key, value))
        return True

    async def async_get_cache(self, key: str) -> str | None:
        return self.locks.get(key)

    async def async_delete_cache(self, key: str) -> int:
        return 1 if self.locks.pop(key, None) is not None else 0


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_logging_registered", [True, False])
async def test_scheduled_vantage_export_without_a_redis_lock_runs_until_the_database_boundary(
    monkeypatch: pytest.MonkeyPatch, proxy_logging_registered: bool
) -> None:
    monkeypatch.setattr(proxy_server.proxy_logging_obj.db_spend_update_writer.pod_lock_manager, "redis_cache", None)
    if not proxy_logging_registered:
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    logger = VantageLogger(api_key="vantage-key", integration_token="vantage-token")

    with pytest.raises(RuntimeError, match="Database not connected"):
        await logger.initialize_focus_export_job()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locks_before", "locks_after", "granted"),
    [
        pytest.param({}, {}, [(VANTAGE_LOCK, "this-pod")], id="lock-free"),
        pytest.param({VANTAGE_LOCK: "this-pod"}, {}, [], id="lock-already-held-by-this-pod"),
        pytest.param(
            {FOCUS_LOCK: "another-pod"},
            {FOCUS_LOCK: "another-pod"},
            [(VANTAGE_LOCK, "this-pod")],
            id="focus-lock-held-by-another-pod",
        ),
    ],
)
async def test_scheduled_vantage_export_runs_under_the_vantage_pod_lock_and_releases_it_when_the_export_fails(
    monkeypatch: pytest.MonkeyPatch,
    locks_before: dict[str, str],
    locks_after: dict[str, str],
    granted: list[tuple[str, str]],
) -> None:
    lock_manager = proxy_server.proxy_logging_obj.db_spend_update_writer.pod_lock_manager
    redis = _InMemoryRedis(dict(locks_before))
    monkeypatch.setattr(lock_manager, "redis_cache", redis)
    monkeypatch.setattr(lock_manager, "pod_id", "this-pod")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    logger = VantageLogger(api_key="vantage-key", integration_token="vantage-token")

    with pytest.raises(RuntimeError, match="Database not connected"):
        await logger.initialize_focus_export_job()

    assert redis.locks == locks_after
    assert redis.granted == granted


@pytest.mark.asyncio
async def test_scheduled_vantage_export_is_skipped_while_another_pod_holds_the_vantage_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_manager = proxy_server.proxy_logging_obj.db_spend_update_writer.pod_lock_manager
    redis = _InMemoryRedis({VANTAGE_LOCK: "another-pod"})
    monkeypatch.setattr(lock_manager, "redis_cache", redis)
    monkeypatch.setattr(lock_manager, "pod_id", "this-pod")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    logger = VantageLogger(api_key="vantage-key", integration_token="vantage-token")

    await logger.initialize_focus_export_job()

    assert redis.locks == {VANTAGE_LOCK: "another-pod"}
    assert redis.granted == []
