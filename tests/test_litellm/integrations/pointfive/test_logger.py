import asyncio
import gzip
import json
import logging
from collections.abc import Callable

import pytest

from litellm.integrations.pointfive.logger import PointFiveLogger
from litellm.integrations.pointfive.upload_client import PointFiveUploadError
from litellm.types.integrations.pointfive import PointFiveInitParams, PointFiveUploadFailure

OBJECT_KEY = "some/object.ndjson.gz"


class FakeUploadClient:
    """Records the objects a flush produced, so tests can read what would have shipped."""

    def __init__(
        self,
        outcomes: list[str | PointFiveUploadFailure] | None = None,
        ping_failure: PointFiveUploadFailure | None = None,
    ) -> None:
        self.outcomes = outcomes or [OBJECT_KEY]
        self.bodies: list[bytes] = []
        self.on_upload: Callable[[], None] | None = None
        self.ping_failure = ping_failure
        self.pings = 0

    async def ping(self) -> PointFiveUploadFailure | None:
        self.pings += 1
        return self.ping_failure

    async def upload(self, body: bytes) -> str | PointFiveUploadFailure:
        if self.on_upload is not None:
            self.on_upload()
        self.bodies.append(body)
        return self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]

    def records(self) -> list[dict]:
        return [json.loads(line) for body in self.bodies for line in gzip.decompress(body).decode().splitlines()]


def _logger(upload_client: FakeUploadClient, **params) -> PointFiveLogger:
    return PointFiveLogger(params=PointFiveInitParams(**params), upload_client=upload_client)


def _event(request_id: str, size: int = 0) -> dict:
    return {"standard_logging_object": {"id": request_id, "model": "gpt-4o", "blob": "x" * size}}


@pytest.mark.asyncio
async def test_a_flush_ships_one_object_holding_every_buffered_record():
    """One object per flush is the whole point: s3_v2 sends one per request."""
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=3)

    for request_id in ("a", "b", "c"):
        await logger.async_log_success_event(_event(request_id), None, None, None)

    assert len(upload_client.bodies) == 1
    assert [record["id"] for record in upload_client.records()] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_records_are_held_until_the_batch_is_full():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=3)

    await logger.async_log_success_event(_event("a"), None, None, None)

    assert upload_client.bodies == []
    assert len(logger.log_queue) == 1


@pytest.mark.asyncio
async def test_failed_requests_are_logged_too():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)

    await logger.async_log_failure_event(_event("failed"), None, None, None)

    assert [record["id"] for record in upload_client.records()] == ["failed"]


@pytest.mark.asyncio
async def test_an_event_without_a_standard_payload_is_skipped():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)

    await logger.async_log_success_event({"kwargs": "but no payload"}, None, None, None)

    assert upload_client.bodies == []
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_a_batch_over_the_byte_cap_ships_as_several_objects():
    """Record count cannot bound an object: an unredacted payload dwarfs a redacted one."""
    cap = 600
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=4, max_batch_bytes=cap)

    for request_id in ("a", "b", "c", "d"):
        await logger.async_log_success_event(_event(request_id, size=200), None, None, None)

    assert len(upload_client.bodies) > 1
    assert [record["id"] for record in upload_client.records()] == ["a", "b", "c", "d"]
    assert all(len(gzip.decompress(body)) <= cap for body in upload_client.bodies)


@pytest.mark.asyncio
async def test_a_retryable_failure_keeps_the_batch_for_the_next_flush():
    upload_client = FakeUploadClient([PointFiveUploadFailure("upload target is down", retryable=True)])
    logger = _logger(upload_client, batch_size=2)

    for request_id in ("a", "b"):
        await logger.async_log_success_event(_event(request_id), None, None, None)

    assert [record["id"] for record in logger.log_queue] == ["a", "b"]


@pytest.mark.asyncio
async def test_a_retryable_failure_surfaces_so_the_base_logger_can_preserve_it():
    upload_client = FakeUploadClient([PointFiveUploadFailure("upload target is down", retryable=True)])
    logger = _logger(upload_client, batch_size=99)
    logger.log_queue.append(_event("a")["standard_logging_object"])

    with pytest.raises(PointFiveUploadError, match="upload target is down"):
        await logger.async_send_batch()


@pytest.mark.asyncio
async def test_a_rejected_batch_is_dropped_rather_than_blocking_the_queue():
    """Retrying a rejection forever would stall every record queued behind it."""
    upload_client = FakeUploadClient([PointFiveUploadFailure("object too large", retryable=False)])
    logger = _logger(upload_client, batch_size=2)

    for request_id in ("a", "b"):
        await logger.async_log_success_event(_event(request_id), None, None, None)

    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_records_already_queued_ship_with_the_event_that_triggers_the_flush():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)
    logger.log_queue.append(_event("mid-flight")["standard_logging_object"])

    await logger.async_log_success_event(_event("a"), None, None, None)

    assert [record["id"] for record in upload_client.records()] == ["mid-flight", "a"]
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_a_record_that_arrives_mid_flush_is_kept_for_the_next_one():
    """The queue is drained by count, so a record appended mid-upload must survive."""
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)
    upload_client.on_upload = lambda: logger.log_queue.append(_event("late")["standard_logging_object"])

    await logger.async_log_success_event(_event("first"), None, None, None)

    assert [record["id"] for record in upload_client.records()] == ["first"]
    assert [record["id"] for record in logger.log_queue] == ["late"]


def test_defaults_favour_fewer_larger_uploads_over_freshness():
    upload_client = FakeUploadClient()

    logger = _logger(upload_client)

    assert logger.batch_size == 10_000
    assert logger.flush_interval == 300
    assert logger.max_batch_bytes == 8 * 1024 * 1024


def test_the_default_api_url_is_the_pointfive_ingress(monkeypatch):
    """api.pointfive.co is the host the ingress serves; .com does not resolve to it."""
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_env")

    logger = PointFiveLogger()

    assert logger.upload_client.api_url == "https://api.pointfive.co/query"


def test_the_api_key_can_come_from_the_environment(monkeypatch):
    """The proxy ui configures a callback by writing environment variables."""
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_from_env")

    logger = PointFiveLogger()

    assert logger.upload_client.api_key == "p5tu_from_env"


def test_the_api_url_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_env")
    monkeypatch.setenv("POINTFIVE_API_URL", "https://api.staging.pointfive.co/query")

    logger = PointFiveLogger()

    assert logger.upload_client.api_url == "https://api.staging.pointfive.co/query"


def test_config_yaml_wins_over_the_environment(monkeypatch):
    """A value set in config.yaml is explicit, so it outranks whatever the ui left behind."""
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_from_env")
    monkeypatch.setenv("POINTFIVE_API_URL", "https://from-env.example/query")

    logger = PointFiveLogger(
        params=PointFiveInitParams(api_key="p5tu_from_config", api_url="https://from-config.example/query")
    )

    assert logger.upload_client.api_key == "p5tu_from_config"
    assert logger.upload_client.api_url == "https://from-config.example/query"


def test_a_missing_api_key_fails_at_startup_not_at_the_first_flush():
    with pytest.raises(ValueError, match="api key"):
        PointFiveLogger(params=PointFiveInitParams())


def test_an_api_key_can_be_an_environment_reference(monkeypatch):
    """config.yaml spells secrets as `os.environ/NAME`, so the plugin must resolve one."""
    monkeypatch.setenv("POINTFIVE_TEST_KEY", "p5tu_from_env")

    logger = PointFiveLogger(params=PointFiveInitParams(api_key="os.environ/POINTFIVE_TEST_KEY"))

    assert logger.upload_client.api_key == "p5tu_from_env"


def test_params_are_read_from_litellm_settings(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "pointfive_params", {"api_key": "p5tu_configured", "batch_size": 7})

    logger = PointFiveLogger()

    assert logger.upload_client.api_key == "p5tu_configured"
    assert logger.batch_size == 7


def test_an_out_of_range_setting_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        PointFiveInitParams(api_key="p5tu_k", batch_size=0)


@pytest.mark.asyncio
async def test_an_idle_flush_reports_liveness_instead_of_uploading():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=99)

    await logger.flush_queue()

    assert upload_client.pings == 1
    assert upload_client.bodies == []


@pytest.mark.asyncio
async def test_a_flush_with_records_uploads_and_does_not_ping():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)

    await logger.async_log_success_event(_event("a"), None, None, None)

    assert upload_client.pings == 0
    assert len(upload_client.bodies) == 1


@pytest.mark.asyncio
async def test_a_failed_ping_does_not_raise():
    """Liveness is bookkeeping; a proxy must not see errors from it."""
    upload_client = FakeUploadClient(ping_failure=PointFiveUploadFailure("api down", retryable=True))
    logger = _logger(upload_client, batch_size=99)

    await logger.flush_queue()

    assert upload_client.pings == 1


@pytest.mark.asyncio
async def test_health_check_is_healthy_when_the_api_accepts_the_key():
    upload_client = FakeUploadClient()

    assert await _logger(upload_client).async_health_check() == {"status": "healthy", "error_message": None}
    assert upload_client.pings == 1


@pytest.mark.asyncio
async def test_health_check_reports_why_the_api_refused():
    """The ui test button shows this message, so a rejected key has to say so rather than pass."""
    upload_client = FakeUploadClient(ping_failure=PointFiveUploadFailure("key was revoked", retryable=False))

    outcome = await _logger(upload_client).async_health_check()

    assert outcome == {"status": "unhealthy", "error_message": "key was revoked"}


def test_the_client_follows_a_key_and_url_changed_after_startup(monkeypatch):
    """
    The proxy ui writes new values into a running proxy's environment.

    Reading them once at construction would leave the logger talking to the old endpoint
    until someone restarted the proxy.
    """
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_first")
    monkeypatch.setenv("POINTFIVE_API_URL", "https://first.example.invalid/query")
    logger = PointFiveLogger(params=PointFiveInitParams())

    assert logger.upload_client.api_key == "p5tu_first"
    assert logger.upload_client.api_url == "https://first.example.invalid/query"

    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_second")
    monkeypatch.setenv("POINTFIVE_API_URL", "https://second.example.invalid/query")

    assert logger.upload_client.api_key == "p5tu_second"
    assert logger.upload_client.api_url == "https://second.example.invalid/query"


def test_a_configured_key_still_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_from_env")
    logger = PointFiveLogger(params=PointFiveInitParams(api_key="p5tu_from_config"))

    assert logger.upload_client.api_key == "p5tu_from_config"


@pytest.mark.asyncio
async def test_health_check_says_so_when_the_key_was_removed(monkeypatch):
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_present")
    logger = PointFiveLogger(params=PointFiveInitParams())
    monkeypatch.delenv("POINTFIVE_API_KEY")

    outcome = await logger.async_health_check()

    assert outcome["status"] == "unhealthy"
    assert "requires an api key" in (outcome["error_message"] or "")


def _pending_flush_tasks() -> tuple[asyncio.Task, ...]:
    return tuple(task for task in asyncio.all_tasks() if "periodic_flush" in str(task.get_coro()))


@pytest.mark.asyncio
async def test_a_one_shot_logger_leaves_no_flush_task_behind():
    """
    A health check builds a logger for a single answer and drops it.

    Without this, every check would leave a flusher running that keeps pinging for the
    lifetime of the proxy.
    """
    before = _pending_flush_tasks()

    logger = PointFiveLogger(params=PointFiveInitParams(), upload_client=FakeUploadClient(), start_periodic_flush=False)

    assert logger._periodic_flush_task is None
    assert _pending_flush_tasks() == before


@pytest.mark.asyncio
async def test_the_logger_flushes_periodically_by_default():
    logger = PointFiveLogger(params=PointFiveInitParams(), upload_client=FakeUploadClient())

    assert logger._periodic_flush_task is not None
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_params_already_built_are_used_as_they_are(monkeypatch):
    """config.yaml is validated once into a params object; a second validation would be wasted."""
    import litellm

    monkeypatch.setattr(litellm, "pointfive_params", PointFiveInitParams(max_batch_bytes=4096))

    logger = PointFiveLogger(upload_client=FakeUploadClient())

    assert logger.max_batch_bytes == 4096


@pytest.mark.asyncio
async def test_a_dead_flush_task_is_restarted_by_the_next_event():
    """A cancelled or crashed flusher would otherwise leave the queue growing forever."""
    logger = _logger(FakeUploadClient())
    logger._periodic_flush_task.cancel()
    await asyncio.sleep(0)  # let the cancellation land, so the task reports itself done

    await logger.async_log_success_event(_event("after-cancel"), None, None, None)

    assert logger._periodic_flush_task is not None
    assert not logger._periodic_flush_task.done()
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_a_failure_while_queueing_never_breaks_the_request():
    """Logging sits on the request path, so a fault here must not surface to the caller."""

    class ExplodingQueue(list):
        def append(self, _item):
            raise RuntimeError("queue is broken")

    upload_client = FakeUploadClient()
    logger = _logger(upload_client)
    logger.log_queue = ExplodingQueue()

    await logger.async_log_success_event(_event("boom"), None, None, None)

    logger.log_queue = []
    await logger.async_log_success_event(_event("after-the-fault"), None, None, None)
    assert [record["id"] for record in logger.log_queue] == ["after-the-fault"]


@pytest.mark.asyncio
async def test_a_flush_with_nothing_queued_uploads_nothing():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client)

    await logger.async_send_batch()

    assert upload_client.bodies == []


@pytest.mark.asyncio
async def test_the_idle_ping_is_skipped_when_the_key_was_removed(monkeypatch, caplog):
    """A key pulled mid-flight must not turn the periodic flush into an exception."""
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_present")
    logger = PointFiveLogger(params=PointFiveInitParams())
    monkeypatch.delenv("POINTFIVE_API_KEY")

    with caplog.at_level(logging.WARNING):
        await logger.flush_queue()

    assert "liveness ping skipped" in caplog.text
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_a_full_batch_stands_down_while_a_flush_is_already_running():
    """
    Under load every event landing mid-upload also crosses the batch threshold.

    Letting each one flush turns a single burst into a stream of tiny objects, which is
    what batching exists to avoid, so a full batch defers to the flush already running.
    """
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=2)
    release = asyncio.Event()
    finish_upload = upload_client.upload

    async def held_upload(body: bytes):
        await release.wait()
        return await finish_upload(body)

    upload_client.upload = held_upload
    logger.log_queue.extend(_event(f"first-{index}")["standard_logging_object"] for index in range(2))

    flushing = asyncio.create_task(logger.flush_queue())
    await asyncio.sleep(0)

    # Bounded: without the guard these block on the flush lock the held upload owns.
    for index in range(6):
        await asyncio.wait_for(logger.async_log_success_event(_event(f"mid-{index}"), None, None, None), timeout=2)

    assert upload_client.bodies == []

    release.set()
    await flushing

    assert len(upload_client.bodies) == 1
    assert [record["id"] for record in upload_client.records()] == ["first-0", "first-1"]
    assert [record["id"] for record in logger.log_queue] == [f"mid-{index}" for index in range(6)]
