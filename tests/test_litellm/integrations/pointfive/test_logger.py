import asyncio
import gzip
import json
import logging
from collections.abc import Callable

import pytest

from litellm.integrations.pointfive.logger import PointFiveLogger
from litellm.integrations.pointfive.upload_client import PointFiveUploadError
from litellm.types.integrations.pointfive import DEFAULT_API_URL, PointFiveInitParams, PointFiveUploadFailure

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

    await _settle(logger)
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

    await _settle(logger)
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

    await _settle(logger)
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

    await _settle(logger)
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_records_already_queued_ship_with_the_event_that_triggers_the_flush():
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)
    logger.log_queue.append(_event("mid-flight")["standard_logging_object"])

    await logger.async_log_success_event(_event("a"), None, None, None)

    await _settle(logger)
    assert [record["id"] for record in upload_client.records()] == ["mid-flight", "a"]
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_a_record_that_arrives_mid_flush_is_kept_for_the_next_one():
    """The queue is drained by count, so a record appended mid-upload must survive."""
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)
    upload_client.on_upload = lambda: logger.log_queue.append(_event("late")["standard_logging_object"])

    await logger.async_log_success_event(_event("first"), None, None, None)

    await _settle(logger)
    assert [record["id"] for record in upload_client.records()] == ["first"]
    assert [record["id"] for record in logger.log_queue] == ["late"]


def test_defaults_favour_fewer_larger_uploads_over_freshness():
    upload_client = FakeUploadClient()

    logger = _logger(upload_client)

    assert logger.batch_size == 1_000
    assert logger.flush_interval == 300
    assert logger.max_batch_bytes == 8 * 1024 * 1024


def test_the_default_api_url_is_the_pointfive_ingress(monkeypatch):
    """api.pointfive.co is the host the ingress serves; .com does not resolve to it."""
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_env")

    logger = PointFiveLogger()

    assert logger.upload_client.api_url == "https://api.pointfive.co/api/v1/ingestion"


def test_the_api_key_can_come_from_the_environment(monkeypatch):
    """The proxy ui configures a callback by writing environment variables."""
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_from_env")

    logger = PointFiveLogger()

    assert logger.upload_client.api_key == "p5tu_from_env"


def test_the_api_url_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_env")
    monkeypatch.setenv("POINTFIVE_API_URL", "https://api.staging.pointfive.co/api/v1/ingestion")

    logger = PointFiveLogger()

    assert logger.upload_client.api_url == "https://api.staging.pointfive.co/api/v1/ingestion"


def test_config_yaml_wins_over_the_environment(monkeypatch):
    """A value set in config.yaml is explicit, so it outranks whatever the ui left behind."""
    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_from_env")
    monkeypatch.setenv("POINTFIVE_API_URL", "https://from-env.example/api/v1/ingestion")

    logger = PointFiveLogger(
        params=PointFiveInitParams(api_key="p5tu_from_config", api_url="https://from-config.example/api/v1/ingestion")
    )

    assert logger.upload_client.api_key == "p5tu_from_config"
    assert logger.upload_client.api_url == "https://from-config.example/api/v1/ingestion"


def test_a_missing_api_key_fails_at_startup_not_at_the_first_flush(monkeypatch):
    monkeypatch.delenv("POINTFIVE_API_KEY", raising=False)

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

    await _settle(logger)
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
    monkeypatch.setenv("POINTFIVE_API_URL", "https://first.example.invalid/api/v1/ingestion")
    logger = PointFiveLogger(params=PointFiveInitParams())

    assert logger.upload_client.api_key == "p5tu_first"
    assert logger.upload_client.api_url == "https://first.example.invalid/api/v1/ingestion"

    monkeypatch.setenv("POINTFIVE_API_KEY", "p5tu_second")
    monkeypatch.setenv("POINTFIVE_API_URL", "https://second.example.invalid/api/v1/ingestion")

    assert logger.upload_client.api_key == "p5tu_second"
    assert logger.upload_client.api_url == "https://second.example.invalid/api/v1/ingestion"


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
    # No periodic task: this test drives the flushes itself, and the loop's opening cycle
    # would otherwise ship the queue it seeds below.
    logger = PointFiveLogger(
        params=PointFiveInitParams(batch_size=2),
        upload_client=upload_client,
        start_periodic_flush=False,
    )
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


async def _settle(logger) -> None:
    """
    Wait out the flush a full batch schedules, the way the proxy's loop would.

    The upload runs off the request path now, and either the batch task or the periodic
    loop can be the one carrying it, so this waits for whichever is in flight to finish.
    """
    for _ in range(200):
        await asyncio.sleep(0.001)
        task = logger._batch_flush_task
        if task is not None and not task.done():
            await task
        if not logger._flushing:
            return
    raise AssertionError("the flush never finished")


async def _until(done: Callable[[], bool], ticks: int = 400) -> None:
    """Wait for a condition the flush path reaches only after gzip finishes on a worker thread."""
    for _ in range(ticks):
        if done():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_a_new_logger_announces_itself_without_waiting_for_the_interval():
    """
    Configuring the callback must make the integration connect, with no traffic and no test click.

    The inherited loop sleeps a whole interval before its first flush, which left a freshly
    configured proxy silent for five minutes and the integration looking unconfigured.
    """
    upload_client = FakeUploadClient()
    logger = _logger(upload_client)

    await asyncio.sleep(0)  # let the flush task reach its first cycle

    assert upload_client.pings == 1
    assert upload_client.bodies == []
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_the_first_cycle_ships_records_rather_than_announcing():
    """Announcing is only for an empty queue: records already waiting must go out as an upload."""
    upload_client = FakeUploadClient()
    logger = PointFiveLogger(
        params=PointFiveInitParams(batch_size=100),
        upload_client=upload_client,
        start_periodic_flush=False,
    )
    logger.log_queue.append(_event("queued-before-start")["standard_logging_object"])

    logger._periodic_flush_task = logger._start_periodic_flush_task()
    await _until(lambda: bool(upload_client.bodies))

    assert upload_client.pings == 0
    assert [record["id"] for record in upload_client.records()] == ["queued-before-start"]
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_a_failed_request_ships_redacted_when_message_logging_is_off():
    """
    Failure events skip the framework's redaction, so the callback has to redact what it buffers.

    Without this the prompt of every failed request reaches PointFive in full, even though
    the integration was configured not to send message content.
    """
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1, turn_off_message_logging=True)
    event = _event("failed-request")
    event["standard_logging_object"]["messages"] = [{"role": "user", "content": "my secret prompt"}]
    event["standard_logging_object"]["response"] = "the secret answer"

    await logger.async_log_failure_event(event, None, None, None)
    await _settle(logger)

    shipped = upload_client.records()[0]
    assert "my secret prompt" not in json.dumps(shipped)
    assert "the secret answer" not in json.dumps(shipped)
    assert event["standard_logging_object"]["messages"][0]["content"] == "my secret prompt"
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_a_dead_loop_does_not_strand_the_flusher():
    """A task whose loop was closed never runs and never reports done, so it must be replaced."""
    logger = PointFiveLogger(params=PointFiveInitParams(), upload_client=FakeUploadClient(), start_periodic_flush=False)
    stranded_loop = asyncio.new_event_loop()
    forever = asyncio.sleep(3600)
    logger._periodic_flush_task = stranded_loop.create_task(forever)
    stranded_loop.close()
    forever.close()

    await logger.async_log_success_event(_event("after-loop-close"), None, None, None)

    assert logger._periodic_flush_task is not None
    assert logger._periodic_flush_task.get_loop() is asyncio.get_running_loop()
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_a_retry_does_not_resend_objects_that_already_landed():
    """
    A failure part way through a multi-object flush used to hand the whole batch back.

    Every record already shipped, and every record already refused for good, went out
    again on the next flush, so PointFive received duplicates of both.
    """
    upload_client = FakeUploadClient(outcomes=[OBJECT_KEY, PointFiveUploadFailure("service busy", retryable=True)])
    logger = PointFiveLogger(
        params=PointFiveInitParams(max_batch_bytes=1),  # one record per object
        upload_client=upload_client,
        start_periodic_flush=False,
    )
    logger.log_queue.extend(_event(request_id)["standard_logging_object"] for request_id in ("first", "second"))

    with pytest.raises(PointFiveUploadError):
        await logger.async_send_batch()

    assert [record["id"] for record in logger.log_queue] == ["second"]


@pytest.mark.asyncio
async def test_the_queue_stops_growing_at_its_cap_without_waiting_for_a_failure():
    """The base class trims only after a failed send, so a proxy that keeps flushing never trims."""
    logger = _logger(FakeUploadClient(), batch_size=10_000)
    logger.max_queue_size = 3

    for request_id in ("a", "b", "c", "d", "e"):
        await logger.async_log_success_event(_event(request_id), None, None, None)

    assert [record["id"] for record in logger.log_queue] == ["c", "d", "e"]
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_a_failed_request_honours_the_global_redaction_setting(monkeypatch):
    """
    Redaction can be turned on globally or per request, not only on this callback.

    The async failure path hands the payload over untouched, so a tenant could trigger a
    provider failure and ship prompts that the operator had already asked to be redacted.
    """
    import litellm

    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1)
    event = _event("globally-redacted")
    event["standard_logging_object"]["messages"] = [{"role": "user", "content": "my secret prompt"}]

    await logger.async_log_failure_event(event, None, None, None)

    await _settle(logger)
    assert "my secret prompt" not in json.dumps(upload_client.records()[0])
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_excluded_fields_are_dropped_from_a_failed_request(monkeypatch):
    """standard_logging_payload_excluded_fields drops a field entirely; failures skipped it too."""
    import litellm

    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", ["messages"])
    upload_client = FakeUploadClient()
    logger = _logger(upload_client, batch_size=1, turn_off_message_logging=True)
    event = _event("field-excluded")
    event["standard_logging_object"]["messages"] = [{"role": "user", "content": "my secret prompt"}]

    await logger.async_log_failure_event(event, None, None, None)
    await _settle(logger)

    shipped = upload_client.records()[0]
    assert "messages" not in shipped
    assert shipped["id"] == "field-excluded"
    logger._periodic_flush_task.cancel()


def _held_upload(upload_client: FakeUploadClient, release: asyncio.Event) -> None:
    finish = upload_client.upload

    async def held(body: bytes):
        await release.wait()
        return await finish(body)

    upload_client.upload = held


@pytest.mark.asyncio
async def test_a_full_batch_does_not_hold_the_request():
    """
    The upload belongs off the request path.

    Awaiting it inline meant a hung PointFive api held the caller's response open for as
    long as the attempts and their backoff took.
    """
    upload_client = FakeUploadClient()
    release = asyncio.Event()
    _held_upload(upload_client, release)
    logger = _logger(upload_client, batch_size=1)

    await asyncio.wait_for(logger.async_log_success_event(_event("first"), None, None, None), timeout=2)

    assert upload_client.bodies == []
    release.set()
    await _settle(logger)
    assert [record["id"] for record in upload_client.records()] == ["first"]
    logger._periodic_flush_task.cancel()


@pytest.mark.asyncio
async def test_records_arriving_during_a_flush_survive_the_queue_cap():
    """
    The flush drains by count, so trimming the front underneath it loses records.

    Records that arrived while the upload was in flight would be deleted by that drain
    without ever being sent.
    """
    upload_client = FakeUploadClient()
    release = asyncio.Event()
    _held_upload(upload_client, release)
    logger = PointFiveLogger(
        params=PointFiveInitParams(batch_size=2),
        upload_client=upload_client,
        start_periodic_flush=False,
    )
    logger.max_queue_size = 2
    logger.log_queue.extend(_event(request_id)["standard_logging_object"] for request_id in ("a", "b"))

    flushing = asyncio.create_task(logger.flush_queue())
    await asyncio.sleep(0.01)
    for request_id in ("c", "d", "e"):
        await logger.async_log_success_event(_event(request_id), None, None, None)
    release.set()
    await flushing

    assert [record["id"] for record in upload_client.records()] == ["a", "b"]
    assert [record["id"] for record in logger.log_queue] == ["c", "d", "e"]
    logger._periodic_flush_task.cancel()


def test_an_unset_env_reference_is_never_used_as_the_key(monkeypatch):
    """
    A config that names a missing variable has no key, and must say so.

    Falling back to the reference text sent the literal "os.environ/NAME" as the bearer
    token, so the callback started and every upload was rejected for the wrong reason.
    """
    monkeypatch.delenv("POINTFIVE_API_KEY", raising=False)
    monkeypatch.delenv("POINTFIVE_MISSING_KEY", raising=False)

    with pytest.raises(ValueError, match="requires an api key"):
        PointFiveLogger(
            params=PointFiveInitParams(api_key="os.environ/POINTFIVE_MISSING_KEY"),
            start_periodic_flush=False,
        )


def test_an_unset_url_reference_falls_back_to_the_public_endpoint(monkeypatch):
    """An unresolved url reference must not become the destination the proxy uploads to."""
    from litellm.integrations.pointfive.logger import _resolved_api_url

    monkeypatch.delenv("POINTFIVE_API_URL", raising=False)
    monkeypatch.delenv("POINTFIVE_MISSING_URL", raising=False)

    assert _resolved_api_url(PointFiveInitParams(api_url="os.environ/POINTFIVE_MISSING_URL")) == DEFAULT_API_URL
