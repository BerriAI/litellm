import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import chain, repeat

import pytest

import litellm
from litellm.integrations.zerobus.client import ZerobusIngestError
from litellm.integrations.zerobus.logger import ZerobusLogger, connection_for
from litellm.types.integrations.zerobus import ZerobusIngestFailure, ZerobusInitParams

WORKSPACE_URL = "https://dbc-a1b2c3d4-e5f6.cloud.databricks.com"
SERVER_ENDPOINT = "https://1234567890123456.zerobus.us-west-2.cloud.databricks.com"


Row = Mapping[str, object]


class FakeIngestClient:
    """Records the rows each flush would have written; outcomes are served in order and the last one repeats."""

    def __init__(
        self,
        outcomes: Sequence[ZerobusIngestFailure | None] = (None,),
        on_insert: Callable[[], None] | None = None,
    ) -> None:
        self.outcomes: Iterator[ZerobusIngestFailure | None] = chain(outcomes[:-1], repeat(outcomes[-1]))
        self.on_insert = on_insert
        self.batches: tuple[tuple[Row, ...], ...] = ()

    async def insert(self, rows: Sequence[Row]) -> ZerobusIngestFailure | None:
        if self.on_insert is not None:
            self.on_insert()
        self.batches = (*self.batches, tuple(rows))
        return next(self.outcomes)

    def ids(self) -> tuple[object, ...]:
        return tuple(row["id"] for batch in self.batches for row in batch)


def _logger(client: FakeIngestClient, **params: object) -> ZerobusLogger:
    return ZerobusLogger(params=ZerobusInitParams.model_validate(params), client=client)


def _event(request_id: str, **payload: object) -> dict[str, object]:
    return {
        "standard_logging_object": {
            "id": request_id,
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "response": {"choices": []},
            **payload,
        }
    }


async def _settle(logger: ZerobusLogger) -> None:
    for _ in range(200):
        await asyncio.sleep(0.001)
        task = logger._batch_flush_task
        if (task is None or task.done()) and not logger._flushing:
            return


@pytest.mark.asyncio
async def test_a_full_batch_is_written_as_one_insert_of_table_rows():
    client = FakeIngestClient()
    logger = _logger(client, batch_size=3)

    for request_id in ("a", "b", "c"):
        await logger.async_log_success_event(_event(request_id), None, None, None)

    await _settle(logger)
    assert len(client.batches) == 1
    assert client.ids() == ("a", "b", "c")
    assert client.batches[0][0]["model"] == "gpt-4o"
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_rows_are_held_until_the_batch_is_full():
    client = FakeIngestClient()
    logger = _logger(client, batch_size=3)

    await logger.async_log_success_event(_event("a"), None, None, None)

    assert client.batches == ()
    assert len(logger.log_queue) == 1


@pytest.mark.asyncio
async def test_failed_requests_are_written_too():
    client = FakeIngestClient()
    logger = _logger(client, batch_size=1)

    await logger.async_log_failure_event(_event("failed", status="failure", error_str="boom"), None, None, None)

    await _settle(logger)
    assert client.ids() == ("failed",)
    assert client.batches[0][0]["status"] == "failure"
    assert client.batches[0][0]["error_str"] == "boom"


@pytest.mark.asyncio
async def test_an_event_without_a_standard_payload_is_skipped():
    client = FakeIngestClient()
    logger = _logger(client, batch_size=1)

    await logger.async_log_success_event({"kwargs": "but no payload"}, None, None, None)

    assert client.batches == ()
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_a_retryable_failure_keeps_the_rows_for_the_next_flush():
    client = FakeIngestClient([ZerobusIngestFailure("zerobus is down", retryable=True)])
    logger = _logger(client, batch_size=2)

    for request_id in ("a", "b"):
        await logger.async_log_success_event(_event(request_id), None, None, None)

    await _settle(logger)
    assert [row["id"] for row in logger.log_queue] == ["a", "b"]


@pytest.mark.asyncio
async def test_a_retryable_failure_surfaces_so_the_base_logger_can_preserve_it():
    client = FakeIngestClient([ZerobusIngestFailure("zerobus is down", retryable=True)])
    logger = _logger(client, batch_size=99)
    logger.log_queue.append({"id": "a"})

    with pytest.raises(ZerobusIngestError, match="zerobus is down"):
        await logger.async_send_batch()


@pytest.mark.asyncio
async def test_a_rejected_batch_is_dropped_rather_than_blocking_the_queue():
    client = FakeIngestClient([ZerobusIngestFailure("unknown column", retryable=False)])
    logger = _logger(client, batch_size=2)

    for request_id in ("a", "b"):
        await logger.async_log_success_event(_event(request_id), None, None, None)

    await _settle(logger)
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_a_row_that_arrives_mid_flush_is_kept_for_the_next_one():
    client = FakeIngestClient()
    logger = _logger(client, batch_size=1)
    client.on_insert = lambda: logger.log_queue.append({"id": "late"})

    await logger.async_log_success_event(_event("first"), None, None, None)

    await _settle(logger)
    assert client.ids() == ("first",)
    assert [row["id"] for row in logger.log_queue] == ["late"]


@pytest.mark.asyncio
async def test_the_queue_cap_holds_while_an_insert_is_in_flight():
    """A slow insert must not let the queue grow past max_queue_size, nor disturb the in-flight head."""
    insert_started = asyncio.Event()
    finish_insert = asyncio.Event()

    class SlowClient:
        batches: tuple[tuple[Row, ...], ...] = ()

        async def insert(self, rows: Sequence[Row]) -> None:
            insert_started.set()
            await finish_insert.wait()
            self.batches = (*self.batches, tuple(rows))

    client = SlowClient()
    logger = ZerobusLogger(params=ZerobusInitParams(batch_size=2), client=client)
    logger.max_queue_size = 3

    for request_id in ("a", "b"):
        await logger.async_log_success_event(_event(request_id), None, None, None)
    await insert_started.wait()
    for request_id in ("c", "d", "e"):
        await logger.async_log_success_event(_event(request_id), None, None, None)
    finish_insert.set()
    await _settle(logger)

    assert [[row["id"] for row in batch] for batch in client.batches] == [["a", "b"]]
    assert [row["id"] for row in logger.log_queue] == ["c"]


@pytest.mark.asyncio
async def test_a_client_error_does_not_break_the_request_path():
    class ExplodingClient:
        async def insert(self, rows: Sequence[Row]) -> None:
            raise RuntimeError("bug")

    logger = ZerobusLogger(params=ZerobusInitParams(batch_size=1), client=ExplodingClient())

    await logger.async_log_success_event(_event("a"), None, None, None)
    await _settle(logger)

    assert [row["id"] for row in logger.log_queue] == ["a"]


@pytest.mark.asyncio
async def test_turn_off_message_logging_redacts_prompts_and_responses_but_keeps_the_rest():
    client = FakeIngestClient()
    logger = _logger(client, batch_size=1, turn_off_message_logging=True)

    await logger.async_log_success_event(
        _event("a", prompt_tokens=10, response={"choices": [{"message": {"content": "the secret answer"}}]}),
        None,
        None,
        None,
    )

    await _settle(logger)
    (row,) = client.batches[0]
    assert row["id"] == "a"
    assert row["prompt_tokens"] == 10
    assert '"hi"' not in str(row["messages"])
    assert "the secret answer" not in str(row["response"])


def test_connection_comes_from_the_environment_the_proxy_ui_writes(monkeypatch):
    monkeypatch.setenv("ZEROBUS_WORKSPACE_URL", WORKSPACE_URL)
    monkeypatch.setenv("ZEROBUS_SERVER_ENDPOINT", SERVER_ENDPOINT)
    monkeypatch.setenv("ZEROBUS_CLIENT_ID", "sp-id")
    monkeypatch.setenv("ZEROBUS_CLIENT_SECRET", "sp-secret")
    monkeypatch.setenv("ZEROBUS_TABLE_NAME", "main.litellm.traces")

    connection = connection_for(ZerobusInitParams())

    assert connection.workspace_url == WORKSPACE_URL
    assert connection.server_endpoint == SERVER_ENDPOINT
    assert connection.workspace_id == "1234567890123456"
    assert connection.client_id == "sp-id"
    assert connection.client_secret == "sp-secret"
    assert connection.table_name == "main.litellm.traces"


def test_config_yaml_params_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("ZEROBUS_TABLE_NAME", "env.schema.table")
    monkeypatch.setenv("ZEROBUS_CLIENT_SECRET", "from-env")

    connection = connection_for(
        ZerobusInitParams(
            workspace_url=WORKSPACE_URL,
            server_endpoint=SERVER_ENDPOINT,
            client_id="sp-id",
            client_secret="from-config",
            table_name="cfg.schema.table",
        )
    )

    assert connection.table_name == "cfg.schema.table"
    assert connection.client_secret == "from-config"


def test_a_secret_reference_in_config_yaml_is_resolved(monkeypatch):
    monkeypatch.setenv("MY_SP_SECRET", "resolved-secret")

    connection = connection_for(
        ZerobusInitParams(
            workspace_url=WORKSPACE_URL,
            server_endpoint=SERVER_ENDPOINT,
            client_id="sp-id",
            client_secret="os.environ/MY_SP_SECRET",
            table_name="main.litellm.traces",
        )
    )

    assert connection.client_secret == "resolved-secret"


def test_a_missing_setting_names_the_env_var_to_set(monkeypatch):
    monkeypatch.delenv("ZEROBUS_CLIENT_SECRET", raising=False)

    with pytest.raises(ValueError, match="ZEROBUS_CLIENT_SECRET"):
        connection_for(
            ZerobusInitParams(
                workspace_url=WORKSPACE_URL,
                server_endpoint=SERVER_ENDPOINT,
                client_id="sp-id",
                table_name="main.litellm.traces",
            )
        )


def test_a_table_that_is_not_fully_qualified_is_refused():
    with pytest.raises(ValueError, match=r"catalog\.schema\.table"):
        connection_for(
            ZerobusInitParams(
                workspace_url=WORKSPACE_URL,
                server_endpoint=SERVER_ENDPOINT,
                client_id="sp-id",
                client_secret="sp-secret",
                table_name="traces",
            )
        )


def test_an_endpoint_without_a_workspace_id_is_refused():
    """The token's resource needs the numeric workspace id, which only the Zerobus hostname carries."""
    with pytest.raises(ValueError, match="ZEROBUS_SERVER_ENDPOINT"):
        connection_for(
            ZerobusInitParams(
                workspace_url=WORKSPACE_URL,
                server_endpoint=WORKSPACE_URL,
                client_id="sp-id",
                client_secret="sp-secret",
                table_name="main.litellm.traces",
            )
        )


def test_a_misconfigured_logger_fails_at_startup_not_at_first_flush(monkeypatch):
    for name in ("WORKSPACE_URL", "SERVER_ENDPOINT", "CLIENT_ID", "CLIENT_SECRET", "TABLE_NAME"):
        monkeypatch.delenv(f"ZEROBUS_{name}", raising=False)
    monkeypatch.setattr(litellm, "zerobus_params", None)

    with pytest.raises(ValueError, match="ZEROBUS_"):
        ZerobusLogger()


def test_litellm_zerobus_params_configure_the_logger(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "zerobus_params",
        {
            "workspace_url": WORKSPACE_URL,
            "server_endpoint": SERVER_ENDPOINT,
            "client_id": "sp-id",
            "client_secret": "sp-secret",
            "table_name": "main.litellm.traces",
            "batch_size": 7,
            "flush_interval": 3,
        },
    )

    logger = ZerobusLogger()

    assert logger.batch_size == 7
    assert logger.flush_interval == 3
    assert logger.client.connection.table_name == "main.litellm.traces"


def test_the_client_is_kept_while_the_connection_is_unchanged_and_rebuilt_when_it_changes(monkeypatch):
    """The client caches its token, so it must survive across flushes, yet a UI edit must take effect."""
    monkeypatch.setenv("ZEROBUS_WORKSPACE_URL", WORKSPACE_URL)
    monkeypatch.setenv("ZEROBUS_SERVER_ENDPOINT", SERVER_ENDPOINT)
    monkeypatch.setenv("ZEROBUS_CLIENT_ID", "sp-id")
    monkeypatch.setenv("ZEROBUS_CLIENT_SECRET", "sp-secret")
    monkeypatch.setenv("ZEROBUS_TABLE_NAME", "main.litellm.traces")
    monkeypatch.setattr(litellm, "zerobus_params", None)
    logger = ZerobusLogger()

    first = logger.client
    unchanged = logger.client
    monkeypatch.setenv("ZEROBUS_TABLE_NAME", "main.litellm.traces_v2")
    rebuilt = logger.client

    assert unchanged is first
    assert rebuilt is not first
    assert rebuilt.connection.table_name == "main.litellm.traces_v2"


def test_callbacks_zerobus_builds_one_logger_and_reuses_it(monkeypatch):
    """`litellm_settings.callbacks: ["zerobus"]` goes through litellm_logging, which must hand back one instance."""
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setenv("ZEROBUS_WORKSPACE_URL", WORKSPACE_URL)
    monkeypatch.setenv("ZEROBUS_SERVER_ENDPOINT", SERVER_ENDPOINT)
    monkeypatch.setenv("ZEROBUS_CLIENT_ID", "sp-id")
    monkeypatch.setenv("ZEROBUS_CLIENT_SECRET", "sp-secret")
    monkeypatch.setenv("ZEROBUS_TABLE_NAME", "main.litellm.traces")
    monkeypatch.setattr(litellm, "zerobus_params", None)
    monkeypatch.setattr(logging_module, "_in_memory_loggers", [])

    assert logging_module.get_custom_logger_compatible_class("zerobus") is None

    first = logging_module._init_custom_logger_compatible_class(
        logging_integration="zerobus", internal_usage_cache=None, llm_router=None, custom_logger_init_args={}
    )
    second = logging_module._init_custom_logger_compatible_class(
        logging_integration="zerobus", internal_usage_cache=None, llm_router=None, custom_logger_init_args={}
    )

    assert isinstance(first, ZerobusLogger)
    assert second is first
    assert logging_module.get_custom_logger_compatible_class("zerobus") is first
