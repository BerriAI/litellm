import asyncio
import json
import threading
from collections.abc import Callable, Iterator
from typing import Final, cast
from unittest.mock import patch

import pytest
from tests.unit.integrations.s3_v2_rust_support import ScriptedBucket, scripted_bucket

import litellm
from litellm.integrations import s3_v2
from litellm.integrations.custom_logger_v2 import (
    CallbackRegistry,
    CallContext,
    ContractViolation,
    CustomLoggerV2,
    Dispatcher,
    Field,
    Hook,
    RequestSelection,
    on,
)
from litellm.integrations.custom_logger_v2.host import bind_scope
from litellm.integrations.custom_logger_v2.rust import RustLogger, flush_rust_loggers
from litellm.integrations.s3_v2 import S3Logger

RustFactory = Callable[[ScriptedBucket], RustLogger]


def _s3_v2(bucket: ScriptedBucket) -> RustLogger:
    logger: Final = S3Logger(
        {
            "s3_bucket_name": "enforced-bucket",
            "s3_endpoint_url": bucket.url,
            "s3_aws_access_key_id": "test-key",
            "s3_aws_secret_access_key": "test-secret",
            "s3_region_name": "us-east-1",
        }
    )
    _ = logger.periodic_flush_task.cancel()
    return logger


RUST_LOGGERS: Final[dict[str, RustFactory]] = {"s3_v2": _s3_v2}


@pytest.fixture
def bucket() -> Iterator[ScriptedBucket]:
    with scripted_bucket() as scripted:
        yield scripted


@pytest.fixture
def enforced(monkeypatch: pytest.MonkeyPatch) -> Iterator[Dispatcher]:
    for name in ("callbacks", "input_callback", "success_callback", "failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    for name in ("_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [])
    dispatcher: Final = Dispatcher(CallbackRegistry(), enforce=True)
    with dispatcher.open_scope(RequestSelection()) as scope, bind_scope(scope):
        yield dispatcher


class _UndeclaredReader(CustomLoggerV2):
    def __init__(self) -> None:
        self.refused: list[str] = []  # mutable-ok: the refusals the enforced view raised
        self.done = threading.Event()

    @on(Hook.attempt_ended, reads={Field.model}, awaited=True)
    async def attempt_ended(self, view: object, ctx: CallContext[object]) -> None:
        try:
            self.refused.append(repr(cast(object, getattr(view, Field.messages.value))))
        except ContractViolation as refusal:
            self.refused.append(str(refusal))
        self.done.set()


async def test_the_enforced_scope_refuses_an_undeclared_read(
    enforced: Dispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader: Final = _UndeclaredReader()
    monkeypatch.setattr(litellm, "callbacks", [reader])

    _ = await litellm.acompletion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], mock_response="ok")

    assert await asyncio.to_thread(reader.done.wait, 30)
    assert reader.refused == ["messages is not among the declared reads"]


@pytest.mark.parametrize("factory", RUST_LOGGERS.values(), ids=RUST_LOGGERS.keys())
async def test_a_rust_logger_delivers_its_row_with_only_its_declared_reads(
    enforced: Dispatcher, monkeypatch: pytest.MonkeyPatch, bucket: ScriptedBucket, factory: RustFactory
) -> None:
    logger: Final = factory(bucket)
    monkeypatch.setattr(litellm, "callbacks", [logger])

    with patch.object(s3_v2._Host, "callback_failed") as failed:  # pyright: ignore[reportPrivateUsage]  # the S3 port reports a swallowed failure here
        response: Final = await litellm.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], mock_response="ok"
        )
        for _ in range(500):
            await flush_rust_loggers([logger])
            if bucket.puts():
                break
            await asyncio.sleep(0.01)

    (put,) = bucket.puts()
    assert isinstance(response, litellm.ModelResponse)
    assert json.loads(put.body)["id"] == response.id
    failed.assert_not_called()
    assert enforced.failures == {}
