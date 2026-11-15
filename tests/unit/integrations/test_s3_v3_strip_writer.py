import asyncio
import copy
import datetime
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.custom_logger_v2.manifest import build_manifest
from litellm.integrations.s3_v3 import STRIP_CALLBACK_NAME, S3Logger, S3StripBase64Files
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge import _native
from litellm.types.integrations.custom_logger_v2 import Outcome
from litellm.types.utils import ModelResponse
from tests.unit.integrations.s3_v3_rust_support import ScriptedBucket, scripted_bucket

pytestmark = pytest.mark.skipif(not hasattr(_native, "NativeS3Log"), reason="the built _native has no NativeS3Log")

START: Final = datetime.datetime(2025, 9, 14, 9, 5, 7, 42)

RECORDED_AT_86B70FEF16: Final = {
    "stripped": (
        "/test-bucket/2025-09-14/time-09-05-07-000042_stripped.json",
        "31e903035e2185abea6cac867139417cb42a37a0da6b7d4efea47b9d1c24f1e1",
    ),
    "excluded": (
        "/test-bucket/2025-09-14/time-09-05-07-000042_excluded.json",
        "5958206c8eee87b3221e6ee760b48a0ead92343768adbf8f6f80295bd4d51329",
    ),
    "failed": (
        "/test-bucket/2025-09-14/time-09-05-07-000042_failed.json",
        "04dcb95630cb5b61fbf96560be30411b5fcacebbb4f4ed03aeaccc2e0687c95f",
    ),
}

STRIPPED_MESSAGES: Final = [
    {"role": "user", "content": [{"type": "text", "text": "see [BASE64_REDACTED]"}]},
    {"role": "assistant", "content": "plain [BASE64_REDACTED]", "name": "[BASE64_REDACTED]"},
]


def _messages() -> list[object]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see data:image/png;base64,AAAA== inline"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
                {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBER"}},
            ],
        },
        {"role": "assistant", "content": "plain data:audio/wav;base64,CCCC tail", "name": "data:image/gif;base64,DDDD"},
        "not a message",
    ]


def _payload(response_id: str, status: str = "success") -> dict[str, object]:
    return {
        "id": response_id,
        "status": status,
        "metadata": {},
        "response": {"text": "ok"},
        "messages": _messages(),
    }


class _Reader(CustomLogger):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: Final = name
        self.seen: list[object] = []

    def saw(self, kwargs: object) -> None:
        payload: Final = kwargs.get("standard_logging_object") if isinstance(kwargs, Mapping) else None
        self.seen.append(copy.deepcopy(payload.get("messages")) if isinstance(payload, Mapping) else None)

    async def async_log_success_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.saw(kwargs)


class _SyncReader(_Reader):
    def log_success_event(self, kwargs: object, response_obj: object, start_time: object, end_time: object) -> None:
        self.saw(kwargs)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("S3_PARTITION_GRANULARITY", "S3_LOG_PROMPTS_ONLY", "AWS_REGION", "AWS_REGION_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", None)
    monkeypatch.setattr(litellm, "s3_callback_params", None)
    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", None)
    for name in (
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    yield


@pytest.fixture
def bucket() -> Iterator[ScriptedBucket]:
    with scripted_bucket() as scripted:
        yield scripted


def _params(bucket: ScriptedBucket, strip: bool, region: str = "us-east-1") -> dict[str, object]:
    return {
        "s3_bucket_name": "test-bucket",
        "s3_endpoint_url": bucket.url,
        "s3_aws_access_key_id": "test-key",
        "s3_aws_secret_access_key": "test-secret",
        "s3_region_name": region,
        **({"s3_strip_base64_files": True} if strip else {}),
    }


def _s3(bucket: ScriptedBucket, strip: bool = True, region: str = "us-east-1") -> S3Logger:
    logger: Final = S3Logger(_params(bucket, strip, region))
    _ = logger.periodic_flush_task.cancel()
    return logger


def _registered(*callbacks: object) -> list[object]:
    for callback in callbacks:
        assert isinstance(callback, CustomLogger | S3Logger)
        litellm.logging_callback_manager.add_litellm_async_success_callback(callback)
    return list(litellm._async_success_callback)


def _deliver(callback: object, kwargs: object) -> None:
    if isinstance(callback, _Reader):
        callback.saw(kwargs)


def _prepared(_callback: object, _kwargs: object) -> None:
    return None


async def _run_line(callbacks: Sequence[object], kwargs: dict[str, object], outcome: Outcome = Outcome.success) -> None:
    if outcome is Outcome.success:
        await litellm_logging._run_callbacks(
            Hook.log_prepare,
            callbacks,
            kwargs,
            _prepared,
            args=HookArgs(request=kwargs, start_time=START, end_time=START),
        )
    await litellm_logging._run_callbacks(
        Hook.attempt_ended,
        callbacks,
        kwargs,
        _deliver,
        args=HookArgs(request=kwargs, start_time=START, end_time=START, outcome=outcome),
    )


async def _flushed(logger: S3Logger) -> None:
    for _ in range(100):
        if logger.sink.queued_keys():
            break
        await asyncio.sleep(0.02)
    await logger.flush_queue()


def _recorded(bucket: ScriptedBucket, case: str) -> bytes:
    (put,) = bucket.puts()
    assert (put.path, hashlib.sha256(put.body).hexdigest()) == RECORDED_AT_86B70FEF16[case]
    return put.body


@pytest.mark.asyncio
async def test_strip_on_success_patches_readers_before_and_after_s3_and_uploads_the_recorded_object(
    bucket: ScriptedBucket,
) -> None:
    before: Final = _Reader("before")
    after: Final = _Reader("after")
    s3: Final = _s3(bucket)
    callbacks: Final = _registered(before, s3, after)
    assert callbacks == [before, s3.v2_companion, s3, after]
    kwargs: Final[dict[str, object]] = {"standard_logging_object": _payload("stripped"), "litellm_params": {}}

    await _run_line(callbacks, kwargs)
    await _flushed(s3)

    assert before.seen == after.seen == [STRIPPED_MESSAGES]
    body: Final = _recorded(bucket, "stripped")
    assert json.loads(body)["messages"] == STRIPPED_MESSAGES


@pytest.mark.asyncio
async def test_the_object_matches_the_attempt_ended_strip_alone(bucket: ScriptedBucket) -> None:
    patched: Final = _s3(bucket)
    await _run_line(_registered(patched), {"standard_logging_object": _payload("same"), "litellm_params": {}})
    await _flushed(patched)
    unpatched: Final = _s3(bucket)
    await _run_line([unpatched], {"standard_logging_object": _payload("same"), "litellm_params": {}})
    await _flushed(unpatched)

    first, second = bucket.puts()
    assert (first.path, first.body) == (second.path, second.body)


@pytest.mark.asyncio
async def test_strip_on_with_excluded_fields_uploads_the_recorded_object(
    bucket: ScriptedBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", ["response"])
    after: Final = _Reader("after")
    s3: Final = _s3(bucket)
    kwargs: Final[dict[str, object]] = {"standard_logging_object": _payload("excluded"), "litellm_params": {}}

    await _run_line(_registered(s3, after), kwargs)
    await _flushed(s3)

    assert after.seen == [STRIPPED_MESSAGES]
    body: Final = json.loads(_recorded(bucket, "excluded"))
    assert "response" not in body
    assert body["messages"] == STRIPPED_MESSAGES


@pytest.mark.asyncio
async def test_strip_on_failure_still_uploads_the_recorded_stripped_object(bucket: ScriptedBucket) -> None:
    s3: Final = _s3(bucket)
    kwargs: Final[dict[str, object]] = {
        "standard_logging_object": _payload("failed", status="failure"),
        "litellm_params": {},
    }

    await _run_line(_registered(s3), kwargs, Outcome.failure)
    await _flushed(s3)

    assert json.loads(_recorded(bucket, "failed"))["messages"] == STRIPPED_MESSAGES


@pytest.mark.asyncio
async def test_strip_off_registers_no_writer_and_readers_see_the_original_messages(bucket: ScriptedBucket) -> None:
    before: Final = _Reader("before")
    s3: Final = _s3(bucket, strip=False)
    callbacks: Final = _registered(before, s3)
    kwargs: Final[dict[str, object]] = {"standard_logging_object": _payload("kept"), "litellm_params": {}}

    await _run_line(callbacks, kwargs)
    await _flushed(s3)

    assert s3.v2_companion is None
    assert callbacks == [before, s3]
    assert before.seen == [_messages()]
    assert json.loads(bucket.puts()[0].body)["messages"] == _messages()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "litellm_params",
    [
        pytest.param({"no-log": True}, id="no-log"),
        pytest.param({"proxy_server_request": {"headers": {"x-litellm-disable-callbacks": "s3_v2"}}}, id="opt-out"),
    ],
)
async def test_a_request_s3_skips_gets_no_patch_and_nothing_is_uploaded(
    bucket: ScriptedBucket, monkeypatch: pytest.MonkeyPatch, litellm_params: dict[str, object]
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "allow_dynamic_callback_disabling", True)
    monkeypatch.setattr(proxy_server, "premium_user", True)
    before: Final = _Reader("before")
    after: Final = _Reader("after")
    s3: Final = _s3(bucket)
    kwargs: Final[dict[str, object]] = {
        "standard_logging_object": _payload("skipped"),
        "litellm_params": litellm_params,
    }

    await _run_line(_registered(before, s3, after), kwargs)
    await s3.flush_queue()

    assert before.seen == after.seen == [_messages()]
    assert bucket.puts() == []


@pytest.mark.asyncio
async def test_the_writer_patches_nothing_on_the_sync_line(bucket: ScriptedBucket) -> None:
    s3: Final = _s3(bucket)
    writer: Final = s3.v2_companion
    assert isinstance(writer, S3StripBase64Files)
    payload: Final = _payload("sync")
    kwargs: Final = {"standard_logging_object": payload, "litellm_params": {}}

    await asyncio.to_thread(
        litellm_logging._run_callbacks_sync,
        Hook.log_prepare,
        [writer, s3],
        kwargs,
        lambda _callback, _kwargs: None,
        args=HookArgs(request=kwargs, start_time=START, end_time=START),
    )

    assert payload["messages"] == _messages()


@pytest.mark.asyncio
async def test_an_acompletion_shows_every_reader_the_messages_s3_stores(
    bucket: ScriptedBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    before: Final = _Reader("before")
    after: Final = _Reader("after")
    s3: Final = _s3(bucket)
    monkeypatch.setattr(litellm, "callbacks", [before, s3, after])

    await litellm.acompletion(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see data:image/png;base64,AAAA== inline"},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBER"}},
                ],
            }
        ],
        mock_response="hi there",
    )
    for _ in range(100):
        if after.seen:
            break
        await asyncio.sleep(0.02)
    await _flushed(s3)

    assert litellm._async_success_callback == [before, s3.v2_companion, s3, after]
    expected: Final = [{"role": "user", "content": [{"type": "text", "text": "see [BASE64_REDACTED]"}]}]
    assert before.seen == after.seen == [expected]
    assert json.loads(bucket.puts()[0].body)["messages"] == expected


def test_a_sync_success_leaves_the_messages_for_readers_on_the_sync_line(bucket: ScriptedBucket) -> None:
    async def built() -> S3Logger:
        return _s3(bucket)

    s3: Final = asyncio.run(built())
    before: Final = _SyncReader("before")
    litellm.logging_callback_manager.add_litellm_success_callback(before)
    litellm.logging_callback_manager.add_litellm_success_callback(s3)
    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "see data:image/png;base64,AAAA== inline"}],
        stream=False,
        call_type="completion",
        start_time=START,
        litellm_call_id="call-sync",
        function_id="fn-sync",
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={})

    logging_obj.success_handler(result=ModelResponse(id="served"), start_time=START, end_time=START)

    assert litellm.success_callback == [before, s3.v2_companion, s3]
    assert before.seen == [[{"role": "user", "content": "see data:image/png;base64,AAAA== inline"}]]
    assert bucket.puts() == []


@pytest.mark.asyncio
async def test_registering_s3_again_adds_no_second_writer(bucket: ScriptedBucket) -> None:
    s3: Final = _s3(bucket)
    _ = _registered(s3, s3)
    litellm.logging_callback_manager.add_litellm_success_callback(s3)
    litellm.logging_callback_manager.add_litellm_success_callback(s3)

    assert litellm._async_success_callback == litellm.success_callback == [s3.v2_companion, s3]


@pytest.mark.asyncio
async def test_reinitialising_proxy_callbacks_keeps_one_writer_right_before_s3(
    bucket: ScriptedBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(ProxyLogging, "_add_proxy_hooks", lambda _self, _router=None: None)
    monkeypatch.setattr(litellm, "s3_callback_params", _params(bucket, strip=True))
    reader: Final = _Reader("reader")
    monkeypatch.setattr(litellm, "callbacks", [reader, "s3_v2"])
    proxy_logging: Final = ProxyLogging(user_api_key_cache=MagicMock())

    proxy_logging._init_litellm_callbacks(llm_router=None)
    proxy_logging._init_litellm_callbacks(llm_router=None)

    (s3,) = (entry for entry in litellm.callbacks if isinstance(entry, S3Logger))
    _ = s3.periodic_flush_task.cancel()
    lists: Final = (
        litellm.callbacks,
        litellm.success_callback,
        litellm.failure_callback,
        litellm._async_success_callback,
        litellm._async_failure_callback,
    )
    assert [[entry for entry in listed if entry is not proxy_logging.service_logging_obj] for listed in lists] == [
        [reader, s3.v2_companion, s3]
    ] * len(lists)


@pytest.mark.asyncio
async def test_removing_s3_removes_its_writer(bucket: ScriptedBucket) -> None:
    dropped: Final = _s3(bucket)
    removed: Final = _s3(bucket, region="eu-west-1")
    reader: Final = _Reader("reader")
    registered: Final = _registered(dropped, reader, removed)
    assert registered == [dropped.v2_companion, dropped, reader, removed.v2_companion, removed]

    _ = litellm.logging_callback_manager.drop_callbacks((dropped,))
    litellm.logging_callback_manager.remove_callback_from_all_lists(removed)

    assert litellm._async_success_callback == [reader]


@pytest.mark.asyncio
async def test_the_writer_manifest_builds_as_a_blocking_log_prepare_writer(bucket: ScriptedBucket) -> None:
    s3: Final = _s3(bucket)
    writer: Final = s3.v2_companion
    assert isinstance(writer, S3StripBase64Files)

    manifest: Final = build_manifest(writer)
    subscription: Final = manifest.subscriptions[Hook.log_prepare]

    assert manifest.selection.name == STRIP_CALLBACK_NAME
    assert manifest.selection.honors_no_log is False
    assert set(manifest.subscriptions) == {Hook.log_prepare}
    assert subscription.writes == frozenset({Field.standard_logging_payload})
    assert subscription.reads == frozenset({Field.deprecated_kwargs, Field.standard_logging_payload})
    assert build_manifest(s3).selection.honors_no_log is True
    assert [entry["hook"] for entry in json.loads(s3.sink.callback().manifest())["subscriptions"]] == ["attempt_ended"]
