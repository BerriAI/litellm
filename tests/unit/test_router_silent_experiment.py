import asyncio
import datetime
import json
import threading
import time
from collections.abc import Callable, Generator, Mapping
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router import Router
from litellm.router import _silent_experiment_kwargs_snapshot
from litellm.router import _silent_experiment_targets


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.success_kwargs: list[dict[str, object]] = []
        self.failure_kwargs: list[dict[str, object]] = []
        self._logged: Final = threading.Condition()

    def _record(self, outcome: Literal["success", "failure"], kwargs: dict[str, object]) -> None:
        with self._logged:
            match outcome:
                case "success":
                    self.success_kwargs.append(kwargs)
                case "failure":
                    self.failure_kwargs.append(kwargs)
            self._logged.notify_all()

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self._record("success", kwargs)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self._record("failure", kwargs)

    def wait_until(self, logged_enough: Callable[[], bool], timeout: float = 30.0) -> None:
        with self._logged:
            self._logged.wait_for(logged_enough, timeout=timeout)

    def judge_flags(self) -> dict[str, object]:
        return {
            call["litellm_params"]["metadata"]["judge_of"]: call["standard_logging_object"]["metadata"].get(
                "is_silent_experiment"
            )
            for call in self.success_kwargs
            if "judge_of" in call["litellm_params"]["metadata"]
        }

    def shadow_successes(self) -> list[dict[str, object]]:
        return [
            call
            for call in self.success_kwargs
            if call.get("litellm_params", {}).get("metadata", {}).get("is_silent_experiment") is True
        ]


class _SyncAndAsyncRecordingLogger(_RecordingLogger):
    def log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self._record("success", kwargs)


async def _settle_shared_logging_worker() -> None:
    try:
        await GLOBAL_LOGGING_WORKER.flush()
    finally:
        await GLOBAL_LOGGING_WORKER.stop()


class _JudgingLogger(_RecordingLogger):
    async def _judge(self, kwargs: Mapping[str, object]) -> None:
        metadata: Final = kwargs["litellm_params"]["metadata"]
        if "judge_of" in metadata:
            return
        await litellm.acompletion(
            model="openai/gpt-5.4-nano",
            api_key="fake-key",
            messages=[{"role": "user", "content": "score it"}],
            mock_response="ok",
            metadata={"judge_of": metadata["model_group"]},
        )

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        await super().async_log_success_event(kwargs, response_obj, start_time, end_time)
        await self._judge(kwargs)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        await self._judge(kwargs)


def _installed(logger: _RecordingLogger) -> Generator[_RecordingLogger]:
    settle_loop: Final = asyncio.new_event_loop()
    try:
        settle_loop.run_until_complete(_settle_shared_logging_worker())
    finally:
        settle_loop.close()
    original_callbacks: Final = litellm.callbacks
    litellm.callbacks = [logger]
    try:
        yield logger
    finally:
        litellm.callbacks = original_callbacks


@pytest.fixture
def recording_logger():
    yield from _installed(_RecordingLogger())


@pytest.fixture
def sync_and_async_recording_logger():
    yield from _installed(_SyncAndAsyncRecordingLogger())


@pytest.fixture
def judging_logger():
    yield from _installed(_JudgingLogger())


async def _wait_for_shadow_successes(logger: _RecordingLogger, expected: int, timeout: float = 30.0) -> None:
    await asyncio.to_thread(logger.wait_until, lambda: len(logger.shadow_successes()) >= expected, timeout)


async def _wait_for_all_successes(logger: _RecordingLogger, expected: int) -> None:
    await asyncio.to_thread(logger.wait_until, lambda: len(logger.success_kwargs) >= expected)


async def _wait_for_judges(logger: _RecordingLogger, judged: set[str]) -> None:
    await asyncio.to_thread(logger.wait_until, lambda: judged <= logger.judge_flags().keys())


def _silent_flag_by_group(logger: _RecordingLogger) -> dict[str, object]:
    return {
        call["standard_logging_object"]["model_group"]: call["standard_logging_object"]["metadata"].get(
            "is_silent_experiment"
        )
        for call in logger.success_kwargs
    }


def _wait_for_shadow_successes_sync(logger: _RecordingLogger, expected: int) -> None:
    logger.wait_until(lambda: len(logger.shadow_successes()) >= expected)


def _streaming_model_list(silent_model: object) -> list[dict[str, object]]:
    return [
        {
            "model_name": "primary-model",
            "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "fake-key", "silent_model": silent_model},
        },
        {
            "model_name": "shadow-a",
            "litellm_params": {"model": "openai/gpt-5.4-nano", "api_key": "fake-key", "silent_model": "shadow-b"},
        },
        {
            "model_name": "shadow-b",
            "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "fake-key"},
        },
    ]


class _NonCopyableSpan:
    """Mimics an OTel Span which raises on deepcopy, forcing safe_deep_copy
    to fall back to the original reference."""

    def __deepcopy__(self, memo):
        raise TypeError("OTel spans cannot be deepcopied")


class _FakeUserAPIKeyAuth:
    """Mimics UserAPIKeyAuth which contains a parent_otel_span that is not
    deepcopy-able. This is what actually causes safe_deep_copy to fail for
    the metadata dict in production — safe_deep_copy handles the top-level
    litellm_parent_otel_span specially (pops it before copying), but does
    NOT handle user_api_key_auth.parent_otel_span inside it."""

    def __init__(self, key_alias, parent_otel_span):
        self.key_alias = key_alias
        self.parent_otel_span = parent_otel_span

    def __deepcopy__(self, memo):
        raise TypeError("Contains OTel span that cannot be deepcopied")


def test_get_silent_experiment_kwargs():
    """
    Test _get_silent_experiment_kwargs returns isolated kwargs with silent experiment metadata.

    Uses a non-copyable user_api_key_auth (mimicking the real proxy scenario)
    so that safe_deep_copy falls back to the original metadata reference —
    exercising the identity-check fix path.
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    mock_span = _NonCopyableSpan()
    mock_auth = _FakeUserAPIKeyAuth(
        key_alias="HaneefKeyNonTeamProd",
        parent_otel_span=mock_span,
    )
    kwargs = {
        "metadata": {
            "foo": "bar",
            "litellm_parent_otel_span": mock_span,
            "user_api_key_auth": mock_auth,
        },
        "litellm_call_id": "call-123",
        "stream": True,
        "proxy_server_request": {"body": {"model": "test"}},
    }
    result = router._get_silent_experiment_kwargs(**kwargs)
    assert result["metadata"]["is_silent_experiment"] is True
    assert result["metadata"]["foo"] == "bar"
    assert "litellm_call_id" not in result
    assert result["stream"] is True
    # proxy_server_request must be preserved for spend log metadata
    assert "proxy_server_request" in result
    # CRITICAL: metadata must be a DIFFERENT dict object than the original,
    # so that setting model_group / is_silent_experiment on the silent dict
    # doesn't corrupt the primary call's metadata.
    assert result["metadata"] is not kwargs["metadata"]
    # OTel span must be stripped from the silent copy — it's not safe to use
    # across event loops (silent experiment runs in a new event loop).
    assert "litellm_parent_otel_span" not in result["metadata"]
    # Original metadata must NOT be mutated — must carry the real span,
    # not safe_deep_copy's temporary "placeholder" string.
    assert "is_silent_experiment" not in kwargs["metadata"]
    assert kwargs["metadata"]["litellm_parent_otel_span"] is mock_span
    assert kwargs["metadata"]["user_api_key_auth"] is mock_auth
    # Shallow copy must preserve user_api_key_auth so the silent experiment
    # can attribute billing / spend logs to the correct key/team.
    assert result["metadata"]["user_api_key_auth"] is mock_auth


def test_get_silent_experiment_kwargs_without_stream_stays_non_streaming():
    router = Router(model_list=[{"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k"}}])
    result = router._get_silent_experiment_kwargs(metadata={"foo": "bar"}, stream=False)
    assert result["stream"] is False
    assert "stream" not in router._get_silent_experiment_kwargs(metadata={"foo": "bar"})


@pytest.mark.parametrize(
    "silent_model, expected",
    [
        ("shadow-a", ("shadow-a",)),
        (["shadow-a", "shadow-b"], ("shadow-a", "shadow-b")),
        ([], ()),
        (None, ()),
        (42, ()),
        (["shadow-a", 42], ()),
    ],
)
def test_silent_experiment_targets(silent_model, expected):
    assert _silent_experiment_targets(silent_model) == expected


@pytest.mark.asyncio
async def test_streaming_shadow_is_streamed_and_drained_async(recording_logger):
    router = Router(model_list=_streaming_model_list("shadow-a"))
    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
        mock_response="pong",
        metadata={"foo": "bar"},
    )
    chunks = [chunk async for chunk in response]
    assert chunks
    await _wait_for_shadow_successes(recording_logger, expected=1)

    shadow_successes = recording_logger.shadow_successes()
    assert len(shadow_successes) == 1
    shadow = shadow_successes[0]
    assert shadow["stream"] is True
    assert shadow["stream_options"] == {"include_usage": True}
    assert shadow["litellm_params"]["metadata"]["model_group"] == "shadow-a"
    assert shadow["async_complete_streaming_response"] is not None


def test_streaming_shadow_is_streamed_and_drained_sync(recording_logger):
    router = Router(model_list=_streaming_model_list("shadow-a"))
    response = router.completion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        mock_response="pong",
        metadata={"foo": "bar"},
    )
    chunks = list(response)
    assert chunks
    _wait_for_shadow_successes_sync(recording_logger, expected=1)

    shadow_successes = recording_logger.shadow_successes()
    assert len(shadow_successes) == 1
    assert shadow_successes[0]["stream"] is True
    assert shadow_successes[0]["litellm_params"]["metadata"]["model_group"] == "shadow-a"
    assert shadow_successes[0]["async_complete_streaming_response"] is not None


@pytest.mark.asyncio
async def test_multiple_shadow_targets_fan_out_async(recording_logger):
    router = Router(model_list=_streaming_model_list(["shadow-a", "shadow-b"]))
    metadata = {"foo": "bar"}
    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        mock_response="pong",
        metadata=metadata,
    )
    assert [chunk async for chunk in response]
    await _wait_for_shadow_successes(recording_logger, expected=2)
    await _wait_for_all_successes(recording_logger, expected=3)

    shadow_successes = recording_logger.shadow_successes()
    model_groups = sorted(call["litellm_params"]["metadata"]["model_group"] for call in shadow_successes)
    assert model_groups == ["shadow-a", "shadow-b"]
    shadow_metadatas = [call["litellm_params"]["metadata"] for call in shadow_successes]
    assert shadow_metadatas[0] is not shadow_metadatas[1]
    assert all(call["stream"] is True for call in shadow_successes)
    assert "is_silent_experiment" not in metadata
    assert metadata.get("model_group") != "shadow-a"
    primary_successes = [call for call in recording_logger.success_kwargs if call not in shadow_successes]
    assert len(primary_successes) == 1
    assert primary_successes[0]["litellm_params"]["metadata"]["model_group"] == "primary-model"


@pytest.mark.asyncio
async def test_standard_logging_payload_marks_only_the_mirrored_calls_as_silent_experiments(
    recording_logger: _RecordingLogger,
) -> None:
    router: Final = Router(model_list=_streaming_model_list(["shadow-a", "shadow-b"]))
    response: Final = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong",
    )
    assert response.choices[0].message.content == "pong"
    await _wait_for_all_successes(recording_logger, expected=3)

    payloads: Final = [call["standard_logging_object"] for call in recording_logger.success_kwargs]
    assert sorted(payload["model_group"] for payload in payloads) == ["primary-model", "shadow-a", "shadow-b"]
    metadata_by_group: Final = {payload["model_group"]: payload["metadata"] for payload in payloads}
    assert metadata_by_group["shadow-a"]["is_silent_experiment"] is True
    assert metadata_by_group["shadow-b"]["is_silent_experiment"] is True
    assert "is_silent_experiment" not in metadata_by_group["primary-model"]


def test_standard_logging_payload_marks_only_the_mirrored_calls_on_the_sync_path(
    sync_and_async_recording_logger: _SyncAndAsyncRecordingLogger,
) -> None:
    router: Final = Router(model_list=_streaming_model_list(["shadow-a", "shadow-b"]))
    response: Final = router.completion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong",
    )
    assert response.choices[0].message.content == "pong"
    sync_and_async_recording_logger.wait_until(
        lambda: (
            _silent_flag_by_group(sync_and_async_recording_logger).keys() >= {"primary-model", "shadow-a", "shadow-b"}
        )
    )

    assert _silent_flag_by_group(sync_and_async_recording_logger) == {
        "primary-model": None,
        "shadow-a": True,
        "shadow-b": True,
    }


@pytest.mark.asyncio
async def test_a_failed_mirrored_call_reaches_failure_callbacks_marked_as_mirrored(
    recording_logger: _RecordingLogger,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "fake-key",
                    "silent_model": "shadow-a",
                    "mock_response": "pong",
                },
            },
            {
                "model_name": "shadow-a",
                "litellm_params": {
                    "model": "openai/gpt-5.4-nano",
                    "api_key": "fake-key",
                    "mock_response": "litellm.InternalServerError",
                },
            },
        ]
    )
    await router.acompletion(model="primary-model", messages=[{"role": "user", "content": "hi"}])
    await asyncio.to_thread(recording_logger.wait_until, lambda: len(recording_logger.failure_kwargs) >= 1)

    assert {
        (
            call["standard_logging_object"]["model_group"],
            call["standard_logging_object"]["metadata"].get("is_silent_experiment"),
        )
        for call in recording_logger.failure_kwargs
    } == {("shadow-a", True)}


_SPEND_ROW_TIME: Final = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _spend_row_is_silent_experiment(call: Mapping[str, object]) -> object:
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    row: Final = get_logging_payload(
        kwargs=dict(call),
        response_obj=litellm.ModelResponse(id="chatcmpl-test", choices=[]),
        start_time=_SPEND_ROW_TIME,
        end_time=_SPEND_ROW_TIME,
    )
    return json.loads(row["metadata"]).get("is_silent_experiment")


def _through_the_spend_collector(call: Mapping[str, object]) -> Mapping[str, object]:
    from litellm.proxy.spend_tracking.spend_event import (
        SpendEvent,
        build_spend_event,
        decode_spend_event,
        spend_event_callback_args,
    )

    line: Final = build_spend_event(
        call,
        litellm.ModelResponse(id="chatcmpl-test", choices=[]),
        _SPEND_ROW_TIME,
        _SPEND_ROW_TIME,
        store_bodies=False,
    )
    assert isinstance(line, bytes)
    event: Final = decode_spend_event(line)
    assert isinstance(event, SpendEvent)
    return spend_event_callback_args(event).kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spend_kwargs",
    [pytest.param(lambda call: call, id="in_process"), pytest.param(_through_the_spend_collector, id="collector")],
)
async def test_spend_rows_mark_only_the_mirrored_calls_when_the_caller_sends_litellm_metadata(
    recording_logger: _RecordingLogger,
    spend_kwargs: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> None:
    router: Final = Router(model_list=_streaming_model_list(["shadow-a", "shadow-b"]))
    await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong",
        litellm_metadata={"user_api_key": "test-key"},
    )
    await _wait_for_all_successes(recording_logger, expected=3)

    silent_by_group: Final = {
        call["standard_logging_object"]["model_group"]: _spend_row_is_silent_experiment(spend_kwargs(call))
        for call in recording_logger.success_kwargs
    }
    assert silent_by_group == {"primary-model": None, "shadow-a": True, "shadow-b": True}


@pytest.mark.asyncio
async def test_router_counts_a_mirrored_call_once_against_its_target_rate_limits(
    recording_logger: _RecordingLogger,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "fake-key", "silent_model": "shadow-b"},
            },
            {
                "model_name": "shadow-b",
                "litellm_params": {"model": "openai/gpt-5.4-nano", "api_key": "fake-key", "rpm": 100, "tpm": 100000},
            },
        ]
    )
    await router.acompletion(model="primary-model", messages=[{"role": "user", "content": "hi"}], mock_response="pong")
    await _wait_for_shadow_successes(recording_logger, expected=1)
    await GLOBAL_LOGGING_WORKER.flush()

    shadow_tokens: Final = recording_logger.shadow_successes()[0]["standard_logging_object"]["total_tokens"]
    assert await router.get_model_group_usage("shadow-b") == (shadow_tokens, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bucket", "planted"),
    [
        ("metadata", {"is_silent_experiment": True}),
        ("litellm_metadata", {"is_silent_experiment": True}),
    ],
)
async def test_a_caller_sent_silent_experiment_flag_does_not_mark_a_real_call_as_mirrored(
    recording_logger: _RecordingLogger, bucket: str, planted: dict[str, object]
) -> None:
    router: Final = Router(model_list=_streaming_model_list(["shadow-a", "shadow-b"]))
    await router.acompletion(
        model="shadow-b",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong",
        **{bucket: planted},
    )
    await _wait_for_all_successes(recording_logger, expected=1)

    call: Final = recording_logger.success_kwargs[0]
    assert "is_silent_experiment" not in call["standard_logging_object"]["metadata"]
    assert _spend_row_is_silent_experiment(call) is None


def test_multiple_shadow_targets_fan_out_sync(recording_logger):
    router = Router(model_list=_streaming_model_list(["shadow-a", "shadow-b"]))
    response = router.completion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong",
        metadata={"foo": "bar"},
    )
    assert response.choices[0].message.content == "pong"
    _wait_for_shadow_successes_sync(recording_logger, expected=2)

    shadow_successes = recording_logger.shadow_successes()
    model_groups = sorted(call["litellm_params"]["metadata"]["model_group"] for call in shadow_successes)
    assert model_groups == ["shadow-a", "shadow-b"]
    assert all(call["stream"] is False for call in shadow_successes)


def _tagged_primary_model_list() -> list[dict[str, object]]:
    return [
        {
            "model_name": "primary-model",
            "litellm_params": {
                "model": "openai/gpt-5.4-mini",
                "api_key": "fake-key",
                "silent_model": "shadow-b",
                "tags": ["primary-only"],
            },
        },
        {
            "model_name": "shadow-b",
            "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "fake-key"},
        },
    ]


def test_silent_experiment_kwargs_snapshot_is_isolated_from_later_primary_mutations():
    metadata = {"foo": "bar"}
    kwargs: dict[str, object] = {"metadata": metadata, "stream": True}
    snapshot = _silent_experiment_kwargs_snapshot(kwargs)
    kwargs["messages"] = [{"role": "user", "content": "added by the primary"}]
    metadata["tags"] = ["primary-only"]

    assert dict(snapshot) == {"metadata": {"foo": "bar"}, "stream": True}
    assert dict(_silent_experiment_kwargs_snapshot({"stream": False, "metadata": None})) == {
        "stream": False,
        "metadata": None,
    }


def test_sync_shadow_gets_kwargs_snapshot_taken_before_primary_mutates_them(recording_logger):
    deferred: list[Callable[[], None]] = []

    class _DeferredThread:
        def __init__(self, target, args, kwargs, daemon) -> None:
            deferred.append(lambda: target(*args, **kwargs))

        def start(self) -> None:
            return None

    router = Router(model_list=_tagged_primary_model_list())
    with patch(  # test-quality-ok: Router has no thread factory to inject; deferring start is the only deterministic way to expose the race
        "litellm.router.threading", SimpleNamespace(Thread=_DeferredThread)
    ):
        response = router.completion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="pong",
            metadata={"foo": "bar"},
        )
    assert response.choices[0].message.content == "pong"
    assert len(deferred) == 1
    deferred[0]()
    _wait_for_shadow_successes_sync(recording_logger, expected=1)

    shadow_successes = recording_logger.shadow_successes()
    assert len(shadow_successes) == 1
    shadow_metadata = shadow_successes[0]["litellm_params"]["metadata"]
    assert shadow_metadata["model_group"] == "shadow-b"
    assert "primary-only" not in shadow_metadata.get("tags", [])


def test_sync_shadow_workers_do_not_share_metadata_with_each_other(recording_logger):
    workers: list[tuple[Mapping[str, object], Callable[[], None]]] = []

    class _DeferredThread:
        def __init__(self, target, args, kwargs, daemon) -> None:
            workers.append((kwargs, lambda: target(*args, **kwargs)))

        def start(self) -> None:
            return None

    router = Router(model_list=_streaming_model_list(["shadow-a", "shadow-b"]))
    with patch(  # test-quality-ok: Router has no thread factory to inject; deferring start is the only deterministic way to expose the race
        "litellm.router.threading", SimpleNamespace(Thread=_DeferredThread)
    ):
        router.completion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="pong",
            metadata={"foo": "bar"},
        )
    assert len(workers) == 2
    (first_kwargs, run_first), (_, run_second) = workers
    first_kwargs["metadata"].pop("foo")
    run_second()
    run_first()
    _wait_for_shadow_successes_sync(recording_logger, expected=2)

    metadata_by_group = {
        call["litellm_params"]["metadata"]["model_group"]: call["litellm_params"]["metadata"]
        for call in recording_logger.shadow_successes()
    }
    assert metadata_by_group["shadow-b"]["foo"] == "bar"
    assert "foo" not in metadata_by_group["shadow-a"]


@pytest.mark.asyncio
async def test_async_shadow_does_not_inherit_primary_deployment_tags(recording_logger):
    router = Router(model_list=_tagged_primary_model_list())
    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong",
        metadata={"foo": "bar"},
    )
    assert response.choices[0].message.content == "pong"
    await _wait_for_shadow_successes(recording_logger, expected=1)

    shadow_successes = recording_logger.shadow_successes()
    assert len(shadow_successes) == 1
    assert "primary-only" not in shadow_successes[0]["litellm_params"]["metadata"].get("tags", [])


@pytest.mark.asyncio
async def test_shadow_of_a_shadow_is_not_launched(recording_logger):
    router = Router(model_list=_streaming_model_list(["shadow-a"]))
    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong",
    )
    assert response.choices[0].message.content == "pong"
    await _wait_for_shadow_successes(recording_logger, expected=2, timeout=1.0)

    model_groups = [call["litellm_params"]["metadata"]["model_group"] for call in recording_logger.shadow_successes()]
    assert model_groups == ["shadow-a"]


@pytest.mark.asyncio
async def test_run_silent_experiment_drains_stream_so_callbacks_fire(recording_logger):
    router = Router(model_list=_streaming_model_list(None))
    silent_kwargs: Final = {
        "stream": True,
        "stream_options": {"include_usage": True},
        "mock_response": "pong",
        "metadata": {"is_silent_experiment": True, "model_group": "shadow-b"},
    }
    await router._run_silent_experiment("shadow-b", [{"role": "user", "content": "hi"}], silent_kwargs)
    await _wait_for_shadow_successes(recording_logger, expected=1)

    shadow_successes = recording_logger.shadow_successes()
    assert len(shadow_successes) == 1
    assert shadow_successes[0]["stream"] is True
    assert shadow_successes[0]["async_complete_streaming_response"] is not None
    assert silent_kwargs["stream"] is True


@pytest.mark.asyncio
async def test_router_silent_experiment_acompletion():
    """
    Test that silent_model triggers a background acompletion call
    and that the silent_model parameter is stripped from both calls.
    """
    model_list = [
        {
            "model_name": "primary-model",
            "litellm_params": {
                "model": "openai/gpt-3.5-turbo",
                "api_key": "fake-key",
                "silent_model": "silent-model",
            },
        },
        {
            "model_name": "silent-model",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "fake-key",
            },
        },
    ]

    router = Router(model_list=model_list)

    # Use AsyncMock for async function mocking
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])
    mock_acompletion = AsyncMock(return_value=mock_response)

    # Patch at the litellm.router module level where it's imported and used
    with patch.object(litellm, "acompletion", mock_acompletion):
        response = await router.acompletion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert response.choices[0].message.content == "hello"

        # Give the background task a moment to trigger (it's an asyncio task)
        await asyncio.sleep(0.1)

        # Should have 2 calls: one for primary, one for silent
        assert mock_acompletion.call_count == 2

        # Check call arguments
        call_args_list = mock_acompletion.call_args_list

        # Verify no silent_model in any call to litellm.acompletion
        for call in call_args_list:
            args, kwargs = call
            assert "silent_model" not in kwargs
            if "metadata" in kwargs:
                # One call should have is_silent_experiment=True
                pass

        # Find the silent call
        silent_call = next(
            (
                c
                for c in call_args_list
                if c[1].get("metadata", {}).get("is_silent_experiment") is True
            ),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"

        # Find the primary call
        primary_call = next(
            (
                c
                for c in call_args_list
                if not c[1].get("metadata", {}).get("is_silent_experiment")
            ),
            None,
        )
        assert primary_call is not None
        assert primary_call[1]["model"] == "openai/gpt-3.5-turbo"


def test_router_silent_experiment_completion():
    """
    Test that silent_model triggers a background completion call (sync)
    and that the silent_model parameter is stripped.
    """
    model_list = [
        {
            "model_name": "primary-model",
            "litellm_params": {
                "model": "openai/gpt-3.5-turbo",
                "api_key": "fake-key",
                "silent_model": "silent-model",
            },
        },
        {
            "model_name": "silent-model",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "fake-key",
            },
        },
    ]

    router = Router(model_list=model_list)

    # Mock litellm.acompletion
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])

    # We need an async mock for acompletion
    async def mock_acompletion(*args, **kwargs):
        return mock_response

    mock_acompletion_mock = AsyncMock(side_effect=mock_acompletion)
    mock_completion_mock = MagicMock(return_value=mock_response)

    # Patch at the litellm module level
    with (
        patch.object(litellm, "acompletion", mock_acompletion_mock),
        patch.object(litellm, "completion", mock_completion_mock),
    ):
        response = router.completion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert response.choices[0].message.content == "hello"

        # The sync background call uses a thread pool. We might need to wait.
        time.sleep(2.0)

        # Should have 1 acompletion call (the silent background call)
        assert mock_acompletion_mock.call_count == 1

        call_args_list = mock_acompletion_mock.call_args_list

        # Verify no silent_model in any call
        for call in call_args_list:
            args, kwargs = call
            assert "silent_model" not in kwargs

        # Find the silent call
        silent_call = next(
            (
                c
                for c in call_args_list
                if c[1].get("metadata", {}).get("is_silent_experiment") is True
            ),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"
        # Verify model_group is set to the silent model name for correct metric attribution
        assert silent_call[1]["metadata"]["model_group"] == "silent-model"


SILENT_EXPERIMENT_RUNNERS: Final = (
    pytest.param(lambda router, **kwargs: router._silent_experiment_completion(**kwargs), id="sync"),
    pytest.param(lambda router, **kwargs: asyncio.run(router._silent_experiment_acompletion(**kwargs)), id="async"),
)


@pytest.mark.parametrize("run_silent_experiment", SILENT_EXPERIMENT_RUNNERS)
def test_silent_experiment_sends_shadow_request_attributed_to_the_silent_model(run_silent_experiment):
    router = Router(model_list=_streaming_model_list(["shadow-a"]))
    primary_metadata: Final = {"model_group": "primary-model"}
    with patch.object(router, "acompletion", new_callable=AsyncMock, return_value=None) as acompletion:
        run_silent_experiment(
            router,
            silent_model="shadow-a",
            messages=[{"role": "user", "content": "hi"}],
            metadata=primary_metadata,
        )

    acompletion.assert_awaited_once()
    shadow_call: Final = acompletion.await_args.kwargs
    assert shadow_call["model"] == "shadow-a"
    assert shadow_call["messages"] == [{"role": "user", "content": "hi"}]
    assert shadow_call["metadata"]["model_group"] == "shadow-a"
    assert shadow_call["metadata"]["is_silent_experiment"] is True
    assert primary_metadata == {"model_group": "primary-model"}




_EMBEDDING_API_BASE: Final = "https://embeddings.example.test/v1"


def _strict_embedding_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{_EMBEDDING_API_BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "embed-model",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )


def _embedding_router_with_silent_model() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "embed-primary",
                "litellm_params": {
                    "model": "openai/embed-model",
                    "api_base": _EMBEDDING_API_BASE,
                    "api_key": "fake-key",
                    "silent_model": "embed-shadow",
                },
            }
        ]
    )


def test_embedding_with_silent_model_sends_provider_body_without_it(respx_mock: respx.MockRouter) -> None:
    route: Final = _strict_embedding_route(respx_mock)

    response: Final = _embedding_router_with_silent_model().embedding(
        model="embed-primary", input=["black dresses"], input_type="query"
    )

    request_body: Final = json.loads(route.calls.last.request.read())
    assert request_body == {"model": "embed-model", "input": ["black dresses"], "input_type": "query"}
    assert response.data[0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_aembedding_with_silent_model_sends_provider_body_without_it(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = _strict_embedding_route(respx_mock)

    response: Final = await _embedding_router_with_silent_model().aembedding(
        model="embed-primary", input=["black dresses"], input_type="query"
    )

    request_body: Final = json.loads(route.calls.last.request.read())
    assert request_body == {"model": "embed-model", "input": ["black dresses"], "input_type": "query"}
    assert response.data[0]["embedding"] == [0.1, 0.2]


@pytest.mark.parametrize("run_silent_experiment", SILENT_EXPERIMENT_RUNNERS)
def test_silent_experiment_does_not_launch_from_a_shadow_request(run_silent_experiment):
    router = Router(model_list=_streaming_model_list(["shadow-a"]))
    with patch.object(router, "acompletion", new_callable=AsyncMock, return_value=None) as acompletion:
        run_silent_experiment(
            router,
            silent_model="shadow-a",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"is_silent_experiment": True},
        )

    acompletion.assert_not_awaited()


def _judged_router(shadow_outcome: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "fake-key",
                    "silent_model": "shadow-a",
                    "mock_response": "pong",
                },
            },
            {
                "model_name": "shadow-a",
                "litellm_params": {
                    "model": "openai/gpt-5.4-nano",
                    "api_key": "fake-key",
                    "mock_response": shadow_outcome,
                },
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shadow_outcome", ["pong", "litellm.InternalServerError"])
async def test_calls_a_callback_starts_while_logging_a_mirrored_call_are_not_marked_as_mirrored(
    judging_logger: _JudgingLogger, shadow_outcome: str
) -> None:
    router: Final = _judged_router(shadow_outcome)
    await router.acompletion(model="primary-model", messages=[{"role": "user", "content": "hi"}])
    await _wait_for_judges(judging_logger, {"primary-model", "shadow-a"})

    assert judging_logger.judge_flags() == {"primary-model": None, "shadow-a": None}


def _start_judge_call(
    kwargs: Mapping[str, object], completion_response: object, start_time: object, end_time: object
) -> None:
    litellm.completion(
        model="openai/gpt-5.4-nano",
        api_key="fake-key",
        messages=[{"role": "user", "content": "score it"}],
        mock_response="ok",
        metadata={"judge_of": kwargs["litellm_params"]["metadata"]["model_group"]},
    )


@pytest.mark.asyncio
async def test_calls_a_failure_function_callback_starts_for_a_mirrored_call_are_not_marked_as_mirrored(
    sync_and_async_recording_logger: _SyncAndAsyncRecordingLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "failure_callback", [_start_judge_call])
    router: Final = _judged_router("litellm.InternalServerError")
    await router.acompletion(model="primary-model", messages=[{"role": "user", "content": "hi"}])
    await _wait_for_judges(sync_and_async_recording_logger, {"shadow-a"})

    assert sync_and_async_recording_logger.judge_flags() == {"shadow-a": None}
