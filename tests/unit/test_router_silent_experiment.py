import asyncio
import datetime
import json
import threading
import time
from collections.abc import Callable, Generator, Mapping
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router import Router


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
            (c for c in call_args_list if c[1].get("metadata", {}).get("is_silent_experiment") is True),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"

        # Find the primary call
        primary_call = next(
            (c for c in call_args_list if not c[1].get("metadata", {}).get("is_silent_experiment")),
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
            (c for c in call_args_list if c[1].get("metadata", {}).get("is_silent_experiment") is True),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"
        # Verify model_group is set to the silent model name for correct metric attribution
        assert silent_call[1]["metadata"]["model_group"] == "silent-model"


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


def _generic_shadow_reply(surface: str, text: str) -> dict[str, object]:
    if surface == "responses":
        return {
            "id": f"resp_{text}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "id": "message-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    return {
        "id": f"msg_{text}",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-haiku-latest",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.parametrize("retry", [False, True])
async def test_generic_silent_model_mirrors_once_across_retries(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, surface: str, retry: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    shadow_seen: Final = asyncio.Event()

    def shadow_response(request: httpx.Request) -> httpx.Response:
        shadow_seen.set()
        return httpx.Response(200, json=_generic_shadow_reply(surface, "shadow"))

    path: Final = "/v1/responses" if surface == "responses" else "/v1/messages"
    provider_model: Final = "openai/gpt-4o-mini" if surface == "responses" else "anthropic/claude-3-5-haiku-latest"
    primary_response: Final = httpx.Response(200, json=_generic_shadow_reply(surface, "primary"))
    failures: Final = (httpx.Response(500, json={"error": {"message": "retry", "type": "server_error"}}),)
    primary: Final = respx_mock.post(f"https://primary.example.test{path}").mock(
        side_effect=(*failures, primary_response) if retry else (primary_response,)
    )
    shadow: Final = respx_mock.post(f"https://shadow.example.test{path}").mock(side_effect=shadow_response)
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": provider_model,
                    "api_base": "https://primary.example.test/v1"
                    if surface == "responses"
                    else "https://primary.example.test",
                    "api_key": "fake-key",
                    "silent_model": "shadow",
                },
            },
            {
                "model_name": "shadow",
                "litellm_params": {
                    "model": provider_model,
                    "api_base": "https://shadow.example.test/v1"
                    if surface == "responses"
                    else "https://shadow.example.test",
                    "api_key": "fake-key",
                },
            },
        ],
        num_retries=1 if retry else 0,
        retry_after=0,
        disable_cooldowns=True,
    )
    if surface == "responses":
        response: Final = await router.aresponses(model="primary", input="hello", metadata={"user_id": "test-user"})
    else:
        response: Final = await router.aanthropic_messages(
            model="primary", messages=[{"role": "user", "content": "hello"}], max_tokens=8
        )
    await asyncio.wait_for(shadow_seen.wait(), timeout=15)

    assert "primary" in str(response)
    assert primary.call_count == (2 if retry else 1)
    assert shadow.call_count == 1
    assert "silent_model" not in json.loads(primary.calls.last.request.content)
    assert "silent_model" not in json.loads(shadow.calls.last.request.content)
    if surface == "responses":
        assert json.loads(shadow.calls.last.request.content)["metadata"] == {"user_id": "test-user"}


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["responses", "messages"])
async def test_generic_silent_model_mirrors_once_across_fallbacks(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    shadow_seen: Final = asyncio.Event()

    def shadow_response(request: httpx.Request) -> httpx.Response:
        shadow_seen.set()
        return httpx.Response(200, json=_generic_shadow_reply(surface, "shadow"))

    path: Final = "/v1/responses" if surface == "responses" else "/v1/messages"
    model: Final = "openai/gpt-4o-mini" if surface == "responses" else "anthropic/claude-3-5-haiku-latest"
    primary: Final = respx_mock.post(f"https://primary.example.test{path}").mock(
        return_value=httpx.Response(500, json={"error": {"message": "try fallback", "type": "server_error"}})
    )
    fallback: Final = respx_mock.post(f"https://fallback.example.test{path}").mock(
        return_value=httpx.Response(200, json=_generic_shadow_reply(surface, "fallback"))
    )
    shadow: Final = respx_mock.post(f"https://shadow.example.test{path}").mock(side_effect=shadow_response)
    suffix: Final = "/v1" if surface == "responses" else ""
    router: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": model,
                    "api_base": f"https://{name}.example.test{suffix}",
                    "api_key": "fake-key",
                    "silent_model": "shadow",
                },
            }
            for name in ("primary", "fallback")
        ]
        + [
            {
                "model_name": "shadow",
                "litellm_params": {
                    "model": model,
                    "api_base": f"https://shadow.example.test{suffix}",
                    "api_key": "fake-key",
                },
            }
        ],
        num_retries=0,
        disable_cooldowns=True,
        fallbacks=[{"primary": ["fallback"]}],
    )
    if surface == "responses":
        response: Final = await router.aresponses(model="primary", input="hello")
    else:
        response: Final = await router.aanthropic_messages(
            model="primary", messages=[{"role": "user", "content": "hello"}], max_tokens=8
        )
    await asyncio.wait_for(shadow_seen.wait(), timeout=15)

    assert "fallback" in str(response)
    assert primary.call_count >= 1
    assert fallback.call_count == 1
    assert shadow.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["request", "primary", "shadow"])
@pytest.mark.parametrize(
    "mcp_config",
    [
        {"tools": [{"type": "mcp", "server_label": "sample", "server_url": "https://mcp.example.test"}]},
        {"mcp_servers": [{"type": "url", "name": "sample", "url": "https://mcp.example.test"}]},
    ],
    ids=["tools", "servers"],
)
async def test_generic_silent_model_does_not_run_mcp(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    mcp_config: dict[str, object],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    primary: Final = respx_mock.post("https://primary.example.test/v1/responses").mock(
        return_value=httpx.Response(200, json=_generic_shadow_reply("responses", "primary"))
    )
    shadow: Final = respx_mock.post("https://shadow.example.test/v1/responses").mock(
        return_value=httpx.Response(200, json=_generic_shadow_reply("responses", "shadow"))
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"https://{name}.example.test/v1",
                    "api_key": "fake-key",
                    **({"silent_model": "shadow"} if name == "primary" else {}),
                    **(mcp_config if location == name else {}),
                },
            }
            for name in ("primary", "shadow")
        ],
        num_retries=0,
        disable_cooldowns=True,
    )
    response: Final = await router.aresponses(
        model="primary", input="hello", **(mcp_config if location == "request" else {})
    )
    pending: Final = tuple(
        task for task in asyncio.all_tasks() if task.get_coro().__qualname__.endswith("_run_generic_silent")
    )
    await asyncio.gather(*pending)

    assert "primary" in str(response)
    assert primary.call_count == 1
    assert shadow.call_count == 0


@pytest.mark.asyncio
async def test_generic_silent_fanout_preserves_request_and_logging_metadata(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, recording_logger: _RecordingLogger
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    endpoints: Final = ("primary", "shadow-a", "shadow-b")
    routes: Final = {
        name: respx_mock.post(f"https://{name}.example.test/v1/responses").mock(
            return_value=httpx.Response(200, json=_generic_shadow_reply("responses", name))
        )
        for name in endpoints
    }
    router: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"https://{name}.example.test/v1",
                    "api_key": "fake-key",
                    "silent_model": ["shadow-a", "shadow-b"] if name == "primary" else "primary",
                },
            }
            for name in endpoints
        ],
        num_retries=0,
        disable_cooldowns=True,
    )
    request_metadata: Final = {"user_id": "test-user"}
    router_metadata: Final = {"user_api_key": "test-key"}
    response: Final = await router.aresponses(
        model="primary", input="hello", metadata=request_metadata, litellm_metadata=router_metadata
    )
    await _wait_for_all_successes(recording_logger, expected=3)

    assert "primary" in str(response)
    assert all(route.call_count == 1 for route in routes.values())
    assert all(
        json.loads(route.calls.last.request.content)["metadata"] == {"user_id": "test-user"}
        for route in routes.values()
    )
    assert request_metadata == {"user_id": "test-user"}
    assert router_metadata["user_api_key"] == "test-key"
    assert "is_silent_experiment" not in router_metadata
    assert _silent_flag_by_group(recording_logger) == {"primary": None, "shadow-a": True, "shadow-b": True}


@pytest.mark.asyncio
async def test_streaming_responses_are_mirrored_as_non_streaming_requests(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    completed: Final = _generic_shadow_reply("responses", "primary")
    events: Final = (
        {"type": "response.created", "sequence_number": 0, "response": {**completed, "status": "in_progress"}},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "message-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "primary",
        },
        {"type": "response.completed", "sequence_number": 2, "response": completed},
    )
    stream: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    primary: Final = respx_mock.post("https://primary.example.test/v1/responses").mock(
        return_value=httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})
    )
    shadow: Final = respx_mock.post("https://shadow.example.test/v1/responses").mock(
        return_value=httpx.Response(200, json=_generic_shadow_reply("responses", "shadow"))
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"https://{name}.example.test/v1",
                    "api_key": "fake-key",
                    **({"silent_model": "shadow"} if name == "primary" else {}),
                },
            }
            for name in ("primary", "shadow")
        ],
        num_retries=0,
        disable_cooldowns=True,
    )
    response: Final = await router.aresponses(model="primary", input="hello", stream=True)
    observed: Final = [item async for item in response]
    pending: Final = tuple(
        task for task in asyncio.all_tasks() if task.get_coro().__qualname__.endswith("_run_generic_silent")
    )
    await asyncio.gather(*pending)

    assert any(getattr(item, "type", "") == "response.completed" for item in observed)
    assert primary.call_count == 1
    assert shadow.call_count == 1
    assert json.loads(primary.calls.last.request.content)["stream"] is True
    assert not json.loads(shadow.calls.last.request.content).get("stream", False)
