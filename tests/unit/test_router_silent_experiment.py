import asyncio
import time
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router
from litellm.router import _silent_experiment_kwargs_snapshot
from litellm.router import _silent_experiment_targets


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.success_kwargs: list[dict[str, object]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.success_kwargs.append(kwargs)

    def shadow_successes(self) -> list[dict[str, object]]:
        return [
            call
            for call in self.success_kwargs
            if call.get("litellm_params", {}).get("metadata", {}).get("is_silent_experiment") is True
        ]


@pytest.fixture
def recording_logger():
    original_callbacks: Final = litellm.callbacks
    logger: Final = _RecordingLogger()
    litellm.callbacks = [logger]
    try:
        yield logger
    finally:
        litellm.callbacks = original_callbacks


async def _wait_for_shadow_successes(logger: _RecordingLogger, expected: int, timeout: float = 5.0) -> None:
    deadline: Final = time.monotonic() + timeout
    while len(logger.shadow_successes()) < expected and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


def _wait_for_shadow_successes_sync(logger: _RecordingLogger, expected: int, timeout: float = 5.0) -> None:
    deadline: Final = time.monotonic() + timeout
    while len(logger.shadow_successes()) < expected and time.monotonic() < deadline:
        time.sleep(0.05)


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


def test_silent_experiment_completion_direct():
    """
    Test _silent_experiment_completion directly (for router code coverage).
    Mocks router.completion to avoid real API call.
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "hi"}]
    with patch.object(router, "acompletion", new_callable=AsyncMock, return_value=None):
        router._silent_experiment_completion(
            silent_model="gpt-3.5-turbo",
            messages=messages,
        )


@pytest.mark.asyncio
async def test_silent_experiment_acompletion_direct():
    """
    Test _silent_experiment_acompletion directly (for router code coverage).
    Mocks router.acompletion to avoid real API call.
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "hi"}]
    with patch.object(router, "acompletion", new_callable=AsyncMock, return_value=None):
        await router._silent_experiment_acompletion(
            silent_model="gpt-3.5-turbo",
            messages=messages,
        )


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
