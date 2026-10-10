import concurrent.futures
import json
from collections.abc import Iterator
from threading import Event
from typing import Final, Literal

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel

import litellm
from litellm.batch_completion.main import batch_completion_models, batch_completion_models_all_responses
from litellm.exceptions import MidStreamFallbackError


class _BatchRequest(BaseModel):
    model: str


@pytest.mark.parametrize("mode", ["models", "deployments", "same_model_deployments"])
@pytest.mark.parametrize("scenario", ["slow_first", "failed_first", "all_failed"])
def test_batch_completion_models_returns_first_success(
    mode: Literal["models", "deployments", "same_model_deployments"],
    scenario: Literal["slow_first", "failed_first", "all_failed"],
) -> None:
    first_started: Final = Event()
    release_first: Final = Event()
    first_finished: Final = Event()

    def handle(request: httpx.Request) -> httpx.Response:
        model: Final = request.headers.get("x-test-response", _BatchRequest.model_validate_json(request.content).model)
        if model == "first":
            first_started.set()
            try:
                if scenario == "slow_first":
                    assert release_first.wait(timeout=20), "First request was never released"
            finally:
                first_finished.set()
        else:
            assert first_started.wait(timeout=20), "First request never started"
        if scenario == "all_failed" or (model == "first" and scenario == "failed_first"):
            return httpx.Response(500, json={"error": {"message": f"{model} unavailable", "type": "server_error"}})
        return httpx.Response(
            200,
            json={
                "id": "batch-test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": model}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with (
        OpenAI(
            api_key="test-key",
            base_url="https://batch.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as caller,
    ):

        def run_batch() -> object:
            if mode == "models":
                return batch_completion_models(
                    models=["openai/first", "openai/second"],
                    messages=[{"role": "user", "content": "hello"}],
                    client=client,
                    num_retries=0,
                )
            deployments: Final = (
                [
                    {"model": "openai/shared", "extra_headers": {"x-test-response": "second"}},
                    {"model": "openai/shared", "extra_headers": {"x-test-response": "first"}},
                ]
                if mode == "same_model_deployments"
                else [{"model": "openai/first"}, {"model": "openai/second"}]
            )
            return batch_completion_models(
                deployments=deployments,
                model_list=[],
                messages=[{"role": "user", "content": "hello"}],
                client=client,
                num_retries=0,
            )

        pending: Final = caller.submit(run_batch)
        try:
            if scenario == "all_failed" and mode == "models":
                with pytest.raises(litellm.InternalServerError, match="first unavailable"):
                    pending.result(timeout=10)
                return
            response: Final = pending.result(timeout=10)
            if scenario == "all_failed":
                assert response is None
            else:
                assert isinstance(response, litellm.ModelResponse)
                assert response.model == "second"
                assert not release_first.is_set()
        finally:
            release_first.set()
            assert first_finished.wait(timeout=20), "First request did not finish"


@pytest.mark.parametrize("mode", ["models", "deployments"])
@pytest.mark.parametrize("stream_error", [False, True])
def test_batch_completion_models_returns_unconsumed_stream(
    mode: Literal["models", "deployments"], stream_error: bool
) -> None:
    slow_started: Final = Event()
    release_slow: Final = Event()
    slow_finished: Final = Event()
    release_tokens: Final = Event()
    tokens_requested: Final = Event()

    class ProviderStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            tokens_requested.set()
            assert release_tokens.wait(timeout=20), "Stream tokens were never released"
            payload: Final = (
                {"error": {"message": "stream failed", "type": "server_error", "code": "server_error"}}
                if stream_error
                else {
                    "id": "batch-stream-test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "fast",
                    "choices": [{"index": 0, "delta": {"content": "winner"}, "finish_reason": "stop"}],
                }
            )
            yield f"data: {json.dumps(payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    def handle(request: httpx.Request) -> httpx.Response:
        model: Final = _BatchRequest.model_validate_json(request.content).model
        if model == "slow":
            slow_started.set()
            try:
                assert release_slow.wait(timeout=20), "Slow request was never released"
                return httpx.Response(500, json={"error": {"message": "slow failed", "type": "server_error"}})
            finally:
                slow_finished.set()
        assert slow_started.wait(timeout=20), "Slow request never started"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ProviderStream())

    with (
        OpenAI(
            api_key="test-key",
            base_url="https://batch.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as caller,
    ):

        def run_batch() -> object:
            if mode == "models":
                return batch_completion_models(
                    models=["openai/slow", "openai/fast"],
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                    client=client,
                    num_retries=0,
                )
            return batch_completion_models(
                deployments=[{"model": "openai/slow"}, {"model": "openai/fast"}],
                model_list=[],
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
                client=client,
                num_retries=0,
            )

        pending: Final = caller.submit(run_batch)
        try:
            response: Final = pending.result(timeout=10)
            assert isinstance(response, litellm.CustomStreamWrapper)
            assert not tokens_requested.is_set()
            assert not release_slow.is_set()
            release_tokens.set()
            if stream_error:
                with pytest.raises(MidStreamFallbackError, match="stream failed"):
                    tuple(response)
            else:
                assert "".join(chunk.choices[0].delta.content or "" for chunk in response) == "winner"
        finally:
            release_tokens.set()
            release_slow.set()
            assert slow_finished.wait(timeout=20), "Slow request did not finish"


def test_batch_completion_models_all_responses_submits_before_waiting(monkeypatch):
    """
    Regression test for issue #20704.
    Ensures all model calls are submitted to the thread pool before waiting on results.
    """
    models = ["model-a", "model-b", "model-c"]
    called_models = []

    class _AssertingFuture:
        def __init__(self, result, executor, expected_submissions):
            self._result = result
            self._executor = executor
            self._expected_submissions = expected_submissions

        def result(self):
            if self._executor.submit_count != self._expected_submissions:
                raise AssertionError("Not all model calls were submitted before waiting")
            return self._result

    class _RecordingThreadPoolExecutor:
        def __init__(self, max_workers, *args, **kwargs):
            self.max_workers = max_workers
            self.submit_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            self.submit_count += 1
            result = fn(*args, **kwargs)
            return _AssertingFuture(
                result=result,
                executor=self,
                expected_submissions=len(models),
            )

    def _mock_completion(*args, model, **kwargs):
        called_models.append(model)
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _RecordingThreadPoolExecutor)

    responses = batch_completion_models_all_responses(
        models=models,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert sorted(called_models) == sorted(models)
    assert len(responses) == len(models)
    assert sorted(response["model"] for response in responses) == sorted(models)


def test_batch_completion_models_all_responses_continues_on_model_error(monkeypatch):
    models = ["model-a", "model-error", "model-b"]

    def _mock_completion(*args, model, **kwargs):
        if model == "model-error":
            raise RuntimeError("simulated model failure")
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models=models,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert len(responses) == 2
    assert sorted(response["model"] for response in responses) == ["model-a", "model-b"]


def test_batch_completion_models_all_responses_returns_empty_for_empty_models(
    monkeypatch,
):
    called = False

    def _mock_completion(*args, model, **kwargs):
        nonlocal called
        called = True
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models=[],
        messages=[{"role": "user", "content": "hello"}],
    )

    assert responses == []
    assert called is False


def test_batch_completion_models_all_responses_accepts_single_model_string(monkeypatch):
    called_models = []

    def _mock_completion(*args, model, **kwargs):
        called_models.append(model)
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models="model-a",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert called_models == ["model-a"]
    assert responses == [{"model": "model-a"}]
