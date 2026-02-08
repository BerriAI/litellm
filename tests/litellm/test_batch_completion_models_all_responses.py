import threading

import litellm
from litellm.batch_completion.main import batch_completion_models_all_responses


def test_batch_completion_models_all_responses_submits_before_waiting(monkeypatch):
    """
    Regression test for issue #20704.
    Ensures all model calls are submitted to the thread pool before waiting on results.
    """
    models = ["model-a", "model-b", "model-c"]
    all_submitted = threading.Event()
    lock = threading.Lock()
    submitted_count = 0

    def _mock_completion(*args, model, **kwargs):
        nonlocal submitted_count
        with lock:
            submitted_count += 1
            if submitted_count == len(models):
                all_submitted.set()

        # If the implementation blocks on the first future before submitting all tasks,
        # this wait times out and the test fails with a clear assertion.
        if not all_submitted.wait(timeout=15):
            raise AssertionError("Not all model calls were submitted before waiting")
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models=models,
        messages=[{"role": "user", "content": "hello"}],
    )

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
