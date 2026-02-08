import threading

import litellm
from litellm.batch_completion.main import batch_completion_models_all_responses


def test_batch_completion_models_all_responses_submits_before_waiting(monkeypatch):
    """
    Regression test for issue #20704.
    Ensures all model calls are submitted to the thread pool before waiting on results.
    """
    models = ["model-a", "model-b", "model-c"]
    barrier = threading.Barrier(parties=len(models), timeout=2)

    def _mock_completion(*args, model, **kwargs):
        barrier.wait()
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models=models,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert len(responses) == len(models)
    assert sorted(response["model"] for response in responses) == sorted(models)
