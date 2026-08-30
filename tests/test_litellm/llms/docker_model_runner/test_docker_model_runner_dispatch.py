"""Tests for the Docker Model Runner dispatch wiring in litellm/main.py."""

from unittest.mock import MagicMock, patch

import litellm
from litellm.main import _complete_docker_model_runner
from litellm.types.completion import _CompletionDispatchContext


def _make_ctx() -> _CompletionDispatchContext:
    return _CompletionDispatchContext(
        _azure_detection_model="",
        acompletion=False,
        api_base=None,
        api_key=None,
        api_version=None,
        client=None,
        custom_llm_provider="docker_model_runner",
        custom_prompt_dict={},
        extra_headers=None,
        headers={},
        hf_model_name=None,
        kwargs={},
        litellm_params={},
        logger_fn=None,
        logging=MagicMock(),
        max_retries=None,
        max_tokens=None,
        messages=[{"role": "user", "content": "hi"}],
        metadata=None,
        model="docker_model_runner/ai/smollm2",
        model_response=litellm.ModelResponse(),
        optional_params={},
        organization=None,
        provider_config=None,
        shared_session=None,
        stream=False,
        temperature=None,
        text_completion=False,
        timeout=None,
        top_p=None,
    )


def test_complete_docker_model_runner_uses_shared_http_handler():
    """The dispatch forwards the context to the shared handler; the provider config resolves the default URL."""
    ctx = _make_ctx()
    sentinel = MagicMock(name="response")

    with patch(  # test-quality-ok: the dispatch's only injection seam is the shared handler sink
        "litellm.main.base_llm_http_handler.completion", return_value=sentinel
    ) as mock_completion:
        result = _complete_docker_model_runner(ctx)

    assert result is sentinel
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["api_base"] is None
    assert call_kwargs["custom_llm_provider"] == "docker_model_runner"
    assert call_kwargs["messages"] == ctx.messages
    ctx.logging.post_call.assert_called_once()


def test_complete_docker_model_runner_respects_env_api_base(monkeypatch):
    """DOCKER_MODEL_RUNNER_API_BASE is picked up when no explicit api_base is set."""
    monkeypatch.setenv("DOCKER_MODEL_RUNNER_API_BASE", "http://model-runner.docker.internal:12434/engines/v1")
    ctx = _make_ctx()
    sentinel = MagicMock(name="response")

    with patch(  # test-quality-ok: the dispatch's only injection seam is the shared handler sink
        "litellm.main.base_llm_http_handler.completion", return_value=sentinel
    ) as mock_completion:
        result = _complete_docker_model_runner(ctx)

    assert result is sentinel
    assert mock_completion.call_args.kwargs["api_base"] == "http://model-runner.docker.internal:12434/engines/v1"
