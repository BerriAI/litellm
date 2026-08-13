"""An explicit provider outranks a model name that is also a known OpenAI model.

completion() used to dispatch on `model in litellm.open_ai_chat_completion_models`
ahead of custom_llm_provider, so a Gemini call whose model id litellm knows as
OpenAI's went to the OpenAI handler while provider_config was still
VertexGeminiConfig, and raised NotImplementedError("Vertex AI has a custom
implementation of transform_request") before the request was sent.
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm


@pytest.fixture
def restore_model_registry():
    model_cost = dict(litellm.model_cost)
    openai_models = set(litellm.open_ai_chat_completion_models)
    yield
    litellm.model_cost.clear()
    litellm.model_cost.update(model_cost)
    litellm.open_ai_chat_completion_models.clear()
    litellm.open_ai_chat_completion_models.update(openai_models)


def _call(model: str) -> tuple[MagicMock, MagicMock]:
    """Run completion() with both candidate handlers stubbed out."""
    vertex = MagicMock(return_value=litellm.ModelResponse())
    openai = MagicMock(return_value=litellm.ModelResponse())
    with (
        patch.object(litellm.main.vertex_chat_completion, "completion", vertex),
        patch.object(litellm.main.openai_chat_completions, "completion", openai),
    ):
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
        )
    return vertex, openai


def test_gemini_provider_wins_over_an_openai_model_name():
    vertex, openai = _call("gemini/gpt-4o")

    assert vertex.called
    assert not openai.called
    assert vertex.call_args.kwargs["custom_llm_provider"] == "gemini"


def test_openai_model_still_routes_to_openai():
    vertex, openai = _call("gpt-4o")

    assert openai.called
    assert not vertex.called


def test_registered_openai_label_does_not_reroute_gemini(restore_model_registry):
    # register_model() adds the name to litellm.open_ai_chat_completion_models
    # whenever the entry claims "openai", which is how a mislabelled pricing
    # entry reroutes a provider it has nothing to do with.
    litellm.register_model(
        {
            "gemini-2.5-pro": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 4e-06,
            }
        }
    )
    assert "gemini-2.5-pro" in litellm.open_ai_chat_completion_models

    vertex, openai = _call("gemini/gemini-2.5-pro")

    assert vertex.called
    assert not openai.called
