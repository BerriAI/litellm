from litellm.integrations.otel.model.metadata import RequestContext
from litellm.integrations.otel.routing import routed_model_name


def test_routed_model_name_prefers_served_model_when_response_echoes_model_group() -> None:
    """
    Regression test for #51029:
    When router rewrites response_obj["model"] to the requested model_group alias
    (e.g. "qwen-small"), gen_ai.response.model should resolve to the actual served
    model ("Qwen3-0.6B") rather than repeating the model_group alias.
    """
    resolved = routed_model_name(
        model="qwen-small",
        requested_model="qwen-small",
        decision=None,
        served_model="Qwen3-0.6B",
    )
    assert resolved == "Qwen3-0.6B"


def test_request_context_uses_served_model_when_response_model_was_overwritten_by_router() -> None:
    payload = {
        "model": "openai/Qwen3-0.6B",
        "model_group": "qwen-small",
        "custom_llm_provider": "openai",
        "metadata": {},
        "hidden_params": {},
        "response": {
            "id": "chatcmpl-123",
            "model": "qwen-small",
        },
    }

    ctx = RequestContext.from_standard_logging_payload(payload)  # type: ignore[arg-type]
    assert ctx.request_model == "qwen-small"
    assert ctx.response_model == "Qwen3-0.6B"
