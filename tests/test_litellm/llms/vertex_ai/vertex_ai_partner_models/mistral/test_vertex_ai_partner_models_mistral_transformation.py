import litellm


def test_reasoning_effort_stays_unsupported_on_vertex_partner_models(local_model_cost_map):
    assert "reasoning_effort" in litellm.get_supported_openai_params(
        model="mistral-medium-3", custom_llm_provider="mistral"
    )
    assert "reasoning_effort" not in litellm.get_supported_openai_params(
        model="mistral-medium-3", custom_llm_provider="vertex_ai"
    )
    dropped = litellm.get_optional_params(
        model="mistral-medium-3",
        custom_llm_provider="vertex_ai",
        reasoning_effort="high",
        drop_params=True,
    )
    assert "reasoning_effort" not in dropped
