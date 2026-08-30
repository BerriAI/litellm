import pytest

import litellm


@pytest.mark.parametrize("provider", ["azure", "together_ai"])
def test_embedding_dimensions_drop_params_for_openai_compatible_provider(provider):
    previous_drop_params = litellm.drop_params
    try:
        litellm.drop_params = False
        dropped = litellm.utils.get_optional_params_embeddings(
            model=f"{provider}/dummy-model",
            custom_llm_provider=provider,
            dimensions=512,
            drop_params=True,
        )
        assert "dimensions" not in dropped

        litellm.drop_params = True
        dropped_globally = litellm.utils.get_optional_params_embeddings(
            model=f"{provider}/dummy-model",
            custom_llm_provider=provider,
            dimensions=512,
        )
        assert "dimensions" not in dropped_globally

        litellm.drop_params = False
        preserved = litellm.utils.get_optional_params_embeddings(
            model=f"{provider}/dummy-model",
            custom_llm_provider=provider,
            dimensions=512,
        )
        assert preserved["dimensions"] == 512

        litellm.drop_params = True
        model_supported = litellm.utils.get_optional_params_embeddings(
            model=f"{provider}/text-embedding-3-small",
            custom_llm_provider=provider,
            dimensions=512,
        )
        assert model_supported["dimensions"] == 512

        explicitly_allowed = litellm.utils.get_optional_params_embeddings(
            model=f"{provider}/legacy-model",
            custom_llm_provider=provider,
            dimensions=512,
            allowed_openai_params=["dimensions"],
        )
        assert explicitly_allowed["dimensions"] == 512
    finally:
        litellm.drop_params = previous_drop_params


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("nvidia_nim", "nvidia_nim/nv-embedqa-e5-v5"),
        ("fireworks_ai", "fireworks_ai/nomic-ai/nomic-embed-text-v1.5"),
        ("dashscope", "dashscope/text-embedding-v3"),
        ("hosted_vllm", "hosted_vllm/Qwen/Qwen3-Embedding-0.6B"),
    ],
)
def test_embedding_dimensions_preserved_for_provider_mappings(provider, model):
    previous_drop_params = litellm.drop_params
    try:
        litellm.drop_params = True
        optional_params = litellm.utils.get_optional_params_embeddings(
            model=model,
            custom_llm_provider=provider,
            dimensions=128,
        )
    finally:
        litellm.drop_params = previous_drop_params

    assert optional_params["dimensions"] == 128
