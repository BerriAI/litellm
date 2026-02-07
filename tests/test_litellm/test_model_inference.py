"""
Tests for model inference functionality.

These tests verify that self-hosted providers can infer model capabilities
from similar models in the registry.
"""

import pytest

import litellm
from litellm.model_inference import (
    SELF_HOSTED_PROVIDERS,
    aggregate_capabilities,
    extract_base_model_patterns,
    find_similar_models,
    infer_model_capabilities,
)


class TestExtractBaseModelPatterns:
    """Test pattern extraction from model names."""

    def test_llama_full_pattern(self):
        patterns = extract_base_model_patterns("my-custom-llama-3.1-70b-instruct")
        assert "llama-3.1-70b-instruct" in patterns
        assert "llama-3.1-70b" in patterns
        assert "llama-3.1" in patterns
        assert "llama" in patterns

    def test_mistral_pattern(self):
        patterns = extract_base_model_patterns("mistral-7b-instruct-v0.2")
        assert any("mistral" in p for p in patterns)
        assert any("7b" in p.lower() for p in patterns)

    def test_qwen_pattern(self):
        patterns = extract_base_model_patterns("qwen-72b-chat")
        assert any("qwen" in p for p in patterns)
        assert any("72b" in p.lower() for p in patterns)

    def test_removes_custom_prefix(self):
        patterns = extract_base_model_patterns("my-llama-3-8b")
        # Should remove "my-" prefix
        assert "llama" in patterns

    def test_case_insensitive(self):
        patterns = extract_base_model_patterns("Llama-3.1-70B-Instruct")
        # All patterns should be lowercase
        assert all(p.islower() for p in patterns)


class TestFindSimilarModels:
    """Test finding similar models in model_cost."""

    def test_finds_llama_models(self):
        # We know there are llama models in model_cost
        patterns = ["llama-3", "llama"]
        matches = find_similar_models(patterns)

        assert len(matches) > 0
        assert all("info" in m for m in matches)
        assert all("key" in m for m in matches)

    def test_stops_at_first_match(self):
        # Should find matches with first pattern and stop
        patterns = ["llama-3.1", "completely-unknown-model"]
        matches = find_similar_models(patterns)

        # Should only match llama-3.1, not try the unknown pattern
        assert len(matches) > 0
        assert all("llama" in m["key"].lower() for m in matches)

    def test_skips_inferred_models(self):
        # Add a fake inferred model
        litellm.model_cost["test_inferred"] = {"_inferred": True, "max_tokens": 1000}

        matches = find_similar_models(["test_inferred"])

        # Should not include inferred models
        assert not any(m["key"] == "test_inferred" for m in matches)

        # Cleanup
        del litellm.model_cost["test_inferred"]

    def test_no_matches_returns_empty(self):
        patterns = ["completely-unknown-model-xyz-123"]
        matches = find_similar_models(patterns)
        assert len(matches) == 0


class TestAggregateCapabilities:
    """Test aggregation of model capabilities."""

    def test_uses_max_for_numeric_fields(self):
        matches = [
            {"key": "model1", "info": {"max_tokens": 4096, "max_input_tokens": 2048}},
            {"key": "model2", "info": {"max_tokens": 8192, "max_input_tokens": 4096}},
            {"key": "model3", "info": {"max_tokens": 2048, "max_input_tokens": 1024}},
        ]

        result = aggregate_capabilities(matches)

        assert result["max_tokens"] == 8192  # max value
        assert result["max_input_tokens"] == 4096  # max value

    def test_uses_any_for_boolean_fields(self):
        matches = [
            {
                "key": "model1",
                "info": {"supports_vision": False, "supports_function_calling": True},
            },
            {
                "key": "model2",
                "info": {"supports_vision": True, "supports_function_calling": False},
            },
        ]

        result = aggregate_capabilities(matches)

        # Should be True if ANY model supports it
        assert result["supports_vision"] is True
        assert result["supports_function_calling"] is True

    def test_zeros_out_costs(self):
        matches = [
            {
                "key": "model1",
                "info": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
            },
        ]

        result = aggregate_capabilities(matches)

        assert result["input_cost_per_token"] == 0.0
        assert result["output_cost_per_token"] == 0.0

    def test_handles_none_values(self):
        matches = [
            {"key": "model1", "info": {"max_tokens": None, "supports_vision": None}},
            {"key": "model2", "info": {"max_tokens": 4096, "supports_vision": True}},
        ]

        result = aggregate_capabilities(matches)

        # Should ignore None values
        assert result["max_tokens"] == 4096
        assert result["supports_vision"] is True

    def test_empty_matches_returns_empty(self):
        result = aggregate_capabilities([])
        assert result == {}


class TestInferModelCapabilities:
    """Test end-to-end model inference."""

    def test_inference_for_llama_variant(self):
        # Test inferring a custom llama model
        result = infer_model_capabilities(
            model="my-custom-llama-3-70b", custom_llm_provider="hosted_vllm"
        )

        assert result is not None
        assert result["_inferred"] is True
        assert "_inferred_from" in result
        assert result["litellm_provider"] == "hosted_vllm"
        assert result["input_cost_per_token"] == 0.0
        assert result["output_cost_per_token"] == 0.0

        # Should have inferred some capabilities
        assert "max_tokens" in result or "max_input_tokens" in result

    def test_caches_inferred_result(self):
        model = "test-custom-llama-for-caching"
        provider = "hosted_vllm"
        cache_key = f"{provider}/{model}"

        # Clear cache if exists
        if cache_key in litellm.model_cost:
            del litellm.model_cost[cache_key]

        # First call - should infer
        result1 = infer_model_capabilities(model=model, custom_llm_provider=provider)

        # Check it's cached
        assert cache_key in litellm.model_cost

        # Second call - should use cache
        result2 = infer_model_capabilities(model=model, custom_llm_provider=provider)

        # Should return same result
        assert result1 == result2

        # Cleanup
        if cache_key in litellm.model_cost:
            del litellm.model_cost[cache_key]

    def test_caches_not_found_result(self):
        model = "completely-unknown-model-xyz-123-test"
        provider = "hosted_vllm"
        cache_key = f"{provider}/{model}"

        # Clear cache if exists
        if cache_key in litellm.model_cost:
            del litellm.model_cost[cache_key]

        # First call - should fail to infer
        result = infer_model_capabilities(model=model, custom_llm_provider=provider)

        assert result is None

        # Check not_found marker is cached
        assert cache_key in litellm.model_cost
        assert litellm.model_cost[cache_key].get("_not_found") is True

        # Cleanup
        if cache_key in litellm.model_cost:
            del litellm.model_cost[cache_key]

    def test_returns_none_for_non_self_hosted_provider(self):
        # Should not infer for providers like openai, anthropic, etc.
        result = infer_model_capabilities(model="some-model", custom_llm_provider="openai")

        assert result is None

    def test_all_self_hosted_providers_supported(self):
        # Verify all providers in SELF_HOSTED_PROVIDERS work
        for provider in SELF_HOSTED_PROVIDERS:
            # Should attempt inference (may or may not find matches)
            result = infer_model_capabilities(model="test-llama-model", custom_llm_provider=provider)
            # Just checking it doesn't crash
            assert result is None or isinstance(result, dict)

            # Cleanup cache
            cache_key = f"{provider}/test-llama-model"
            if cache_key in litellm.model_cost:
                del litellm.model_cost[cache_key]


class TestIntegrationWithGetModelInfo:
    """Test integration with litellm.get_model_info()."""

    def test_get_model_info_uses_inference(self):
        # This should trigger inference for a custom model name
        try:
            from litellm.utils import get_model_info

            # Test with a custom llama model
            model_info = get_model_info(
                model="my-custom-llama-3-8b", custom_llm_provider="hosted_vllm"
            )

            # Should succeed without raising "model not mapped" error
            assert model_info is not None
            # Model info is a dict with context window info
            has_context = (model_info.get("max_tokens") is not None) or (
                model_info.get("max_input_tokens") is not None
            )
            assert has_context, f"Expected max_tokens or max_input_tokens, got: {model_info}"
            assert model_info.get("input_cost_per_token") == 0.0  # Self-hosted should be free
            assert model_info.get("output_cost_per_token") == 0.0

        except ValueError as e:
            if "isn't mapped yet" in str(e):
                pytest.fail("Model inference should have prevented this error")
            raise

        finally:
            # Cleanup
            cache_key = "hosted_vllm/my-custom-llama-3-8b"
            if cache_key in litellm.model_cost:
                del litellm.model_cost[cache_key]

    def test_unknown_model_still_fails(self):
        # Models that can't be inferred should still fail
        from litellm.utils import get_model_info

        with pytest.raises(Exception, match="isn't mapped yet"):
            get_model_info(
                model="completely-unknown-model-xyz-no-matches",
                custom_llm_provider="hosted_vllm",
            )

        # Cleanup
        cache_key = "hosted_vllm/completely-unknown-model-xyz-no-matches"
        if cache_key in litellm.model_cost:
            del litellm.model_cost[cache_key]

