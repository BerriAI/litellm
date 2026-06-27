"""
Tests for AWS Bedrock embedding model pricing in the model cost map.

Regression test for the Amazon Titan Text Embeddings V2 commercial price,
which was previously set 10x too high (2e-07 instead of 2e-08).
AWS lists Titan Text Embeddings V2 at $0.02 per 1M input tokens
(= $0.00002 per 1K tokens = 2e-08 per token).
"""

import importlib


class TestBedrockEmbeddingPricing:
    """Test suite for Bedrock embedding model pricing in the cost map."""

    def test_titan_embed_v2_commercial_input_cost(self, monkeypatch):
        """Titan Text Embeddings V2 should be priced at $0.02 / 1M tokens (2e-08)."""
        # Scope the local-cost-map flag to this test only, so it does not leak
        # into sibling tests. monkeypatch restores the environment on teardown.
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

        import litellm.litellm_core_utils.get_model_cost_map
        import litellm

        # Reload so the cost map is re-read from the local file with the flag set.
        importlib.reload(litellm.litellm_core_utils.get_model_cost_map)
        importlib.reload(litellm)

        model = litellm.model_cost["amazon.titan-embed-text-v2:0"]

        assert model["input_cost_per_token"] == 2e-08
        assert model["output_cost_per_token"] == 0.0
        assert model["litellm_provider"] == "bedrock"
        assert model["mode"] == "embedding"
