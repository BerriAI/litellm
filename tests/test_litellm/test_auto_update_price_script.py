"""
Unit tests for .github/workflows/auto_update_price_and_context_window_file.py

Covers the three logic issues fixed in PR #23117:
  1. Overly-broad capability heuristic (base family names vs. explicit suffixes)
  2. Rerank models must not carry input_cost_per_token
  3. sync_local_data_with_remote must not overwrite non-zero local prices with 0.0
"""
import importlib.util
from pathlib import Path
import pytest

# The script imports aiohttp at module level; skip the whole module gracefully
# if it is not installed in the test environment.
pytest.importorskip("aiohttp")

# ---------------------------------------------------------------------------
# Load the script as a module without executing main()
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).parents[2] / ".github" / "workflows" / "auto_update_price_and_context_window_file.py"

spec = importlib.util.spec_from_file_location("auto_update_script", _SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

transform_nvidia_nim_data = mod.transform_nvidia_nim_data
transform_openrouter_data = mod.transform_openrouter_data
transform_vercel_ai_gateway_data = mod.transform_vercel_ai_gateway_data
sync_local_data_with_remote = mod.sync_local_data_with_remote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nim_row(model_id):
    return {"id": model_id}


def _openrouter_row(model_id, architecture=None):
    return {
        "id": model_id,
        "context_length": 8192,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "architecture": architecture,  # pass None through to exercise null-handling
    }


# ---------------------------------------------------------------------------
# 1. Capability heuristic: only explicit instruction-tuned suffixes
# ---------------------------------------------------------------------------

class TestNimCapabilityHeuristic:
    def test_instruct_suffix_gets_function_calling(self):
        result = transform_nvidia_nim_data([_nim_row("meta/llama3-8b-instruct")])
        obj = result["nvidia_nim/meta/llama3-8b-instruct"]
        assert obj.get("supports_function_calling") is True
        assert obj.get("supports_tool_choice") is True

    def test_chat_suffix_gets_function_calling(self):
        result = transform_nvidia_nim_data([_nim_row("mistralai/mistral-7b-chat")])
        obj = result["nvidia_nim/mistralai/mistral-7b-chat"]
        assert obj.get("supports_function_calling") is True

    def test_it_suffix_gets_function_calling(self):
        result = transform_nvidia_nim_data([_nim_row("google/gemma-7b-it")])
        obj = result["nvidia_nim/google/gemma-7b-it"]
        assert obj.get("supports_function_calling") is True

    def test_base_family_name_does_not_get_function_calling(self):
        """granite, falcon, codellama, starcoder base models must NOT be flagged."""
        for base_id in [
            "ibm/granite-8b",
            "tiiuae/falcon-7b",
            "meta/codellama-7b",
            "bigcode/starcoder",
        ]:
            result = transform_nvidia_nim_data([_nim_row(base_id)])
            obj = result[f"nvidia_nim/{base_id}"]
            assert "supports_function_calling" not in obj, (
                f"{base_id} should not have supports_function_calling"
            )
            assert "supports_tool_choice" not in obj, (
                f"{base_id} should not have supports_tool_choice"
            )


# ---------------------------------------------------------------------------
# 2. OpenRouter null architecture handling
# ---------------------------------------------------------------------------

class TestOpenRouterArchitecture:
    def test_null_architecture_does_not_crash(self):
        """Explicit null architecture from the API must not raise AttributeError."""
        row = _openrouter_row("some/model", architecture=None)
        result = transform_openrouter_data([row])
        obj = result["openrouter/some/model"]
        assert "supports_vision" not in obj
        assert "supports_audio_input" not in obj

    def test_explicit_null_architecture_does_not_crash(self):
        """Explicit null architecture (not just missing) must not crash."""
        row = {
            "id": "some/model",
            "context_length": 8192,
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "architecture": None,
        }
        result = transform_openrouter_data([row])
        obj = result["openrouter/some/model"]
        assert "supports_vision" not in obj
        assert "supports_audio_input" not in obj

    def test_vision_modality_flagged(self):
        row = _openrouter_row("vision-model", architecture={"input_modalities": ["text", "image"]})
        result = transform_openrouter_data([row])
        obj = result["openrouter/vision-model"]
        assert obj.get("supports_vision") is True

    def test_max_tokens_maps_to_output_not_context(self):
        """max_tokens must reflect the output limit, not the input context window."""
        row = {
            "id": "anthropic/claude-3-5-sonnet",
            "context_length": 200000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "top_provider": {"max_completion_tokens": 8192},
            "architecture": None,
        }
        result = transform_openrouter_data([row])
        obj = result["openrouter/anthropic/claude-3-5-sonnet"]
        assert obj["max_input_tokens"] == 200000, "context_length → max_input_tokens"
        assert obj["max_tokens"] == 8192, "max_tokens must be the output limit"
        assert obj["max_output_tokens"] == 8192

    def test_max_tokens_absent_when_no_completion_tokens(self):
        """When top_provider.max_completion_tokens is absent, max_tokens is not set."""
        row = {
            "id": "some/model",
            "context_length": 32768,
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "architecture": None,
        }
        result = transform_openrouter_data([row])
        obj = result["openrouter/some/model"]
        assert obj["max_input_tokens"] == 32768
        assert "max_tokens" not in obj
        assert "max_output_tokens" not in obj


# ---------------------------------------------------------------------------
# 3. Rerank models must not carry per-token pricing fields
# ---------------------------------------------------------------------------

class TestNimRerankPricing:
    def test_rerank_has_no_input_cost_per_token(self):
        result = transform_nvidia_nim_data([_nim_row("nvidia/nv-rerankqa-mistral-4b-v3")])
        obj = result["nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3"]
        assert obj["mode"] == "rerank"
        assert "input_cost_per_token" not in obj, (
            "Rerank model must not have input_cost_per_token"
        )

    def test_rerank_has_input_cost_per_query(self):
        result = transform_nvidia_nim_data([_nim_row("nvidia/nv-rerankqa-mistral-4b-v3")])
        obj = result["nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3"]
        assert "input_cost_per_query" in obj

    def test_chat_has_input_cost_per_token(self):
        result = transform_nvidia_nim_data([_nim_row("meta/llama3-8b-instruct")])
        obj = result["nvidia_nim/meta/llama3-8b-instruct"]
        assert obj["mode"] == "chat"
        assert "input_cost_per_token" in obj

    def test_embedding_has_input_cost_per_token(self):
        result = transform_nvidia_nim_data([_nim_row("nvidia/nv-embed-v1")])
        obj = result["nvidia_nim/nvidia/nv-embed-v1"]
        assert obj["mode"] == "embedding"
        assert "input_cost_per_token" in obj

    def test_rerankqa_matches_rerank(self):
        """rerankqa contains 'rerank' and is detected as rerank mode."""
        result = transform_nvidia_nim_data([_nim_row("nvidia/nv-rerankqa-mistral-4b-v3")])
        obj = result["nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3"]
        assert obj["mode"] == "rerank"
        assert "input_cost_per_query" in obj
        assert "input_cost_per_token" not in obj


# ---------------------------------------------------------------------------
# 3. sync_local_data_with_remote must preserve non-zero local prices
# ---------------------------------------------------------------------------

class TestSyncLocalDataWithRemote:
    def test_non_zero_nim_price_not_overwritten_by_zero(self):
        """NIM provider preserves non-zero local prices when remote returns 0.0."""
        local = {
            "nvidia_nim/meta/llama3-8b-instruct": {
                "litellm_provider": "nvidia_nim",
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
                "max_tokens": 8192,
            }
        }
        remote = {
            "nvidia_nim/meta/llama3-8b-instruct": {
                "litellm_provider": "nvidia_nim",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "max_tokens": 16384,
            }
        }
        sync_local_data_with_remote(local, remote)
        # Cost fields must be preserved from local (NIM only)
        assert local["nvidia_nim/meta/llama3-8b-instruct"]["input_cost_per_token"] == 0.00003
        assert local["nvidia_nim/meta/llama3-8b-instruct"]["output_cost_per_token"] == 0.00006
        # Non-cost fields should be updated from remote
        assert local["nvidia_nim/meta/llama3-8b-instruct"]["max_tokens"] == 16384

    def test_non_zero_non_nim_price_overwritten_by_zero(self):
        """Non-NIM providers accept 0.0 prices from remote (e.g., model becomes free)."""
        local = {
            "openai/gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
            }
        }
        remote = {
            "openai/gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        }
        sync_local_data_with_remote(local, remote)
        # Cost fields are updated to 0.0 (provider legit made model free)
        assert local["openai/gpt-4"]["input_cost_per_token"] == 0.0
        assert local["openai/gpt-4"]["output_cost_per_token"] == 0.0

    def test_zero_local_price_is_updated_by_nonzero_remote(self):
        local = {"m": {"input_cost_per_token": 0.0}}
        remote = {"m": {"input_cost_per_token": 0.00001}}
        sync_local_data_with_remote(local, remote)
        assert local["m"]["input_cost_per_token"] == 0.00001

    def test_new_models_added_from_remote(self):
        local = {}
        remote = {"new_model": {"input_cost_per_token": 0.0, "mode": "chat"}}
        sync_local_data_with_remote(local, remote)
        assert "new_model" in local
        assert local["new_model"]["input_cost_per_token"] == 0.0

    def test_manually_curated_fields_preserved(self):
        """Fields the remote does not return (supports_assistant_prefill etc.) survive."""
        local = {
            "m": {
                "input_cost_per_token": 0.00001,
                "supports_assistant_prefill": True,
            }
        }
        remote = {"m": {"input_cost_per_token": 0.00001}}
        sync_local_data_with_remote(local, remote)
        assert local["m"].get("supports_assistant_prefill") is True


# ---------------------------------------------------------------------------
# 4. Vercel image/video models must not get per-token pricing fields
# ---------------------------------------------------------------------------

def _vercel_row(model_id, model_type="", pricing=None, context_window=0):
    return {
        "id": model_id,
        "type": model_type,
        "pricing": pricing or {},
        "context_window": context_window,
    }


class TestVercelTokenPricing:
    def test_image_model_has_no_token_pricing(self):
        row = _vercel_row("black-forest-labs/flux-1-schnell", model_type="image",
                          pricing={"input": "0.000001", "output": "0.000002", "image": "0.003"})
        result = transform_vercel_ai_gateway_data([row])
        obj = result["vercel_ai_gateway/black-forest-labs/flux-1-schnell"]
        assert obj["mode"] == "image_generation"
        assert "input_cost_per_token" not in obj, "Image model must not have input_cost_per_token"
        assert "output_cost_per_token" not in obj, "Image model must not have output_cost_per_token"
        assert obj.get("input_cost_per_image") == 0.003

    def test_video_model_has_no_token_pricing(self):
        row = _vercel_row("google/veo-2", model_type="video",
                          pricing={"input": "0.000001", "output": "0.000002"})
        result = transform_vercel_ai_gateway_data([row])
        obj = result["vercel_ai_gateway/google/veo-2"]
        assert obj["mode"] == "video_generation"
        assert "input_cost_per_token" not in obj
        assert "output_cost_per_token" not in obj

    def test_chat_model_has_token_pricing(self):
        row = _vercel_row("openai/gpt-4o", model_type="",
                          pricing={"input": "0.0000025", "output": "0.00001"},
                          context_window=128000)
        result = transform_vercel_ai_gateway_data([row])
        obj = result["vercel_ai_gateway/openai/gpt-4o"]
        assert obj["mode"] == "chat"
        assert "input_cost_per_token" in obj
        assert "output_cost_per_token" in obj

    def test_embedding_model_has_token_pricing(self):
        row = _vercel_row("openai/text-embedding-3-small", model_type="embedding",
                          pricing={"input": "0.00000002"})
        result = transform_vercel_ai_gateway_data([row])
        obj = result["vercel_ai_gateway/openai/text-embedding-3-small"]
        assert obj["mode"] == "embedding"
        assert "input_cost_per_token" in obj

    def test_null_pricing_does_not_crash(self):
        """Explicit null pricing from the API must not raise AttributeError."""
        row = {
            "id": "some/model",
            "type": "chat",
            "pricing": None,
            "context_window": 4096,
        }
        result = transform_vercel_ai_gateway_data([row])
        obj = result["vercel_ai_gateway/some/model"]
        assert obj["mode"] == "chat"
        assert "input_cost_per_token" not in obj
