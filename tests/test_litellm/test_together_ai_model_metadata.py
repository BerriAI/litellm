import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT: Final = Path(__file__).parents[2]

CostMap = dict[str, dict[str, object]]
COST_MAP_ADAPTER: Final = TypeAdapter(CostMap)

SERVERLESS_CHAT_MODELS: Final = (
    "together_ai/moonshotai/Kimi-K3",
    "together_ai/zai-org/GLM-5.2",
    "together_ai/deepseek-ai/DeepSeek-V4-Pro",
    "together_ai/deepseek-ai/DeepSeek-V4-Pro-0813",
    "together_ai/deepseek-ai/DeepSeek-V4-Flash-0731",
    "together_ai/moonshotai/Kimi-K2.7-Code",
    "together_ai/MiniMaxAI/MiniMax-M3",
    "together_ai/thinkingmachines/Inkling",
    "together_ai/thinkingmachines/Inkling-Small",
    "together_ai/Qwen/Qwen3.8-2.4T-A95B",
    "together_ai/Qwen/Qwen3.7-Max",
    "together_ai/Qwen/Qwen3.7-Plus",
    "together_ai/Qwen/Qwen3.6-Plus",
    "together_ai/Qwen/Qwen3.5-9B",
    "together_ai/nvidia/nemotron-3-ultra-550b-a55b",
    "together_ai/meta-models/Muse-Glimmer-30B",
    "together_ai/google/gemma-4-31B-it",
    "together_ai/pearl-ai/gemma-4-31b-it",
    "together_ai/google/gemma-3n-E4B-it",
    "together_ai/arize-ai/qwen-2-1.5b-instruct",
    "together_ai/Prism-ML/Ternary-Bonsai-27B",
    "together_ai/meta-llama/Llama-Guard-4-12B",
    "together_ai/openai/gpt-oss-120b",
    "together_ai/openai/gpt-oss-20b",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

DEPRECATED_MODELS: Final = {
    "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "2026-07-10",
    "together_ai/Qwen/Qwen3.5-397B-A17B": "2026-06-29",
    "together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": "2026-06-04",
    "together_ai/moonshotai/Kimi-K2.5": "2026-05-21",
    "together_ai/deepseek-ai/DeepSeek-R1": "2026-05-14",
    "together_ai/deepseek-ai/DeepSeek-V3.1": "2026-05-14",
    "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507": "2026-04-16",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1": "2026-04-16",
    "together_ai/zai-org/GLM-4.5-Air-FP8": "2026-04-02",
    "together_ai/zai-org/GLM-4.7": "2026-04-02",
    "together_ai/mistralai/Mistral-Small-24B-Instruct-2501": "2026-04-02",
    "together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct": "2026-04-02",
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "2026-03-31",
    "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "2026-03-06",
    "together_ai/moonshotai/Kimi-K2-Instruct-0905": "2026-03-06",
    "together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo": "2026-03-06",
    "together_ai/Qwen/Qwen3-Next-80B-A3B-Thinking": "2026-02-25",
    "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "2026-02-25",
    "together_ai/Qwen/Qwen3-235B-A22B-fp8-tput": "2026-02-06",
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct": "2026-02-06",
    "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo": "2026-02-06",
    "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "2026-02-06",
    "together_ai/deepseek-ai/DeepSeek-R1-0528-tput": "2026-02-03",
    "together_ai/mistralai/Mistral-7B-Instruct-v0.1": "2025-11-13",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "2025-11-13",
}


@pytest.fixture(scope="module")
def cost_map() -> CostMap:
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        return COST_MAP_ADAPTER.validate_python(json.load(f))


@pytest.mark.parametrize("model", SERVERLESS_CHAT_MODELS)
def test_together_serverless_chat_model_is_mapped(cost_map: CostMap, model: str):
    info = cost_map.get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"
    assert info["litellm_provider"] == "together_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] >= 0
    assert info["output_cost_per_token"] >= info["input_cost_per_token"]
    assert "deprecation_date" not in info

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.removeprefix("together_ai/")
    assert provider == "together_ai"


def test_together_kimi_k3_pricing_and_capabilities(cost_map: CostMap):
    info = cost_map["together_ai/moonshotai/Kimi-K3"]
    assert info["input_cost_per_token"] == 3e-06
    assert info["output_cost_per_token"] == 1.5e-05
    assert info["max_input_tokens"] == 1048576
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_vision"] is True
    assert info["supports_reasoning"] is True


def test_together_glm_52_pricing(cost_map: CostMap):
    info = cost_map["together_ai/zai-org/GLM-5.2"]
    assert info["input_cost_per_token"] == 1.4e-06
    assert info["output_cost_per_token"] == 4.4e-06
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True


def test_together_multilingual_e5_embedding_entry(cost_map: CostMap):
    info = cost_map["together_ai/intfloat/multilingual-e5-large-instruct"]
    assert info["mode"] == "embedding"
    assert info["input_cost_per_token"] == 2e-08
    assert info["max_input_tokens"] == 514
    assert info["output_vector_size"] == 1024


def test_together_llama_33_70b_repriced_to_current_together_rate(cost_map: CostMap):
    info = cost_map["together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    assert info["input_cost_per_token"] == 1.04e-06
    assert info["output_cost_per_token"] == 1.04e-06
    assert info["max_input_tokens"] == 131072


@pytest.mark.parametrize("model", sorted(DEPRECATED_MODELS))
def test_together_deprecated_model_carries_deprecation_date(cost_map: CostMap, model: str):
    info = cost_map.get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"
    assert info.get("deprecation_date") == DEPRECATED_MODELS[model]


def _successor(info: dict[str, object]) -> str | None:
    metadata = info.get("metadata")
    if not isinstance(metadata, dict):
        return None
    successor = metadata.get("successor")
    return successor if isinstance(successor, str) else None


def test_together_successor_metadata_points_at_live_models(cost_map: CostMap):
    successors = {
        model: successor
        for model, info in cost_map.items()
        if model.startswith("together_ai/") and (successor := _successor(info)) is not None
    }
    assert len(successors) >= 10
    for model, successor in successors.items():
        target = cost_map.get(successor)
        assert target is not None, f"{model} names successor {successor} that is not in the map"
        assert "deprecation_date" not in target, f"{model} names deprecated successor {successor}"


def test_together_backup_cost_map_in_sync(cost_map: CostMap):
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = COST_MAP_ADAPTER.validate_python(json.load(f))
    together_main = {k: v for k, v in cost_map.items() if k.startswith("together_ai/")}
    together_backup = {k: v for k, v in backup.items() if k.startswith("together_ai/")}
    assert together_backup == together_main
