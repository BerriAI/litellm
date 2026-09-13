import json
from pathlib import Path

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

MUSE_SPARK_MODEL = "meta/muse-spark-1.1"


def test_muse_spark_1_1_model_info():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(MUSE_SPARK_MODEL)
    assert info is not None, f"{MUSE_SPARK_MODEL} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "meta"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 1.25e-06
    assert info["output_cost_per_token"] == 4.25e-06
    assert info["cache_read_input_token_cost"] == 1.5e-07

    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 131072
    assert info["max_tokens"] == 131072

    assert info["supports_function_calling"] is True
    assert info["supports_parallel_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_pdf_input"] is True
    assert info["supports_web_search"] is True
    assert info["supports_minimal_reasoning_effort"] is True
    assert info["supports_xhigh_reasoning_effort"] is True

    assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
    assert info["supported_modalities"] == ["text", "image", "video"]
    assert info["supported_output_modalities"] == ["text"]

    routed_model, provider, _, api_base = get_llm_provider(model=MUSE_SPARK_MODEL, api_key="sk-test")
    assert routed_model == "muse-spark-1.1"
    assert provider == "meta"
    assert api_base == "https://api.meta.ai/v1"


def test_muse_spark_1_1_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    assert backup_cost.get(MUSE_SPARK_MODEL) == main_cost.get(MUSE_SPARK_MODEL), (
        f"{MUSE_SPARK_MODEL} differs between main and backup model cost maps"
    )
