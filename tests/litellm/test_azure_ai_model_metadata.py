"""Regression tests for azure_ai/* model metadata (LIT-3157).

These tests pin the pricing + context-window of newly-added Azure AI Foundry
models in `model_prices_and_context_window.json` so accidental edits
to the JSON file are caught in CI.
"""

import json
from pathlib import Path

import pytest

_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "model_prices_and_context_window.json"
)


@pytest.fixture(scope="module")
def model_cost():
    with _JSON_PATH.open() as f:
        return json.load(f)


# (model_key, input_cost, output_cost, max_input_tokens, max_output_tokens, mode)
_EXPECTED = [
    ("azure_ai/AI21-Jamba-1.5-Large", 2e-06, 8e-06, 256000, 4096, "chat"),
    ("azure_ai/AI21-Jamba-1.5-Mini", 2e-07, 4e-07, 256000, 4096, "chat"),
    ("azure_ai/Codestral-2501", 3e-07, 9e-07, 256000, 4096, "chat"),
    ("azure_ai/Cohere-command-r", 5e-07, 1.5e-06, 128000, 4096, "chat"),
    ("azure_ai/Cohere-command-r-08-2024", 5e-07, 1.5e-06, 128000, 4096, "chat"),
    ("azure_ai/Cohere-command-r-plus", 3e-06, 1.5e-05, 128000, 4096, "chat"),
    ("azure_ai/Cohere-command-r-plus-08-2024", 3e-06, 1.5e-05, 128000, 4096, "chat"),
    ("azure_ai/DeepSeek-R1-0528", 1.35e-06, 5.4e-06, 128000, 8192, "chat"),
    ("azure_ai/Meta-Llama-3-8B-Instruct", 3e-07, 6.1e-07, 8192, 2048, "chat"),
    ("azure_ai/Mistral-Large-2411", 2e-06, 6e-06, 128000, 4096, "chat"),
]


@pytest.mark.parametrize("model,inp,out,max_in,max_out,mode", _EXPECTED)
def test_azure_ai_model_metadata(model_cost, model, inp, out, max_in, max_out, mode):
    info = model_cost.get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == mode
    assert info["input_cost_per_token"] == pytest.approx(inp)
    assert info["output_cost_per_token"] == pytest.approx(out)
    assert info["max_input_tokens"] == max_in
    assert info["max_output_tokens"] == max_out
    assert info["max_tokens"] == max_out


def test_azure_ai_get_model_info_routing(monkeypatch):
    """End-to-end check via litellm.get_model_info — forces local map so
    we exercise the same code path used at proxy runtime."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Re-import to force re-evaluation under the env var.
    import importlib
    import litellm
    importlib.reload(litellm)

    sample_models = [
        "azure_ai/AI21-Jamba-1.5-Large",
        "azure_ai/Cohere-command-r-plus",
        "azure_ai/Meta-Llama-3-8B-Instruct",
    ]
    for m in sample_models:
        info = litellm.get_model_info(m)
        assert info["litellm_provider"] == "azure_ai"
        assert info["input_cost_per_token"] > 0
        assert info["output_cost_per_token"] > 0
