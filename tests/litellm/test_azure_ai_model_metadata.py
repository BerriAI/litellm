"""Regression tests for azure_ai/* model metadata (LIT-3157).

These tests pin the pricing + context-window of newly-added Azure AI Foundry
models in `model_prices_and_context_window.json` so accidental edits
to the JSON file are caught in CI.
"""

import json
import subprocess
import sys
import textwrap
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


def test_azure_ai_model_metadata_in_backup(model_cost):
    """The bundled backup JSON must contain the same entries — that is the
    file litellm.get_model_info() actually loads at runtime when
    LITELLM_LOCAL_MODEL_COST_MAP=True (or when the remote fetch fails)."""
    backup_path = (
        Path(__file__).resolve().parents[2]
        / "litellm"
        / "model_prices_and_context_window_backup.json"
    )
    with backup_path.open() as f:
        backup = json.load(f)

    for model, inp, out, max_in, max_out, mode in _EXPECTED:
        info = backup.get(model)
        assert info is not None, (
            f"{model} missing from model_prices_and_context_window_backup.json — "
            "main JSON and bundled backup must stay in sync."
        )
        assert info["litellm_provider"] == "azure_ai"
        assert info["mode"] == mode
        assert info["input_cost_per_token"] == pytest.approx(inp)
        assert info["output_cost_per_token"] == pytest.approx(out)
        assert info["max_input_tokens"] == max_in


def test_azure_ai_get_model_info_routing():
    """End-to-end check via litellm.get_model_info — runs in a fresh
    Python subprocess so the in-session pytest module state isn't polluted
    by importing litellm with LITELLM_LOCAL_MODEL_COST_MAP=True."""
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import json
        import sys

        import litellm

        sample_models = [
            "azure_ai/AI21-Jamba-1.5-Large",
            "azure_ai/Cohere-command-r-plus",
            "azure_ai/Meta-Llama-3-8B-Instruct",
        ]
        out = {}
        for m in sample_models:
            info = litellm.get_model_info(m)
            out[m] = {
                "litellm_provider": info["litellm_provider"],
                "input_cost_per_token": info["input_cost_per_token"],
                "output_cost_per_token": info["output_cost_per_token"],
            }
        json.dump(out, sys.stdout)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env={
            "PATH": "/usr/bin:/usr/local/bin",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "HOME": "/tmp",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"subprocess failed: stdout={proc.stdout!r} stderr={proc.stderr[-1500:]!r}"
    )
    payload = json.loads(proc.stdout)
    for m in payload:
        assert payload[m]["litellm_provider"] == "azure_ai"
        assert payload[m]["input_cost_per_token"] > 0
        assert payload[m]["output_cost_per_token"] > 0
