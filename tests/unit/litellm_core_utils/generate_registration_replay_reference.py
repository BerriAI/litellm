"""Run with the pre-migration Python tree via --reference-root, never the implementation under test."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Final


def cases() -> tuple[dict[str, object], ...]:
    base: Final = {
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
            "max_input_tokens": 8192,
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "cache_read_input_token_cost": 0.003,
            "explicit_cache_read_input_token_cost": 0.004,
            "aliases": ["parity-alias"],
        },
        "anthropic.parity-model-v1:0": {
            "litellm_provider": "bedrock",
            "mode": "chat",
            "max_input_tokens": 4096,
            "input_cost_per_token": 0.03,
            "output_cost_per_token": 0.04,
            "supports_function_calling": True,
            "cache_read_input_token_cost": 0.005,
        },
    }
    specifications: Final = (
        (
            "custom-provider",
            "parity-custom",
            {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.5},
            {},
        ),
        (
            "scientific-strings",
            "parity-scientific",
            {"litellm_provider": "openai", "input_cost_per_token": "3e-07", "output_cost_per_token": "6e-07"},
            {},
        ),
        ("zero-price", "gpt-4o", {"input_cost_per_token": 0.0}, {}),
        ("provider-prefix", "openai/gpt-4o", {"input_cost_per_token": 0.5}, {}),
        ("canonical", "gpt-4o", {"input_cost_per_token": 0.5}, {}),
        ("alias", "parity-alias", {"input_cost_per_token": 0.5}, {}),
        ("none-provider", "parity-unknown", {"litellm_provider": None, "input_cost_per_token": 0.5}, {}),
        ("none-price", "gpt-4o", {"input_cost_per_token": None, "supports_function_calling": False}, {}),
        ("cache-replaces-explicit", "gpt-4o", {"cache_read_input_token_cost": 0.007}, {}),
        ("explicit-cache", "gpt-4o", {"explicit_cache_read_input_token_cost": 0.008}, {}),
        ("cache-creation", "gpt-4o", {"cache_creation_input_token_cost": 0.009}, {}),
        ("request-scoped", "gpt-4o", {"input_cost_per_token": 0.5}, {"persist_across_reloads": False}),
        ("azure-context", "azure/parity-deployment", {"input_cost_per_token": 0.5}, {"custom_llm_provider": "azure"}),
        (
            "openrouter-bare",
            "parity-router",
            {"litellm_provider": "openrouter", "mode": "chat", "input_cost_per_token": 0.5},
            {},
        ),
        (
            "openrouter-prefixed",
            "openrouter/parity-router",
            {"litellm_provider": "openrouter", "input_cost_per_token": 0.5},
            {},
        ),
        (
            "no-provider-set-write",
            "parity-custom",
            {"litellm_provider": "openai", "input_cost_per_token": 0.5},
            {"add_to_provider_model_sets": False},
        ),
        (
            "time-pricing",
            "parity-time",
            {"litellm_provider": "openai", "input_cost_per_second": 0.5, "output_cost_per_second": 0.6},
            {},
        ),
        (
            "audio-pricing",
            "parity-audio",
            {"litellm_provider": "openai", "input_cost_per_audio_token": 0.5, "output_cost_per_audio_token": 0.6},
            {},
        ),
        ("metadata", "gpt-4o", {"supports_function_calling": True, "max_input_tokens": 16384}, {}),
        ("nested-metadata", "parity-custom", {"litellm_provider": "openai", "metadata": {"one": 1}}, {}),
        (
            "region-inheritance",
            "bedrock/ap-northeast-1.anthropic.parity-model-v1:0",
            {"litellm_provider": "bedrock", "input_cost_per_token": 0.5},
            {},
        ),
        (
            "repeated-prefix",
            "bedrock/bedrock/anthropic.parity-model-v1:0",
            {"litellm_provider": "bedrock", "input_cost_per_token": 0.5},
            {},
        ),
    )
    simple: Final = tuple(
        {
            "name": name,
            "base": copy.deepcopy(base),
            "calls": [{"model_cost": {key: fields}, **kwargs}],
            "reload": copy.deepcopy(base),
        }
        for name, key, fields, kwargs in specifications
    )
    repeated: Final = {
        "name": "sparse-repeat-none-string-and-capability",
        "base": copy.deepcopy(base),
        "calls": [
            {"model_cost": {"openai/gpt-4o": {"input_cost_per_token": 0.5}}},
            {"model_cost": {"openai/gpt-4o": {"input_cost_per_token": None, "supports_function_calling": True}}},
            {"model_cost": {"openai/gpt-4o": {"input_cost_per_token": "0.75"}}},
            {"model_cost": {"openai/gpt-4o": {"supports_function_calling": False}}},
        ],
        "reload": copy.deepcopy(base),
    }
    changed_keys: Final = {
        "name": "reload-only-prefixed-key",
        "base": copy.deepcopy(base),
        "calls": [{"model_cost": {"openai/gpt-4o": {"input_cost_per_token": 0.5}}}],
        "reload": {
            "openai/gpt-4o": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.9,
                "max_input_tokens": 4242,
            }
        },
    }
    return (*simple, repeated, changed_keys)


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args: Final = parser.parse_args()
    reference_root: Final = str(args.reference_root.resolve())
    sys.path.insert(0, reference_root)  # test-quality-ok: load the separate pre-migration reference checkout
    litellm: Final = importlib.import_module("litellm")
    loader: Final = importlib.import_module("litellm.litellm_core_utils.get_model_cost_map")
    utils: Final = importlib.import_module("litellm.utils")
    if hasattr(loader, "LoadedCatalog"):
        parser.error("reference must be the pre-migration Python implementation")
    captured: Final = []  # mutable-ok: append each independent baseline execution to the generated fixture
    for case in cases():
        utils._runtime_registered_model_cost.clear()
        utils._LiveDeploymentReplay.callback = None
        baseline: Final = copy.deepcopy(case["base"])
        litellm.model_cost = loader._finalize_model_cost_map(baseline)  # test-quality-ok: capture reference behavior
        loader.GetModelCostMap._snapshot_loaded_catalog(litellm.model_cost)
        utils._invalidate_model_cost_lowercase_map()
        for call in case["calls"]:
            litellm.register_model(**copy.deepcopy(call))
        before: Final = copy.deepcopy(dict(litellm.model_cost))
        loader.adopt_model_cost_map(loader._finalize_model_cost_map(copy.deepcopy(case["reload"])))
        captured.append({**case, "before_reload": before, "after_reload": copy.deepcopy(dict(litellm.model_cost))})
    args.output.write_text(
        json.dumps({"reference_revision": args.reference_revision, "cases": captured}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
