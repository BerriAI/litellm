"""Unit tests for Morph provider integration."""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import NotRequired, TypedDict

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

import litellm
from litellm import get_llm_provider


class MorphModelInfo(TypedDict):
    cache_read_input_token_cost: float
    input_cost_per_token: float
    litellm_provider: str
    max_input_tokens: int
    max_output_tokens: int
    max_tokens: int
    mode: str
    output_cost_per_token: float
    source: str
    supported_modalities: tuple[str, ...]
    supports_function_calling: bool
    supports_native_streaming: bool
    supports_native_structured_output: NotRequired[bool]
    supports_prompt_caching: NotRequired[bool]
    supports_response_schema: NotRequired[bool]
    supports_system_messages: bool
    supports_vision: bool


@dataclass(frozen=True, slots=True)
class MorphModelExpectation:
    model_id: str
    max_input_tokens: int
    max_output_tokens: int
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float
    modalities: tuple[str, ...]


MORPH_MODELS: Final = (
    MorphModelExpectation("morph-v3-fast", 262144, 131072, 8e-7, 1.2e-6, 0, ("text",)),
    MorphModelExpectation("morph-v3-large", 262144, 131072, 9e-7, 1.9e-6, 0, ("text",)),
    MorphModelExpectation("morph-qwen35-397b", 262144, 131072, 5e-7, 3.5e-6, 3e-7, ("text", "image")),
    MorphModelExpectation("morph-minimax3-428b", 256000, 256000, 3e-7, 1.2e-6, 0, ("text", "image")),
    MorphModelExpectation("morph-glm52-744b", 1048576, 1048576, 1.1e-6, 4.1e-6, 2.2e-7, ("text",)),
    MorphModelExpectation("morph-qwen36-27b", 131072, 131072, 2.89e-7, 2.4e-6, 0, ("text",)),
    MorphModelExpectation("morph-dsv4flash", 1048576, 1048576, 1.39e-7, 2.78e-7, 7e-8, ("text",)),
    MorphModelExpectation("morph-dsv4flash-0731", 1048576, 1048576, 1.39e-7, 2.78e-7, 7e-8, ("text",)),
    MorphModelExpectation("morph-gemma4-31b", 175000, 175000, 1.4e-7, 4e-7, 8e-8, ("text", "image")),
    MorphModelExpectation("morph-kimik3", 1048576, 1048576, 2.5e-6, 1.4e-5, 2.9e-7, ("text", "image")),
    MorphModelExpectation("morph-kimik3-fast", 1048576, 1048576, 6e-6, 2.25e-5, 6e-7, ("text", "image")),
)
_RAW_MODEL_COST_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
_MORPH_MODEL_INFO_ADAPTER: Final = TypeAdapter(MorphModelInfo)


def _load_morph_model_cost(path: Path) -> Mapping[str, MorphModelInfo]:
    raw_model_cost: Final = _RAW_MODEL_COST_ADAPTER.validate_json(path.read_bytes())
    return MappingProxyType(
        {
            key: _MORPH_MODEL_INFO_ADAPTER.validate_python(value)
            for key, value in raw_model_cost.items()
            if key.startswith("morph/")
        }
    )


# Force model loading
litellm.add_known_models()


def test_morph_get_llm_provider():
    """Test that get_llm_provider correctly identifies morph models."""
    for model in MORPH_MODELS:
        _, custom_llm_provider, _, _ = get_llm_provider(f"morph/{model.model_id}")
        assert custom_llm_provider == "morph"


def test_morph_in_provider_lists():
    """Test that morph is included in all necessary provider lists."""
    import litellm
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
    )

    # Check morph is in openai_compatible_providers
    assert "morph" in openai_compatible_providers

    # Check morph endpoint is in openai_compatible_endpoints
    assert "https://api.morphllm.com/v1" in openai_compatible_endpoints

    # Check morph is in provider_list
    assert "morph" in litellm.provider_list

    # Check models are in model_list after initialization
    assert all(f"morph/{model.model_id}" in litellm.model_list for model in MORPH_MODELS)


def test_morph_model_info():
    """Test that morph models have correct configuration."""
    repo_root: Final = Path(__file__).parents[2]
    model_cost: Final = _load_morph_model_cost(repo_root / "model_prices_and_context_window.json")

    for expected in MORPH_MODELS:
        model_info = model_cost[f"morph/{expected.model_id}"]
        public_model_info = litellm.get_model_info(f"morph/{expected.model_id}")
        assert model_info["litellm_provider"] == "morph"
        assert model_info["mode"] == "chat"
        assert model_info["max_input_tokens"] == expected.max_input_tokens
        assert model_info["max_output_tokens"] == expected.max_output_tokens
        assert model_info["max_tokens"] == expected.max_output_tokens
        assert model_info["input_cost_per_token"] == expected.input_cost_per_token
        assert model_info["output_cost_per_token"] == expected.output_cost_per_token
        assert model_info.get("cache_read_input_token_cost") == expected.cache_read_input_token_cost
        assert tuple(model_info["supported_modalities"] or ()) == expected.modalities
        assert model_info["source"] == "https://www.morphllm.com/api/models/json"
        assert model_info["supports_native_streaming"] is True
        assert model_info["supports_vision"] is ("image" in expected.modalities)
        assert public_model_info["max_input_tokens"] == expected.max_input_tokens
        assert public_model_info["max_output_tokens"] == expected.max_output_tokens
        assert public_model_info["input_cost_per_token"] == expected.input_cost_per_token
        assert public_model_info["output_cost_per_token"] == expected.output_cost_per_token

        if expected.model_id.startswith("morph-v3-"):
            assert model_info["supports_function_calling"] is False
            assert model_info["supports_system_messages"] is True
        else:
            assert model_info["supports_function_calling"] is True
            assert model_info.get("supports_native_structured_output") is True
            assert model_info.get("supports_prompt_caching") is True
            assert model_info.get("supports_response_schema") is True
            assert model_info["supports_system_messages"] is True


def test_morph_model_info_matches_backup():
    repo_root: Final = Path(__file__).parents[2]
    model_cost: Final = _load_morph_model_cost(repo_root / "model_prices_and_context_window.json")
    backup_model_cost: Final = _load_morph_model_cost(repo_root / "litellm/model_prices_and_context_window_backup.json")

    for model in MORPH_MODELS:
        assert backup_model_cost[f"morph/{model.model_id}"] == model_cost[f"morph/{model.model_id}"]
