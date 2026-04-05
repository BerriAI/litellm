"""
Tests for gemini-3.1-flash-image-preview supports_reasoning flag.

Verifies that supports_reasoning returns True for all provider variants,
including the fallback from provider-prefixed entries to the base entry.
"""

import os

import pytest

import litellm
from litellm.utils import supports_reasoning


@pytest.fixture(autouse=True)
def load_local_model_cost():
    """Load model cost from local JSON so that uncommitted changes are picked up."""
    original_env = os.getenv("LITELLM_LOCAL_MODEL_COST_MAP")
    original_model_cost = getattr(litellm, "model_cost", None)

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    yield

    if original_env is None:
        os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    else:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = original_env

    if original_model_cost is not None:
        litellm.model_cost = original_model_cost


def test_gemini_3_1_flash_image_preview_supports_reasoning_base():
    """Test that the base entry supports reasoning."""
    assert supports_reasoning(model="gemini-3.1-flash-image-preview") is True


def test_gemini_3_1_flash_image_preview_supports_reasoning_gemini_provider():
    """Test that the gemini/ prefixed entry inherits supports_reasoning via fallback."""
    assert (
        supports_reasoning(
            model="gemini/gemini-3.1-flash-image-preview",
            custom_llm_provider="gemini",
        )
        is True
    )


def test_gemini_3_1_flash_image_preview_supports_reasoning_vertex_ai_provider():
    """Test that the vertex_ai/ prefixed entry inherits supports_reasoning via fallback."""
    assert (
        supports_reasoning(
            model="gemini-3.1-flash-image-preview",
            custom_llm_provider="vertex_ai",
        )
        is True
    )
