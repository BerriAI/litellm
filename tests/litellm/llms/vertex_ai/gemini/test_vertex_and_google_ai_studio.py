import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)


def test_top_logprobs():
    non_default_params = {
        "top_logprobs": 2,
        "logprobs": True,
    }
    optional_params = {}
    model = "gemini"

    v = VertexGeminiConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert v["responseLogprobs"] is non_default_params["logprobs"]
    assert v["logprobs"] is non_default_params["top_logprobs"]


def test_get_model_for_vertex_ai_url():
    # Test case 1: Regular model name
    model = "gemini-pro"
    result = VertexGeminiConfig.get_model_for_vertex_ai_url(model)
    assert result == "gemini-pro"

    # Test case 2: Gemini spec model with UUID
    model = "gemini/ft-uuid-123"
    result = VertexGeminiConfig.get_model_for_vertex_ai_url(model)
    assert result == "ft-uuid-123"


def test_is_model_gemini_spec_model():
    # Test case 1: None input
    assert VertexGeminiConfig._is_model_gemini_spec_model(None) == False

    # Test case 2: Regular model name
    assert VertexGeminiConfig._is_model_gemini_spec_model("gemini-pro") == False

    # Test case 3: Gemini spec model
    assert VertexGeminiConfig._is_model_gemini_spec_model("gemini/custom-model") == True


def test_get_model_name_from_gemini_spec_model():
    # Test case 1: Regular model name
    model = "gemini-pro"
    result = VertexGeminiConfig._get_model_name_from_gemini_spec_model(model)
    assert result == "gemini-pro"

    # Test case 2: Gemini spec model
    model = "gemini/ft-uuid-123"
    result = VertexGeminiConfig._get_model_name_from_gemini_spec_model(model)
    assert result == "ft-uuid-123"
