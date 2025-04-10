import asyncio
from unittest.mock import MagicMock

import pytest

import litellm
from litellm import ModelResponse
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.utils import ChoiceLogprobs


@pytest.mark.asyncio
async def test_transform_response_with_avglogprobs():
    """
    Test that the transform_response method correctly handles the avgLogprobs key
    from Gemini Flash 2.0 responses.
    """
    # Create a mock response with avgLogprobs
    response_json = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Test response"}], "role": "model"},
                "finishReason": "STOP",
                "avgLogprobs": -0.3445799010140555,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }

    # Create a mock HTTP response
    mock_response = MagicMock()
    mock_response.json.return_value = response_json

    # Create a mock logging object
    mock_logging = MagicMock()

    # Create an instance of VertexGeminiConfig
    config = VertexGeminiConfig()

    # Create a ModelResponse object
    model_response = ModelResponse(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gemini-2.0-flash",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    # Call the transform_response method
    transformed_response = config.transform_response(
        model="gemini-2.0-flash",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    # Assert that the avgLogprobs was correctly added to the model response
    assert len(transformed_response.choices) == 1
    assert isinstance(transformed_response.choices[0].logprobs, ChoiceLogprobs)
    assert transformed_response.choices[0].logprobs == -0.3445799010140555


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
