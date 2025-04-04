import pytest
import asyncio
from unittest.mock import MagicMock
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
import litellm
from litellm import ModelResponse

@pytest.mark.asyncio
async def test_transform_response_with_avglogprobs():
    """
    Test that the transform_response method correctly handles the avgLogprobs key
    from Gemini Flash 2.0 responses.
    """
    # Create a mock response with avgLogprobs
    response_json = {
        "candidates": [{
            "content": {"parts": [{"text": "Test response"}], "role": "model"},
            "finishReason": "STOP",
            "avgLogprobs": -0.3445799010140555
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
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
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
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
        encoding=None
    )
    
    # Assert that the avgLogprobs was correctly added to the model response
    assert len(transformed_response.choices) == 1
    assert transformed_response.choices[0].logprobs == -0.3445799010140555
