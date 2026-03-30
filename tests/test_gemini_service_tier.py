import pytest
from unittest.mock import MagicMock
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig, ModelResponseIterator
from litellm.types.utils import ModelResponse

def test_gemini_service_tier_request_mapping():
    """Test that default service_tier is mapped to standard for Gemini API, case-insensitively."""
    messages = [{"role": "user", "content": "test"}]
    litellm_params = {}

    # Test default -> standard
    optional_params = {"service_tier": "default"}
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )
    assert result["serviceTier"] == "standard"

    # Test DEFAULT -> standard
    optional_params = {"service_tier": "DEFAULT"}
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )
    assert result["serviceTier"] == "standard"

    # Test flex -> flex
    optional_params = {"service_tier": "FLEX"}
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )
    assert result["serviceTier"] == "flex"

def test_gemini_service_tier_response_mapping():
    """Test that standard service_tier is mapped back to default for Gemini API, case-insensitively."""
    config = VertexGeminiConfig()
    raw_response = MagicMock()
    raw_response.headers = {"x-gemini-service-tier": "STANDARD"}
    
    logging_obj = MagicMock()
    logging_obj.custom_llm_provider = "gemini"

    completion_response = {
        "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
    }

    result = config._transform_google_generate_content_to_openai_model_response(
        completion_response=completion_response,
        model_response=ModelResponse(),
        model="gemini-pro",
        logging_obj=logging_obj,
        raw_response=raw_response
    )

    assert result.service_tier == "default"

    # Test with lowercase standard
    raw_response.headers = {"x-gemini-service-tier": "standard"}
    result = config._transform_google_generate_content_to_openai_model_response(
        completion_response=completion_response,
        model_response=ModelResponse(),
        model="gemini-pro",
        logging_obj=logging_obj,
        raw_response=raw_response
    )
    assert result.service_tier == "default"

    # Test with flex -> flex
    raw_response.headers = {"x-gemini-service-tier": "FLEX"}
    result = config._transform_google_generate_content_to_openai_model_response(
        completion_response=completion_response,
        model_response=ModelResponse(),
        model="gemini-pro",
        logging_obj=logging_obj,
        raw_response=raw_response
    )
    assert result.service_tier == "flex"

def test_gemini_service_tier_streaming_response_mapping():
    """Test streaming response mapping."""
    logging_obj = MagicMock()
    logging_obj.custom_llm_provider = "gemini"

    iterator = ModelResponseIterator(
        streaming_response=[],
        sync_stream=True,
        logging_obj=logging_obj,
        response_headers={"x-gemini-service-tier": "STANDARD"}
    )

    chunk = {
        "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
    }

    result = iterator.chunk_parser(chunk)
    assert result.service_tier == "default"
