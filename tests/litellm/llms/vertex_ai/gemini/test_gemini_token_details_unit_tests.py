import pytest
import litellm
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.vertex_ai import UsageMetadata
from litellm.types.utils import Usage, PromptTokensDetailsWrapper, CompletionTokensDetailsWrapper

def test_vertex_ai_usage_metadata_cached_tokens():
    """Test that cached tokens are properly reported in the usage metadata"""
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 57,
        "candidatesTokenCount": 74,
        "totalTokenCount": 131,
        "cachedContentTokenCount": 21,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 57}],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})

    assert result.prompt_tokens == 57
    assert result.completion_tokens == 74
    assert result.total_tokens == 131
    assert result.prompt_tokens_details.cached_tokens == 21
    assert result.prompt_tokens_details.text_tokens == 57
    assert result.prompt_tokens_details.audio_tokens is None
    assert result.prompt_tokens_details.image_tokens is None

def test_vertex_ai_usage_metadata_text_tokens():
    """Test that text tokens are properly reported in the usage metadata"""
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 57,
        "candidatesTokenCount": 74,
        "totalTokenCount": 131,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 57}],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})

    assert result.prompt_tokens == 57
    assert result.completion_tokens == 74
    assert result.total_tokens == 131
    assert result.prompt_tokens_details.text_tokens == 57
    assert result.prompt_tokens_details.cached_tokens is None
    assert result.prompt_tokens_details.audio_tokens is None
    assert result.prompt_tokens_details.image_tokens is None

def test_vertex_ai_usage_metadata_audio_tokens():
    """Test that audio tokens are properly reported in the usage metadata"""
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 74,
        "totalTokenCount": 174,
        "promptTokensDetails": [
            {"modality": "TEXT", "tokenCount": 57},
            {"modality": "AUDIO", "tokenCount": 43}
        ],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})

    assert result.prompt_tokens == 100
    assert result.completion_tokens == 74
    assert result.total_tokens == 174
    assert result.prompt_tokens_details.text_tokens == 57
    assert result.prompt_tokens_details.audio_tokens == 43
    assert result.prompt_tokens_details.cached_tokens is None
    assert result.prompt_tokens_details.image_tokens is None

def test_vertex_ai_usage_metadata_image_tokens():
    """Test that image tokens are properly reported in the usage metadata"""
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 74,
        "totalTokenCount": 174,
        "promptTokensDetails": [
            {"modality": "TEXT", "tokenCount": 57},
            {"modality": "IMAGE", "tokenCount": 43}
        ],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})

    assert result.prompt_tokens == 100
    assert result.completion_tokens == 74
    assert result.total_tokens == 174
    assert result.prompt_tokens_details.text_tokens == 57
    assert result.prompt_tokens_details.image_tokens == 43
    assert result.prompt_tokens_details.cached_tokens is None
    assert result.prompt_tokens_details.audio_tokens is None

def test_vertex_ai_usage_metadata_all_token_types():
    """Test that all token types are properly reported in the usage metadata"""
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 150,
        "candidatesTokenCount": 74,
        "totalTokenCount": 224,
        "cachedContentTokenCount": 30,
        "promptTokensDetails": [
            {"modality": "TEXT", "tokenCount": 57},
            {"modality": "AUDIO", "tokenCount": 43},
            {"modality": "IMAGE", "tokenCount": 50}
        ],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})

    assert result.prompt_tokens == 150
    assert result.completion_tokens == 74
    assert result.total_tokens == 224
    assert result.prompt_tokens_details.text_tokens == 57
    assert result.prompt_tokens_details.audio_tokens == 43
    assert result.prompt_tokens_details.image_tokens == 50
    assert result.prompt_tokens_details.cached_tokens == 30

def test_streaming_chunk_includes_all_token_types():
    """Test that streaming chunks include all token types"""
    # Create a VertexGeminiConfig instance
    v = VertexGeminiConfig()

    # Simulate a streaming chunk as would be received from Gemini
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        "usageMetadata": {
            "promptTokenCount": 150,
            "candidatesTokenCount": 74,
            "totalTokenCount": 224,
            "cachedContentTokenCount": 30,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 57},
                {"modality": "AUDIO", "tokenCount": 43},
                {"modality": "IMAGE", "tokenCount": 50}
            ],
        },
    }

    # Calculate usage directly using the _calculate_usage method
    usage = v._calculate_usage(completion_response={"usageMetadata": chunk["usageMetadata"]})

    # Verify that the usage has the correct information
    assert usage is not None
    assert usage.prompt_tokens == 150
    assert usage.completion_tokens == 74
    assert usage.total_tokens == 224
    assert usage.prompt_tokens_details.cached_tokens == 30
    assert usage.prompt_tokens_details.text_tokens == 57
    assert usage.prompt_tokens_details.audio_tokens == 43
    assert usage.prompt_tokens_details.image_tokens == 50

def test_cached_response_includes_all_token_types():
    """Test that cached responses include all token types"""
    from litellm.types.utils import ModelResponse

    # Create a ModelResponse with usage information
    response = ModelResponse(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="gemini-1.5-pro",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response"
                },
                "finish_reason": "stop"
            }
        ],
        usage=Usage(
            prompt_tokens=150,
            completion_tokens=74,
            total_tokens=224,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=30,
                text_tokens=57,
                audio_tokens=43,
                image_tokens=50
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                text_tokens=74
            )
        ),
        custom_llm_provider="cached_response"
    )

    # Verify that the response has the correct usage information
    assert response.usage is not None
    assert response.usage.prompt_tokens == 150
    assert response.usage.completion_tokens == 74
    assert response.usage.total_tokens == 224
    assert response.usage.prompt_tokens_details.cached_tokens == 30
    assert response.usage.prompt_tokens_details.text_tokens == 57
    assert response.usage.prompt_tokens_details.audio_tokens == 43
    assert response.usage.prompt_tokens_details.image_tokens == 50
