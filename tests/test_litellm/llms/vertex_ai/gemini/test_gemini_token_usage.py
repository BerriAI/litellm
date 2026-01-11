
import sys, os
import pytest
sys.path.insert(0, os.path.abspath('../../../../../'))

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.vertex_ai import UsageMetadata

def test_gemini_3_flash_preview_token_usage_fallback():
    """Test fallback logic when candidatesTokensDetails is missing (e.g. Gemini 3 Flash Preview)."""
    v = VertexGeminiConfig()
    
    usage_metadata_dict = {
        "promptTokenCount": 2145,
        "candidatesTokenCount": 509,
        "totalTokenCount": 2654,
        # candidatesTokensDetails intentionally omitted
    }
    
    completion_response = {"usageMetadata": usage_metadata_dict}
    result = v._calculate_usage(completion_response=completion_response)
    
    assert result.completion_tokens == 509
    assert result.prompt_tokens == 2145
    assert result.total_tokens == 2654
    
    # Text tokens should be derived from candidatesTokenCount
    assert result.completion_tokens_details is not None
    assert result.completion_tokens_details.text_tokens == 509
    assert result.completion_tokens_details.image_tokens is None
    assert result.completion_tokens_details.audio_tokens is None

def test_gemini_no_reasoning_fallback():
    """Test fallback when reasoning_effort is absent and details are missing."""
    v = VertexGeminiConfig()
    
    usage_metadata_dict = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 264,
        "totalTokenCount": 364,
    }
    
    completion_response = {"usageMetadata": usage_metadata_dict}
    result = v._calculate_usage(completion_response=completion_response)
    
    assert result.completion_tokens == 264
    assert result.completion_tokens_details is not None
    assert result.completion_tokens_details.text_tokens == 264
    assert result.completion_tokens_details.reasoning_tokens is None or result.completion_tokens_details.reasoning_tokens == 0

def test_gemini_token_usage_standard_response():
    """Verify that standard responses with details are computed correctly and not overwritten."""
    v = VertexGeminiConfig()
    
    usage_metadata_dict = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
        "totalTokenCount": 150,
        "candidatesTokensDetails": [
            {"modality": "TEXT", "tokenCount": 40},
            {"modality": "IMAGE", "tokenCount": 10}
        ]
    }
    
    completion_response = {"usageMetadata": usage_metadata_dict}
    result = v._calculate_usage(completion_response=completion_response)
    
    assert result.completion_tokens == 50
    assert result.completion_tokens_details.text_tokens == 40
    assert result.completion_tokens_details.image_tokens == 10
