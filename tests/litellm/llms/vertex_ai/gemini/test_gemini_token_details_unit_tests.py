from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.utils import Usage, PromptTokensDetailsWrapper, CompletionTokensDetailsWrapper
from tests.litellm.llms.vertex_ai.gemini.gemini_token_details_test_utils import (
    get_text_tokens_test_data,
    get_audio_tokens_test_data,
    get_image_tokens_test_data,
    get_all_token_types_test_data,
    get_cached_tokens_test_data,
    get_streaming_chunk_test_data,
    get_cached_response_test_data,
    assert_token_details,
    run_usage_metadata_test,
)

def test_vertex_ai_usage_metadata_cached_tokens():
    """Test that cached tokens are properly reported in the usage metadata"""
    run_usage_metadata_test(get_cached_tokens_test_data)

def test_vertex_ai_usage_metadata_text_tokens():
    """Test that text tokens are properly reported in the usage metadata"""
    run_usage_metadata_test(get_text_tokens_test_data)


def test_vertex_ai_usage_metadata_audio_tokens():
    """Test that audio tokens are properly reported in the usage metadata"""
    run_usage_metadata_test(get_audio_tokens_test_data)

def test_vertex_ai_usage_metadata_image_tokens():
    """Test that image tokens are properly reported in the usage metadata"""
    run_usage_metadata_test(get_image_tokens_test_data)

def test_vertex_ai_usage_metadata_all_token_types():
    """Test that all token types are properly reported in the usage metadata"""
    run_usage_metadata_test(get_all_token_types_test_data)

def test_streaming_chunk_includes_all_token_types():
    """Test that streaming chunks include all token types"""
    data = get_streaming_chunk_test_data()

    # Create a VertexGeminiConfig instance
    v = VertexGeminiConfig()

    # Calculate usage directly using the _calculate_usage method
    usage = v._calculate_usage(completion_response={"usageMetadata": data["chunk"]["usageMetadata"]})

    # Verify that the usage has the correct information
    assert usage is not None
    assert_token_details(usage, data)

def test_cached_response_includes_all_token_types():
    """Test that cached responses include all token types"""
    from litellm.types.utils import ModelResponse

    data = get_cached_response_test_data()

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
            prompt_tokens=data["expected_prompt_tokens"],
            completion_tokens=data["expected_completion_tokens"],
            total_tokens=data["expected_total_tokens"],
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=data["cached_tokens"],
                text_tokens=data["expected_cached_text_tokens"],
                audio_tokens=data["expected_cached_audio_tokens"],
                image_tokens=data["expected_cached_image_tokens"]
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                text_tokens=data["expected_completion_tokens"]
            )
        ),
        custom_llm_provider="cached_response"
    )

    # Verify that the response has the correct usage information
    assert response.usage is not None
    assert_token_details(response.usage, data)
