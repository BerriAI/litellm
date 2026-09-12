import httpx
from unittest.mock import MagicMock
import pytest
from litellm import ModelResponse, Choices, Message, Usage
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
    ModelResponseIterator,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_process_candidates_with_no_image_finish_reason_and_no_content():
    """
    Issue #40477: Verify that a candidate with finishReason="NO_IMAGE" and no content
    is not dropped, finish_reason is mapped to "content_filter", and raw finish reason
    is preserved in provider_specific_fields.
    """
    candidates = [
        {
            "finishReason": "NO_IMAGE",
            "index": 0,
            "safetyRatings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "probability": "NEGLIGIBLE",
                }
            ],
        }
    ]
    model_response = ModelResponse(choices=[])

    VertexGeminiConfig._process_candidates(
        _candidates=candidates,
        model_response=model_response,
        standard_optional_params={},
        cumulative_tool_call_index=0,
    )

    assert len(model_response.choices) == 1
    choice = model_response.choices[0]
    assert choice.finish_reason == "content_filter"
    assert choice.message.content is None
    assert choice.message.role == "assistant"
    assert choice.provider_specific_fields is not None
    assert choice.provider_specific_fields.get("finish_reason") == "NO_IMAGE"
    assert choice.provider_specific_fields.get("gemini_finish_reason") == "NO_IMAGE"


def test_transform_google_generate_content_to_openai_model_response_contentless_candidate():
    """
    Verify full model response transformation when Gemini returns a candidate
    without content (e.g. finishReason='NO_IMAGE').
    """
    raw_response_dict = {
        "candidates": [
            {
                "finishReason": "NO_IMAGE",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 0,
            "totalTokenCount": 15,
        },
    }
    raw_response = httpx.Response(
        200,
        json=raw_response_dict,
        request=httpx.Request("POST", "https://example.com"),
    )
    model_response = ModelResponse()

    config = VertexGeminiConfig()
    logging_mock = MagicMock()
    logging_mock.optional_params = {}

    result = config._transform_google_generate_content_to_openai_model_response(
        model_response=model_response,
        completion_response=raw_response_dict,
        logging_obj=logging_mock,
        raw_response=raw_response,
        model="gemini-2.5-flash-image",
    )

    assert len(result.choices) == 1
    assert result.choices[0].finish_reason == "content_filter"
    assert result.choices[0].message.content is None
    assert result._hidden_params.get("provider_specific_fields", {}).get("finish_reason") == "NO_IMAGE"
    assert result._hidden_params.get("provider_specific_fields", {}).get("gemini_finish_reason") == "NO_IMAGE"


def test_process_candidates_with_other_contentless_finish_reasons():
    """
    Verify other finish reasons like MALFORMED_FUNCTION_CALL and OTHER without content
    are retained with content=None and their raw finish reason preserved.
    """
    candidates = [
        {
            "finishReason": "MALFORMED_FUNCTION_CALL",
            "index": 0,
        }
    ]
    model_response = ModelResponse(choices=[])

    VertexGeminiConfig._process_candidates(
        _candidates=candidates,
        model_response=model_response,
        standard_optional_params={},
        cumulative_tool_call_index=0,
    )

    assert len(model_response.choices) == 1
    choice = model_response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content is None
    assert choice.provider_specific_fields.get("finish_reason") == "MALFORMED_FUNCTION_CALL"


def test_anthropic_adapter_content_filter_and_refusal():
    """
    Verify Anthropic message adapter maps content_filter and refusal to 'refusal'.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()
    assert (
        adapter._translate_openai_finish_reason_to_anthropic("content_filter")
        == "refusal"
    )
    assert (
        adapter._translate_openai_finish_reason_to_anthropic("refusal")
        == "refusal"
    )

    openai_response = ModelResponse(
        id="chatcmpl-test-content-filter",
        choices=[
            Choices(
                finish_reason="content_filter",
                index=0,
                message=Message(content=None, role="assistant"),
            )
        ],
        model="gemini-2.5-flash-image",
        usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
    )

    anthropic_response = adapter.translate_openai_response_to_anthropic(
        openai_response
    )

    assert anthropic_response["stop_reason"] == "refusal"
    assert anthropic_response["content"] == []


def test_responses_api_content_filter_incomplete_details():
    """
    Verify Responses API transformation populates incomplete_details with reason='content_filter'
    when OpenAI finish_reason is content_filter or refusal.
    """
    openai_response = ModelResponse(
        id="chatcmpl-test-responses-incomplete",
        choices=[
            Choices(
                finish_reason="content_filter",
                index=0,
                message=Message(content=None, role="assistant"),
            )
        ],
        model="gemini-2.5-flash-image",
        created=123456789,
    )

    responses_res = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test",
        responses_api_request={},
        chat_completion_response=openai_response,
    )

    assert responses_res.status == "incomplete"
    assert responses_res.incomplete_details is not None
    assert responses_res.incomplete_details.reason == "content_filter"


def test_streaming_contentless_candidate_iterator():
    """
    Verify streaming iterator handles chunks with finishReason="NO_IMAGE" and no content.
    """
    logging_mock = MagicMock()
    logging_mock.optional_params = {}

    iterator = ModelResponseIterator(
        streaming_response=[],
        sync_stream=True,
        logging_obj=logging_mock,
    )

    chunk = {
        "candidates": [
            {
                "finishReason": "NO_IMAGE",
                "index": 0,
            }
        ]
    }

    result = iterator.chunk_parser(chunk)
    assert result is not None
    assert len(result.choices) == 1
    assert result.choices[0].finish_reason == "content_filter"
    assert result.choices[0].delta.content is None
    assert result.choices[0].delta.provider_specific_fields is not None
    assert result.choices[0].delta.provider_specific_fields.get("finish_reason") == "NO_IMAGE"
