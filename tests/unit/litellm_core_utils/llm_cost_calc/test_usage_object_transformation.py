from litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation import (
    InteractionsUsageObjectTransformation,
)
from litellm.types.utils import Usage

OMNI_VIDEO_USAGE = {
    "total_tokens": 18247,
    "total_input_tokens": 16,
    "input_tokens_by_modality": [{"modality": "text", "tokens": 16}],
    "total_cached_tokens": 0,
    "total_output_tokens": 17937,
    "output_tokens_by_modality": [{"modality": "video", "tokens": 17376}],
    "total_tool_use_tokens": 0,
    "total_thought_tokens": 294,
}


def test_detects_interactions_usage_object():
    assert InteractionsUsageObjectTransformation.is_interactions_usage_object(OMNI_VIDEO_USAGE) is True


def test_rejects_chat_and_responses_api_usage_objects():
    chat_usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    responses_api_usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    assert InteractionsUsageObjectTransformation.is_interactions_usage_object(chat_usage) is False
    assert InteractionsUsageObjectTransformation.is_interactions_usage_object(responses_api_usage) is False
    assert InteractionsUsageObjectTransformation.is_interactions_usage_object(None) is False
    assert InteractionsUsageObjectTransformation.is_interactions_usage_object("usage") is False


def test_transforms_real_omni_video_usage_block():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(OMNI_VIDEO_USAGE)

    assert isinstance(usage, Usage)
    assert usage.prompt_tokens == 16
    assert usage.completion_tokens == 17937 + 294
    assert usage.total_tokens == 18247
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.text_tokens == 16
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.video_tokens == 17376
    assert usage.completion_tokens_details.reasoning_tokens == 294


def test_transforms_reasoning_tokens_spec_field_name():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
        {
            "total_input_tokens": 10,
            "total_output_tokens": 20,
            "total_reasoning_tokens": 5,
        }
    )
    assert usage.completion_tokens == 25
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 5
    assert usage.total_tokens == 35


def test_cached_tokens_subtracted_from_text_input():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
        {
            "total_input_tokens": 1000,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 1000}],
            "total_cached_tokens": 400,
            "total_output_tokens": 50,
        }
    )
    assert usage.prompt_tokens == 1000
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.text_tokens == 600
    assert usage.prompt_tokens_details.cached_tokens == 400
    assert usage._cache_read_input_tokens == 400


def test_cached_tokens_subtracted_per_modality_when_breakdown_present():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
        {
            "total_input_tokens": 1500,
            "input_tokens_by_modality": [
                {"modality": "text", "tokens": 1000},
                {"modality": "audio", "tokens": 500},
            ],
            "total_cached_tokens": 300,
            "cached_tokens_by_modality": [{"modality": "audio", "tokens": 300}],
            "total_output_tokens": 50,
        }
    )
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.text_tokens == 1000
    assert usage.prompt_tokens_details.audio_tokens == 200
    assert usage.prompt_tokens_details.cached_tokens == 300


def test_tool_use_tokens_billed_as_input():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
        {
            "total_input_tokens": 100,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 100}],
            "total_tool_use_tokens": 40,
            "tool_use_tokens_by_modality": [{"modality": "text", "tokens": 40}],
            "total_output_tokens": 10,
        }
    )
    assert usage.prompt_tokens == 140
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.text_tokens == 140


def test_google_search_grounding_count_maps_to_web_search_requests():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
        {
            "total_input_tokens": 103,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 103}],
            "total_output_tokens": 226,
            "total_thought_tokens": 351,
            "grounding_tool_count": [
                {"type": "google_search", "count": 3},
                {"type": "url_context", "count": 2},
            ],
        }
    )
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.web_search_requests == 3


def test_no_grounding_leaves_web_search_requests_unset():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
        {
            "total_input_tokens": 10,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 10}],
            "total_output_tokens": 5,
        }
    )
    assert usage.prompt_tokens_details is not None
    assert getattr(usage.prompt_tokens_details, "web_search_requests", None) is None


def test_document_modality_folds_into_text():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_object(
        {
            "total_input_tokens": 80,
            "input_tokens_by_modality": [
                {"modality": "text", "tokens": 30},
                {"modality": "document", "tokens": 50},
            ],
            "total_output_tokens": 10,
        }
    )
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.text_tokens == 80
