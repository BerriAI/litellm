from litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation import (
    InteractionsUsageObjectTransformation,
)


def _omni_flash_video_usage() -> dict:
    return {
        "total_tokens": 18247,
        "total_input_tokens": 16,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 16}],
        "total_cached_tokens": 0,
        "total_output_tokens": 17937,
        "output_tokens_by_modality": [
            {"modality": "text", "tokens": 561},
            {"modality": "video", "tokens": 17376},
        ],
        "total_tool_use_tokens": 0,
        "total_thought_tokens": 294,
    }


def test_is_interactions_usage_dict():
    assert InteractionsUsageObjectTransformation.is_interactions_usage_dict(_omni_flash_video_usage())
    assert InteractionsUsageObjectTransformation.is_interactions_usage_dict({"total_output_tokens": 5})
    assert not InteractionsUsageObjectTransformation.is_interactions_usage_dict(
        {"prompt_tokens": 1, "completion_tokens": 2}
    )
    assert not InteractionsUsageObjectTransformation.is_interactions_usage_dict(
        {"input_tokens": 1, "output_tokens": 2}
    )


def test_transform_interactions_usage_maps_modalities():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_to_chat_usage(
        _omni_flash_video_usage()
    )
    assert usage.prompt_tokens == 16
    assert usage.completion_tokens == 17937 + 294
    assert usage.total_tokens == 18247
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.text_tokens == 16
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 294
    assert usage.completion_tokens_details.text_tokens == 561
    assert usage.completion_tokens_details.video_tokens == 17376


def test_transform_interactions_usage_reasoning_key_fallback():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_to_chat_usage(
        {"total_input_tokens": 10, "total_output_tokens": 20, "total_reasoning_tokens": 5}
    )
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 35
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 5


def test_transform_interactions_usage_cached_tokens():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_to_chat_usage(
        {"total_input_tokens": 100, "total_cached_tokens": 40, "total_output_tokens": 10}
    )
    assert usage.prompt_tokens == 100
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 40


def test_transform_interactions_usage_empty():
    usage = InteractionsUsageObjectTransformation.transform_interactions_usage_to_chat_usage(None)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
