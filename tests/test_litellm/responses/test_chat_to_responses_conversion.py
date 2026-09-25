from unittest.mock import Mock

import pytest

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)


@pytest.mark.parametrize(
    ("content_block", "expected_content"),
    [
        (
            {"type": "text", "text": "Stable prefix"},
            {"type": "input_text", "text": "Stable prefix"},
        ),
        (
            {"type": "image_url", "image_url": "https://example.com/image.png"},
            {
                "type": "input_image",
                "image_url": "https://example.com/image.png",
                "detail": "auto",
            },
        ),
        (
            {"type": "file", "file": {"file_id": "file-123"}},
            {"type": "input_file", "file_id": "file-123"},
        ),
    ],
    ids=("text", "image_url", "file"),
)
def test_prompt_cache_breakpoint_survives_chat_to_responses_conversion(
    content_block: dict[str, object], expected_content: dict[str, object]
) -> None:
    cache_breakpoint = {"mode": "explicit"}
    request = LiteLLMResponsesTransformationHandler().transform_request(
        model="gpt-5.6-sol",
        messages=[
            {
                "role": "user",
                "content": [{**content_block, "prompt_cache_breakpoint": cache_breakpoint}],
            }
        ],
        optional_params={"prompt_cache_options": cache_breakpoint},
        litellm_params={},
        headers={},
        litellm_logging_obj=Mock(),
    )

    assert request["input"][0] == {
        "type": "message",
        "role": "user",
        "content": [{**expected_content, "prompt_cache_breakpoint": cache_breakpoint}],
    }
    assert request["prompt_cache_options"] == cache_breakpoint
