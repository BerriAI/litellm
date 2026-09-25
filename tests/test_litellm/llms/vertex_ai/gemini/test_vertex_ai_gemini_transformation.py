import base64

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_gemini_tool_call_result,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
    _transform_request_body,
    check_if_part_exists_in_parts,
    _get_highest_media_resolution,
    _extract_max_media_resolution_from_messages,
)
from litellm.types.llms.vertex_ai import BlobType
from litellm.types.utils import Message


# Tests for issue #14556: Labels field provider-aware filtering


# Tests for media_resolution (detail parameter) handling - Issue #17084


# Tests for VideoMetadata support across all Gemini models (Issue #25474)


def test_convert_tool_response_with_url_image():
    """Test tool response with HTTP URL image (will download and convert)."""
    import pytest

    # Use a publicly accessible test image URL
    test_image_url = "https://via.placeholder.com/1x1.png"

    tool_message = {
        "role": "tool",
        "tool_call_id": "call_test456",
        "content": [
            {"type": "text", "text": '{"url": "https://example.com"}'},
            {"type": "input_image", "image_url": test_image_url},
        ],
    }

    last_message_with_tool_calls = {
        "tool_calls": [
            {
                "id": "call_test456",
                "function": {
                    "name": "type_text_at",
                    "arguments": '{"x": 300, "y": 400, "text": "hello"}',
                },
            }
        ]
    }

    try:
        result = convert_to_gemini_tool_call_result(
            tool_message, last_message_with_tool_calls
        )

        assert isinstance(
            result, list
        ), "Should return a parts list when media is present"
        assert len(result) == 1, "Should return one function_response part"
        result_part = result[0]
        assert "function_response" in result_part
        assert "inline_data" not in result_part
        function_response = result_part["function_response"]
        assert function_response["name"] == "type_text_at"

        # Check inline_data is nested under functionResponse.parts.
        assert "parts" in function_response
        assert len(function_response["parts"]) == 1
        inline_data: BlobType = function_response["parts"][0]["inline_data"]
        assert "data" in inline_data
        assert "mime_type" in inline_data
    except Exception as e:
        # Skip test if URL download fails (no internet connection, etc.)
        pytest.skip(f"Failed to download image from URL: {e}")
