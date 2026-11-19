from typing import Final

import pytest

from litellm.responses.litellm_completion_transformation.mcp_calls import mcp_calls_by_call_id

_CALL_ID: Final = "mcptoolu_01WDx6ixKDtjRVLLUhL652Xa"


def _assistant_message(result_content: object) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "The wiki has two pages.",
        "provider_specific_fields": {
            "server_tool_calls": [
                {
                    "id": _CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "read_wiki_structure",
                        "arguments": '{"repoName": "facebook/react"}',
                        "provider_specific_fields": {"server_name": "deepwiki"},
                    },
                }
            ],
            "tool_results": [
                {"type": "mcp_tool_result", "tool_use_id": _CALL_ID, "is_error": False, "content": result_content}
            ],
        },
    }


@pytest.mark.parametrize(
    ("result_content", "expected_output"),
    [
        pytest.param("Available pages for facebook/react", "Available pages for facebook/react", id="string_content"),
        pytest.param(
            [
                {"type": "text", "text": "1 React Repository Overview"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/diagram.png"}},
                {"type": "text", "text": "2 Core Reconciler Architecture"},
            ],
            "1 React Repository Overview\n2 Core Reconciler Architecture",
            id="text_blocks_joined_and_other_blocks_skipped",
        ),
    ],
)
def test_an_mcp_tool_result_fills_the_mcp_call_output_with_its_text(
    result_content: object, expected_output: str
) -> None:
    call: Final = mcp_calls_by_call_id((_assistant_message(result_content),))[_CALL_ID]

    assert (call.id, call.status, call.output, call.error) == (f"mcp_{_CALL_ID}", "completed", expected_output, None)
