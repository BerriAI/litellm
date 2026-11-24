from typing import Final

from litellm.llms.bedrock.chat.tool_use_ids import with_mistral_tool_use_ids
from litellm.types.llms.bedrock import ContentBlock, MessageBlock, ToolResultBlock, ToolResultContentBlock


def test_mistral_tool_result_keeps_its_status_and_content_with_the_mapped_id():
    message: Final = MessageBlock(
        role="user",
        content=[
            ContentBlock(
                toolResult=ToolResultBlock(
                    toolUseId="call_abc123", content=[ToolResultContentBlock(text="ok")], status="success"
                )
            ),
            ContentBlock(text="Sum it up."),
        ],
    )

    mapped: Final = with_mistral_tool_use_ids([message], "mistral.ministral-3-14b-instruct")

    tool_result: Final = mapped[0]["content"][0]["toolResult"]
    assert (tool_result["status"], tool_result["content"]) == ("success", [ToolResultContentBlock(text="ok")])
    assert tool_result["toolUseId"] != "call_abc123"
    assert mapped[0]["content"][1] == ContentBlock(text="Sum it up.")
