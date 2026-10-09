from typing import Final

from litellm.llms.bedrock.chat.tool_result_images import place_tool_result_images
from litellm.types.llms.bedrock import (
    ContentBlock,
    ImageBlock,
    MessageBlock,
    ToolResultBlock,
    ToolResultContentBlock,
)

_PNG_BYTES: Final = b"\x89PNG\r\n\x1a\n"
_TOOL_IMAGE: Final = ImageBlock(format="png", source={"bytes": _PNG_BYTES})


def test_place_tool_result_images_keeps_nested_images_for_claude():
    message: Final = MessageBlock(
        role="user",
        content=[
            ContentBlock(
                toolResult=ToolResultBlock(
                    toolUseId="tooluse_nested",
                    content=[
                        ToolResultContentBlock(text="nested", image=_TOOL_IMAGE),
                    ],
                ),
            ),
        ],
    )
    placed: Final = place_tool_result_images(
        [message],
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    assert placed == (message,)
    assert placed[0] is message


def test_place_tool_result_images_leaves_text_only_tool_results_on_gpt():
    message: Final = MessageBlock(
        role="user",
        content=[
            ContentBlock(text="plain user text"),
            ContentBlock(
                toolResult=ToolResultBlock(
                    toolUseId="tooluse_text",
                    content=[ToolResultContentBlock(text="only text")],
                ),
            ),
        ],
    )
    placed: Final = place_tool_result_images([message], "bedrock/global.openai.gpt-6.1-sol")
    assert placed[0] is message


def test_place_tool_result_images_keeps_text_when_image_is_a_separate_part():
    message: Final = MessageBlock(
        role="user",
        content=[
            ContentBlock(
                toolResult=ToolResultBlock(
                    toolUseId="tooluse_split",
                    content=[
                        ToolResultContentBlock(text="failed"),
                        ToolResultContentBlock(image=_TOOL_IMAGE),
                    ],
                ),
            ),
        ],
    )
    placed: Final = place_tool_result_images([message], "bedrock/global.openai.gpt-6.1-sol")
    tool_result: Final = placed[0]["content"][0]["toolResult"]
    assert isinstance(tool_result, dict)
    assert tool_result["content"] == [{"text": "failed"}]
    assert "status" not in tool_result


def test_place_tool_result_images_preserves_tool_result_status_on_gpt():
    message: Final = MessageBlock(
        role="user",
        content=[
            ContentBlock(
                toolResult=ToolResultBlock(
                    toolUseId="tooluse_status",
                    status="error",
                    content=[ToolResultContentBlock(image=_TOOL_IMAGE)],
                ),
            ),
        ],
    )
    placed: Final = place_tool_result_images([message], "bedrock/global.openai.gpt-6.1-sol")
    tool_result: Final = placed[0]["content"][0]["toolResult"]
    assert isinstance(tool_result, dict)
    assert tool_result["status"] == "error"
    assert tool_result["content"] == [{"text": "Image attached."}]
    image: Final = placed[0]["content"][1]["image"]
    assert isinstance(image, dict)
    assert image["format"] == "png"
