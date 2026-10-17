from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import TOOL_RESULT_IMAGE_PLACEHOLDER
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


def _nested_image_message() -> MessageBlock:
    return MessageBlock(
        role="user",
        content=[
            ContentBlock(
                toolResult=ToolResultBlock(
                    toolUseId="tooluse_nested",
                    content=[ToolResultContentBlock(text="nested"), ToolResultContentBlock(image=_TOOL_IMAGE)],
                ),
            ),
        ],
    )


def _keeps_image_nested(placed: tuple[MessageBlock, ...], message: MessageBlock) -> bool:
    if placed[0] is message:
        return True
    tool_result: Final = placed[0]["content"][0]["toolResult"]
    assert isinstance(tool_result, dict)
    assert tool_result["content"] == [{"text": "nested"}]
    assert placed[0]["content"][1] == {"image": _TOOL_IMAGE}
    return False


@pytest.mark.parametrize(
    ("model", "nested"),
    [
        ("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
        ("bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("bedrock/us.amazon.nova-pro-v1:0", True),
        ("bedrock/converse/us.amazon.nova-2-lite-v1:0", True),
        ("bedrock/global.openai.gpt-6.1-sol", False),
        ("bedrock/qwen.qwen3-vl-235b-a22b", False),
        ("bedrock/global.moonshotai.kimi-k3", False),
        ("bedrock/mistral.mistral-large-3-675b-instruct", False),
        ("bedrock/converse/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123", False),
    ],
)
def test_place_tool_result_images_nests_only_for_claude_and_nova(model: str, nested: bool):
    message: Final = _nested_image_message()
    assert _keeps_image_nested(place_tool_result_images([message], model), message) is nested


@pytest.mark.parametrize(
    ("price_map_key", "flag", "nested"),
    [
        ("qwen.qwen3-vl-235b-a22b", True, True),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", False, False),
    ],
)
def test_price_map_flag_overrides_the_model_family(
    monkeypatch: pytest.MonkeyPatch, price_map_key: str, flag: bool, nested: bool
):
    entry: Final = {**litellm.model_cost.get(price_map_key, {}), "supports_bedrock_converse_tool_result_images": flag}
    monkeypatch.setitem(litellm.model_cost, price_map_key, entry)
    message: Final = _nested_image_message()
    placed: Final = place_tool_result_images([message], f"bedrock/{price_map_key}")
    assert _keeps_image_nested(placed, message) is nested


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
    assert tool_result["content"] == [{"text": TOOL_RESULT_IMAGE_PLACEHOLDER}]
    image: Final = placed[0]["content"][1]["image"]
    assert isinstance(image, dict)
    assert image["format"] == "png"
