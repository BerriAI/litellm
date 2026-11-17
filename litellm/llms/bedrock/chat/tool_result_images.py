"""Place tool-result images where Bedrock Converse will accept them."""

from collections.abc import Iterator, Sequence
from typing import Final

from litellm.litellm_core_utils.prompt_templates.common_utils import TOOL_RESULT_IMAGE_PLACEHOLDER
from litellm.llms.bedrock.common_utils import bedrock_converse_supports_tool_result_images
from litellm.types.llms.bedrock import ContentBlock, ImageBlock, MessageBlock, ToolResultBlock, ToolResultContentBlock


def place_tool_result_images(messages: Sequence[MessageBlock], model: str) -> tuple[MessageBlock, ...]:
    """Move images out of ``toolResult.content`` when this model rejects them there."""
    if bedrock_converse_supports_tool_result_images(model):
        return tuple(messages)
    return tuple(_message_with_sibling_images(message) for message in messages)


def _message_with_sibling_images(message: MessageBlock) -> MessageBlock:
    content: Final = tuple(message.get("content") or ())
    rewritten: Final = tuple(_blocks_with_sibling_images(content))
    if rewritten == content:
        return message
    return MessageBlock(role=message["role"], content=list(rewritten))


def _blocks_with_sibling_images(content: Sequence[ContentBlock]) -> Iterator[ContentBlock]:
    for block in content:
        tool_result = block.get("toolResult")
        if tool_result is None:
            yield block
            continue
        parts = tuple(tool_result.get("content") or ())
        images = tuple(_images(parts))
        if not images:
            yield block
            continue
        kept = tuple(_without_images(parts)) or (ToolResultContentBlock(text=TOOL_RESULT_IMAGE_PLACEHOLDER),)
        yield ContentBlock(toolResult=_tool_result_without_images(tool_result, kept))
        yield from (ContentBlock(image=image) for image in images)


def _images(parts: Sequence[ToolResultContentBlock]) -> Iterator[ImageBlock]:
    for part in parts:
        if "image" in part:
            yield part["image"]


def _without_images(parts: Sequence[ToolResultContentBlock]) -> Iterator[ToolResultContentBlock]:
    for part in parts:
        if "image" not in part:
            yield part


def _tool_result_without_images(
    tool_result: ToolResultBlock,
    kept: Sequence[ToolResultContentBlock],
) -> ToolResultBlock:
    content: Final = list(kept)
    tool_use_id: Final = tool_result["toolUseId"]
    status: Final = tool_result.get("status")
    if status is None:
        return ToolResultBlock(content=content, toolUseId=tool_use_id)
    return ToolResultBlock(content=content, toolUseId=tool_use_id, status=status)
