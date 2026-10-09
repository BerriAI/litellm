"""Place tool-result images where Bedrock Converse will accept them."""

from collections.abc import Iterator, Sequence
from typing import Final

from litellm.llms.bedrock.common_utils import bedrock_converse_supports_tool_result_images
from litellm.types.llms.bedrock import ContentBlock, ImageBlock, MessageBlock, ToolResultBlock, ToolResultContentBlock

_IMAGE_ONLY_NOTE: Final = "Image attached."


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
        tool_result: Final = block.get("toolResult")
        if tool_result is None:
            yield block
            continue
        parts: Final = tuple(tool_result.get("content") or ())
        images: Final = tuple(_images(parts))
        if not images:
            yield block
            continue
        kept: Final = tuple(_without_images(parts)) or (ToolResultContentBlock(text=_IMAGE_ONLY_NOTE),)
        yield ContentBlock(toolResult=_tool_result_without_images(tool_result, kept))
        yield from (ContentBlock(image=image) for image in images)


def _images(parts: Sequence[ToolResultContentBlock]) -> Iterator[ImageBlock]:
    for part in parts:
        image: Final = part.get("image")
        if image is not None:
            yield image


def _without_images(parts: Sequence[ToolResultContentBlock]) -> Iterator[ToolResultContentBlock]:
    for part in parts:
        if part.get("image") is None:
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
