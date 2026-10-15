from typing import Final

from .beta_detection_models import (
    BetaContentBlock,
    BetaDetectionRequest,
    BetaMessage,
    BetaThinking,
    BetaThinkingDisplay,
    BetaThinkingType,
    BetaToolTarget,
    BetaToolTargetType,
)


def thinking_type(request: BetaDetectionRequest) -> BetaThinkingType | None:
    if request.thinking is None or not isinstance(request.thinking.root, BetaThinking):
        return None
    return request.thinking.root.type


def thinking_display(request: BetaDetectionRequest) -> BetaThinkingDisplay | None:
    if request.thinking is None or not isinstance(request.thinking.root, BetaThinking):
        return None
    display: Final = request.thinking.root.display
    return display.root if display is not None and isinstance(display.root, BetaThinkingDisplay) else None


def tool_type(block: BetaContentBlock) -> BetaToolTargetType | None:
    if block.tool is None or not isinstance(block.tool.root, BetaToolTarget):
        return None
    return block.tool.root.type


def content_blocks(message: BetaMessage) -> tuple[BetaContentBlock, ...]:
    if isinstance(message.content, str):
        return ()
    return tuple(value.root for value in message.content if isinstance(value.root, BetaContentBlock))
