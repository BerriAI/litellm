from .beta_detection import content_blocks, thinking_display, thinking_type, tool_type
from .beta_detection_models import (
    BetaContentType,
    BetaDetectionRequest,
    BetaThinkingDisplay,
    BetaThinkingType,
    BetaToolTargetType,
)


def detect_anthropic_betas(request: BetaDetectionRequest) -> frozenset[str]:
    return frozenset(
        beta
        for used, beta in (
            (
                (
                    thinking_type(request)
                    in (
                        BetaThinkingType.enabled,
                        BetaThinkingType.adaptive,
                    )
                    and thinking_display(request) in (BetaThinkingDisplay.updates,)
                ),
                "thinking-display-updates-2026-08-18",
            ),
            (
                any(
                    (
                        message.role in ("system",)
                        and any(
                            (
                                block.type
                                in (
                                    BetaContentType.tool_addition,
                                    BetaContentType.tool_removal,
                                )
                                and tool_type(block) in (BetaToolTargetType.tool_reference,)
                            )
                            for block in content_blocks(message)
                        )
                    )
                    for message in request.messages
                ),
                "mid-conversation-tool-changes-2026-07-01",
            ),
        )
        if used
    )
