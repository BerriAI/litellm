fn thinking_display_updates_used(request: &BetaDetectionRequest) -> bool {
    matches!(
        thinking_type(request),
        Some(BetaThinkingType::Enabled) | Some(BetaThinkingType::Adaptive)
    ) && matches!(
        thinking_display(request),
        Some(BetaThinkingDisplay::Updates)
    )
}

fn mid_conversation_tool_changes_used(request: &BetaDetectionRequest) -> bool {
    request.messages.iter().any(|message| {
        ["system"].contains(&message.role.as_str())
            && content_blocks(message).any(|block| {
                matches!(
                    block.type_,
                    BetaContentType::ToolAddition | BetaContentType::ToolRemoval
                ) && matches!(
                    block
                        .tool
                        .as_ref()
                        .and_then(Recognized::known)
                        .map(|tool| &tool.type_),
                    Some(BetaToolTargetType::ToolReference)
                )
            })
    })
}

pub fn detect_anthropic_betas(request: &BetaDetectionRequest) -> BetaSet {
    [
        thinking_display_updates_used(request)
            .then_some(AnthropicBeta::ThinkingDisplayUpdates20260818),
        mid_conversation_tool_changes_used(request)
            .then_some(AnthropicBeta::MidConversationToolChanges20260701),
    ]
    .into_iter()
    .flatten()
    .collect()
}
