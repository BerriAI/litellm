fn thinking_display_updates_used(request: &MessagesRequest) -> bool {
    matches!(
        request.params.thinking.as_ref().and_then(Recognized::known),
        Some(ThinkingConfig::Enabled(_)) | Some(ThinkingConfig::Adaptive(_))
    ) && matches!(thinking_display(request), Some(ThinkingDisplay::Updates))
}

fn mid_conversation_tool_changes_used(request: &MessagesRequest) -> bool {
    request.messages.iter().any(|message| {
        ["system"].contains(&message.role.as_str())
            && message.blocks().iter().any(|block| {
                block.block_type.as_ref().is_some_and(|kind| match kind {
                    ContentBlockType::Other(value) => {
                        ["tool_addition", "tool_removal"].contains(&value.as_str())
                    }
                    _ => false,
                }) && (block
                    .extra
                    .get("tool")
                    .and_then(Value::as_object)
                    .and_then(|tool| tool.get("type"))
                    .and_then(Value::as_str))
                .is_some_and(|value| ["tool_reference"].contains(&value))
            })
    })
}

pub fn detect_anthropic_betas(request: &MessagesRequest) -> BetaSet {
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
