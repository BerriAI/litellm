use litellm_types::llms::anthropic_messages::anthropic_request::{
    AnthropicMessage, AnthropicMessagesOptionalParams, AnthropicMessagesRequest, ContentBlock,
    MessageContent, SystemPrompt,
};

const SYSTEM_ROLE: &str = "system";

fn content_into_blocks(content: MessageContent) -> Vec<ContentBlock> {
    match content {
        MessageContent::Text(text) => vec![ContentBlock::text(text)],
        MessageContent::Blocks(blocks) => blocks,
    }
}

fn system_into_blocks(system: Option<SystemPrompt>) -> Vec<ContentBlock> {
    match system {
        None => Vec::new(),
        Some(SystemPrompt::Text(text)) => vec![ContentBlock::text(text)],
        Some(SystemPrompt::Blocks(blocks)) => blocks,
    }
}

pub fn fold_system_role_messages(request: AnthropicMessagesRequest) -> AnthropicMessagesRequest {
    if !request.messages.iter().any(|msg| msg.role == SYSTEM_ROLE) {
        return request;
    }

    let (system_messages, chat_messages): (Vec<AnthropicMessage>, Vec<AnthropicMessage>) = request
        .messages
        .into_iter()
        .partition(|msg| msg.role == SYSTEM_ROLE);

    let folded_system: Vec<ContentBlock> = system_into_blocks(request.params.system)
        .into_iter()
        .chain(
            system_messages
                .into_iter()
                .flat_map(|msg| content_into_blocks(msg.content)),
        )
        .collect();

    AnthropicMessagesRequest {
        messages: chat_messages,
        params: AnthropicMessagesOptionalParams {
            system: (!folded_system.is_empty()).then_some(SystemPrompt::Blocks(folded_system)),
            ..request.params
        },
        ..request
    }
}
