use litellm_llms_types::formats::messages::{
    ContentBlock, Message, MessageContent, MessageRole, MessagesOptionalParams, MessagesRequest,
    SystemPrompt,
};

use litellm_llms_types::serde_compat::Nullable;

fn content_into_blocks(content: MessageContent) -> Vec<ContentBlock> {
    match content {
        MessageContent::Text(text) => vec![ContentBlock::text(text)],
        MessageContent::Blocks(blocks) => blocks,
    }
}

fn system_into_blocks(system: Option<Nullable<SystemPrompt>>) -> Vec<ContentBlock> {
    match system {
        None | Some(Nullable::Null) => Vec::new(),
        Some(Nullable::Value(SystemPrompt::Text(text))) => vec![ContentBlock::text(text)],
        Some(Nullable::Value(SystemPrompt::Blocks(blocks))) => blocks,
    }
}

pub fn fold_system_role_messages(request: MessagesRequest) -> MessagesRequest {
    if !request
        .messages
        .iter()
        .any(|msg| msg.role == MessageRole::System)
    {
        return request;
    }

    let (system_messages, chat_messages): (Vec<Message>, Vec<Message>) = request
        .messages
        .into_iter()
        .partition(|msg| msg.role == MessageRole::System);

    let folded_system: Vec<ContentBlock> = system_into_blocks(request.params.system)
        .into_iter()
        .chain(
            system_messages
                .into_iter()
                .flat_map(|msg| content_into_blocks(msg.content)),
        )
        .collect();

    MessagesRequest {
        messages: chat_messages,
        params: MessagesOptionalParams {
            system: (!folded_system.is_empty())
                .then_some(Nullable::Value(SystemPrompt::Blocks(folded_system))),
            ..request.params
        },
        ..request
    }
}
