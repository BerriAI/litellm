use litellm_llms_types::formats::messages::{
    CacheControl, ContentBlock, Message, MessageContent, MessagesOptionalParams, MessagesRequest,
    SystemPrompt,
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

pub fn fold_system_role_messages(request: MessagesRequest) -> MessagesRequest {
    if !request.messages.iter().any(|msg| msg.role == SYSTEM_ROLE) {
        return request;
    }

    let (system_messages, chat_messages): (Vec<Message>, Vec<Message>) = request
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

    MessagesRequest {
        messages: chat_messages,
        params: MessagesOptionalParams {
            system: (!folded_system.is_empty()).then_some(SystemPrompt::Blocks(folded_system)),
            ..request.params
        },
        ..request
    }
}

fn strip_scope_from_block(block: ContentBlock) -> ContentBlock {
    ContentBlock {
        cache_control: block.cache_control.map(|cache_control| CacheControl {
            scope: None,
            ..cache_control
        }),
        ..block
    }
}

fn strip_scope_from_system(system: SystemPrompt) -> SystemPrompt {
    match system {
        SystemPrompt::Blocks(blocks) => {
            SystemPrompt::Blocks(blocks.into_iter().map(strip_scope_from_block).collect())
        }
        text => text,
    }
}

fn strip_scope_from_message(message: Message) -> Message {
    Message {
        content: match message.content {
            MessageContent::Blocks(blocks) => {
                MessageContent::Blocks(blocks.into_iter().map(strip_scope_from_block).collect())
            }
            text => text,
        },
        ..message
    }
}

/// Python's `_remove_scope_from_cache_control`: hosts other than the first-party API reject
/// `cache_control.scope`, so it is dropped from the system prompt and every message block.
pub fn strip_cache_control_scope(request: MessagesRequest) -> MessagesRequest {
    MessagesRequest {
        messages: request
            .messages
            .into_iter()
            .map(strip_scope_from_message)
            .collect(),
        params: MessagesOptionalParams {
            system: request.params.system.map(strip_scope_from_system),
            ..request.params
        },
        ..request
    }
}
