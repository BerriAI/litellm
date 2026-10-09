use litellm_llms_types::formats::messages::{
    CacheControl, ContentBlock, Message, MessageContent, MessagesOptionalParams, MessagesRequest,
    SystemPrompt,
};

use crate::{
    anthropic::common_utils::filter_billing_headers_from_system,
    base_llm::messages::mid_conversation_system::{
        as_system_content_blocks, convert_mid_conversation_system_turns, is_system_role_message,
        message_content_blocks,
    },
};

/// Python's `_normalize_system_role_messages`: a leading run of `role: "system"` entries is
/// hoisted into the top-level `system`, a later one stays in place when the model accepts it
/// and otherwise becomes a user turn where it was, and billing blocks are stripped from the
/// top-level `system` either way.
pub fn normalize_system_role_messages(
    request: MessagesRequest,
    supports_mid_conversation_system: bool,
) -> MessagesRequest {
    let leading_count = request
        .messages
        .iter()
        .take_while(|message| is_system_role_message(message))
        .count();
    let mut messages = request.messages.into_iter();
    let hoisted: Vec<Message> = messages.by_ref().take(leading_count).collect();
    let remaining: Vec<Message> = messages.collect();
    let remaining = if supports_mid_conversation_system {
        remaining
    } else {
        convert_mid_conversation_system_turns(remaining)
    };
    let system = if hoisted.is_empty() {
        request.params.system
    } else {
        Some(SystemPrompt::Blocks(
            as_system_content_blocks(request.params.system)
                .into_iter()
                .chain(
                    hoisted
                        .into_iter()
                        .flat_map(|message| message_content_blocks(message.content)),
                )
                .collect(),
        ))
    };
    MessagesRequest {
        messages: remaining,
        params: MessagesOptionalParams {
            system: system.and_then(filter_billing_headers_from_system),
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
