use litellm_llms_types::formats::messages::{
    ContentBlock, ContentBlockType, Message, MessageContent, MessagesOptionalParams,
    MessagesRequest, SystemPrompt,
};

use litellm_llms_types::serde_compat::Nullable;

const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";
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
    if !request
        .messages
        .iter()
        .any(|msg| msg.role.as_str() == SYSTEM_ROLE)
    {
        return request;
    }

    let (system_messages, chat_messages): (Vec<Message>, Vec<Message>) = request
        .messages
        .into_iter()
        .partition(|msg| msg.role.as_str() == SYSTEM_ROLE);

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

fn is_billing_metadata(text: &str) -> bool {
    text.starts_with(BILLING_HEADER_PREFIX)
}

fn is_billing_metadata_block(block: &ContentBlock) -> bool {
    matches!(
        block.block_type,
        Some(Nullable::Value(ContentBlockType::Text))
    ) && matches!(&block.text, Some(Nullable::Value(text)) if is_billing_metadata(text))
}

fn without_billing_metadata(system: SystemPrompt) -> Option<SystemPrompt> {
    match system {
        SystemPrompt::Text(text) if text.is_empty() || is_billing_metadata(&text) => None,
        SystemPrompt::Text(text) => Some(SystemPrompt::Text(text)),
        SystemPrompt::Blocks(blocks) => {
            let kept: Vec<ContentBlock> = blocks
                .into_iter()
                .filter(|block| !is_billing_metadata_block(block))
                .collect();
            (!kept.is_empty()).then_some(SystemPrompt::Blocks(kept))
        }
    }
}

/// Drops the `x-anthropic-billing-header` text Claude Code puts in `system`, which only the
/// first-party API understands, and omits `system` when nothing else is left.
pub fn strip_billing_metadata(request: MessagesRequest) -> MessagesRequest {
    MessagesRequest {
        params: MessagesOptionalParams {
            system: request.params.system.and_then(without_billing_metadata),
            ..request.params
        },
        ..request
    }
}
