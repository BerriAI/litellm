use litellm_llms_types::formats::messages::{
    ContentBlock, ContentBlockType, Message, MessageContent, MessagesOptionalParams,
    MessagesRequest, SystemPrompt,
};

const SYSTEM_ROLE: &str = "system";
const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

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

fn is_billing_metadata_block(block: &ContentBlock) -> bool {
    block.is_type(ContentBlockType::Text)
        && block
            .text
            .as_deref()
            .is_some_and(|text| text.starts_with(BILLING_HEADER_PREFIX))
}

pub fn strip_billing_metadata(request: MessagesRequest) -> MessagesRequest {
    let system = match request.params.system {
        Some(SystemPrompt::Text(text)) => {
            (!text.starts_with(BILLING_HEADER_PREFIX)).then_some(SystemPrompt::Text(text))
        }
        Some(SystemPrompt::Blocks(blocks)) => {
            let kept: Vec<ContentBlock> = blocks
                .into_iter()
                .filter(|block| !is_billing_metadata_block(block))
                .collect();
            (!kept.is_empty()).then_some(SystemPrompt::Blocks(kept))
        }
        None => None,
    };
    MessagesRequest {
        params: MessagesOptionalParams {
            system,
            ..request.params
        },
        ..request
    }
}
