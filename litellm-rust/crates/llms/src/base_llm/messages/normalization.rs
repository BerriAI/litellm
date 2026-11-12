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

#[cfg(test)]
mod tests {
    use crate::base_llm::messages::normalization::normalize_system_role_messages;
    use litellm_llms_types::formats::messages::MessagesRequest;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::text_system(json!("existing"), json!([{"type": "text", "text": "existing"}]))]
    #[case::block_system(json!([{ "type": "future", "payload": 7 }]), json!([{ "type": "future", "payload": 7 }]))]
    #[case::no_system(Value::Null, json!([]))]
    fn hoisting_preserves_block_fields_order_and_unrelated_request_fields(
        #[case] system: Value,
        #[case] initial_blocks: Value,
    ) {
        let cache_control = json!({"type": "ephemeral", "scope": "global", "future": true});
        let leading_block =
            json!({"type": "text", "text": "second", "cache_control": cache_control});
        let user = json!({"role": "user", "content": "hello", "future_message": 42});
        let later_turn = json!({"role": "system", "content": "mid-conversation reminder"});
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "max_tokens": 64,
            "system": system,
            "messages": [
                {"role": "system", "content": "first"},
                {"role": "system", "content": [leading_block]},
                user,
                later_turn
            ],
            "future_request": {"nested": true}
        }))
        .unwrap();
        let normalized = normalize_system_role_messages(request, true);
        let expected_blocks: Vec<Value> = initial_blocks
            .as_array()
            .unwrap()
            .iter()
            .cloned()
            .chain([json!({"type": "text", "text": "first"}), leading_block])
            .collect();
        assert_eq!(
            serde_json::to_value(&normalized).unwrap(),
            json!({
                "model": "test-model",
                "max_tokens": 64,
                "system": expected_blocks,
                "messages": [user, later_turn],
                "future_request": {"nested": true}
            })
        );
        assert_eq!(
            normalize_system_role_messages(normalized.clone(), true),
            normalized
        );
    }
}
