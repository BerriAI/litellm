use litellm_llms_types::{
    formats::messages::{
        CacheControl, ContentBlock, Message, MessageContent, MessagesOptionalParams,
        MessagesRequest, SystemPrompt,
    },
    providers::anthropic::{BetaProvider, BetaSet},
    recognized::{Recognized, map_known_runs},
};

use crate::{
    anthropic::{
        beta_gated_fields::strip_beta_gated_system_messages,
        common_utils::filter_billing_headers_from_system,
    },
    base_llm::messages::mid_conversation_system::{
        as_system_content_blocks, convert_mid_conversation_system_turns, is_system_role_message,
        message_content_blocks,
    },
};

/// Python's `_normalize_system_role_messages`: a leading run of `role: "system"` entries is
/// hoisted into the top-level `system`, a later one stays in place when the model accepts it
/// and otherwise loses its beta-gated fields under drop_params and becomes a user turn where it
/// was, and billing blocks are stripped from the top-level `system` either way.
pub fn normalize_system_role_messages(
    request: MessagesRequest,
    supports_mid_conversation_system: bool,
    drops_unsupported_params: bool,
    provider: BetaProvider,
) -> MessagesRequest {
    let leading_count = request
        .messages
        .iter()
        .take_while(|message| message.known().is_some_and(is_system_role_message))
        .count();
    let mut messages = request.messages.into_iter();
    let hoisted: Vec<Message> = messages
        .by_ref()
        .take(leading_count)
        .filter_map(|message| match message {
            Recognized::Known(message) => Some(message),
            Recognized::Unrecognized(_) => None,
        })
        .collect();
    let remaining: Vec<Recognized<Message>> = messages.collect();
    let remaining = if supports_mid_conversation_system {
        remaining
    } else {
        map_known_runs(
            strip_beta_gated_system_messages(
                remaining,
                &BetaSet::default(),
                drops_unsupported_params,
                &request.model,
                provider,
            ),
            convert_mid_conversation_system_turns,
        )
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
            .map(|message| message.map_known(strip_scope_from_message))
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
    use crate::base_llm::messages::{
        mid_conversation_system::CONVERTED_SYSTEM_NOTE,
        normalization::normalize_system_role_messages,
    };
    use litellm_llms_types::{
        formats::messages::MessagesRequest, providers::anthropic::BetaProvider,
    };
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
        let normalized =
            normalize_system_role_messages(request, true, false, BetaProvider::AzureAi);
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
            normalize_system_role_messages(normalized.clone(), true, false, BetaProvider::AzureAi),
            normalized
        );
    }

    fn request(fields: Value) -> MessagesRequest {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let mut body = json!({"model": "test-model", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut().unwrap().extend(fields);
        serde_json::from_value(body).unwrap()
    }

    fn reminder_with(content: Value) -> Value {
        json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": content},
            {"role": "user", "content": "go on"}
        ]})
    }

    fn tool_reference_addition() -> Value {
        json!({"type": "tool_addition", "tool": {"type": "tool_reference", "tool_name": "t"}})
    }

    #[rstest]
    #[case::stripped_under_drop_params(
        true,
        json!([{"type": "text", "text": CONVERTED_SYSTEM_NOTE}, {"type": "text", "text": "note"}])
    )]
    #[case::kept_without_drop_params(
        false,
        json!([{"type": "text", "text": CONVERTED_SYSTEM_NOTE}, {"type": "text", "text": "note"}, tool_reference_addition()])
    )]
    fn a_converted_turn_loses_its_tool_changes_under_drop_params(
        #[case] drops_unsupported_params: bool,
        #[case] converted: Value,
    ) {
        let normalized = normalize_system_role_messages(
            request(reminder_with(
                json!([{"type": "text", "text": "note"}, tool_reference_addition()]),
            )),
            false,
            drops_unsupported_params,
            BetaProvider::AzureAi,
        );
        assert_eq!(
            serde_json::to_value(normalized).unwrap()["messages"][2]["content"],
            converted
        );
    }

    #[rstest]
    fn a_turn_kept_in_place_keeps_its_tool_changes_under_drop_params() {
        let sent = reminder_with(json!([tool_reference_addition()]));
        let normalized = normalize_system_role_messages(
            request(sent.clone()),
            true,
            true,
            BetaProvider::AzureAi,
        );
        assert_eq!(
            serde_json::to_value(normalized).unwrap()["messages"],
            sent["messages"]
        );
    }

    #[rstest]
    fn a_converted_turn_emptied_by_the_strip_is_dropped() {
        let normalized = normalize_system_role_messages(
            request(reminder_with(json!([tool_reference_addition()]))),
            false,
            true,
            BetaProvider::AzureAi,
        );
        assert_eq!(
            serde_json::to_value(normalized).unwrap()["messages"],
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "go on"}
            ])
        );
    }
}
