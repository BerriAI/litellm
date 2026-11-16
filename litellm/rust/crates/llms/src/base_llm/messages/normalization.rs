use litellm_llms_types::{
    formats::messages::{
        CacheControl, ContentBlock, Message, MessageContent, MessagesOptionalParams,
        MessagesRequest, SystemPrompt,
    },
    providers::anthropic::{BetaProvider, BetaSet},
    recognized::Recognized,
};
use serde_json::Value;

use crate::{
    anthropic::{
        beta_gated_fields::strip_beta_gated_system_messages,
        common_utils::strip_billing_metadata_from_system,
    },
    base_llm::messages::mid_conversation_system::{
        as_system_content_blocks, convert_mid_conversation_system_turns, is_system_turn,
        message_content_blocks, system_prompt,
    },
};

/// Python's `_normalize_system_role_messages`, run on a request whose billing blocks Python's
/// transform already stripped: a leading run of `role: "system"` entries is hoisted into the
/// top-level `system`, which is always sent as blocks, a later one stays in place when the model
/// accepts it and otherwise loses its beta-gated fields under drop_params and becomes a user
/// turn where it was, and billing blocks are stripped from the top-level `system` either way.
pub fn normalize_system_role_messages(
    request: MessagesRequest,
    supports_mid_conversation_system: bool,
    drops_unsupported_params: bool,
    provider: BetaProvider,
) -> MessagesRequest {
    let leading_count = request
        .messages
        .iter()
        .take_while(|message| is_system_turn(message))
        .count();
    let mut messages = request.messages.into_iter();
    let hoisted: Vec<Recognized<Message>> = messages.by_ref().take(leading_count).collect();
    let remaining: Vec<Recognized<Message>> = messages.collect();
    let remaining = if supports_mid_conversation_system {
        remaining
    } else {
        convert_mid_conversation_system_turns(strip_beta_gated_system_messages(
            remaining,
            &BetaSet::default(),
            drops_unsupported_params,
            &request.model,
            provider,
        ))
    };
    let system_blocks: Vec<Recognized<ContentBlock>> = as_system_content_blocks(
        request
            .params
            .system
            .and_then(strip_billing_metadata_from_system),
    )
    .into_iter()
    .chain(hoisted.into_iter().flat_map(message_content_blocks))
    .collect();
    MessagesRequest {
        messages: remaining,
        params: MessagesOptionalParams {
            system: strip_billing_metadata_from_system(system_prompt(system_blocks)),
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

fn strip_scope_from_system(system: Recognized<SystemPrompt>) -> Recognized<SystemPrompt> {
    match system {
        Recognized::Known(SystemPrompt::Blocks(blocks)) => Recognized::Known(SystemPrompt::Blocks(
            blocks.into_iter().map(strip_scope_from_block).collect(),
        )),
        Recognized::Unrecognized(Value::Array(blocks)) => Recognized::Unrecognized(Value::Array(
            blocks.into_iter().map(strip_scope_from_raw_block).collect(),
        )),
        system => system,
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

/// Python's sanitizer reads any mapping, so a message that did not parse loses the scope of
/// every `cache_control` object among its content blocks too.
fn strip_scope_from_raw_message(message: Value) -> Value {
    let Value::Object(fields) = message else {
        return message;
    };
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| match value {
                Value::Array(blocks) if key == "content" => (
                    key,
                    Value::Array(blocks.into_iter().map(strip_scope_from_raw_block).collect()),
                ),
                value => (key, value),
            })
            .collect(),
    )
}

fn strip_scope_from_raw_block(block: Value) -> Value {
    let Value::Object(fields) = block else {
        return block;
    };
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| match value {
                Value::Object(cache_control) if key == "cache_control" => (
                    key,
                    Value::Object(
                        cache_control
                            .into_iter()
                            .filter(|(field, _)| field != "scope")
                            .collect(),
                    ),
                ),
                value => (key, value),
            })
            .collect(),
    )
}

/// Python's `_remove_scope_from_cache_control`: hosts other than the first-party API reject
/// `cache_control.scope`, so it is dropped from the system prompt and every message block.
pub fn strip_cache_control_scope(request: MessagesRequest) -> MessagesRequest {
    MessagesRequest {
        messages: request
            .messages
            .into_iter()
            .map(|message| match message {
                Recognized::Known(message) => Recognized::Known(strip_scope_from_message(message)),
                Recognized::Unrecognized(message) => {
                    Recognized::Unrecognized(strip_scope_from_raw_message(message))
                }
            })
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
        normalization::{normalize_system_role_messages, strip_cache_control_scope},
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

    fn sent_system(fields: Value) -> Option<Value> {
        serde_json::to_value(normalize_system_role_messages(
            request(fields),
            false,
            false,
            BetaProvider::AzureAi,
        ))
        .unwrap()
        .get("system")
        .cloned()
    }

    #[rstest]
    #[case::string(json!("be terse"), Some(json!([{"type": "text", "text": "be terse"}])))]
    #[case::blank_string(json!(" "), Some(json!([{"type": "text", "text": " "}])))]
    #[case::empty_string(json!(""), None)]
    #[case::billing_string(json!("x-anthropic-billing-header: cc_version=1"), None)]
    #[case::empty_blocks(json!([]), None)]
    #[case::blocks(json!([{"type": "text", "text": "be terse"}]), Some(json!([{"type": "text", "text": "be terse"}])))]
    fn the_top_level_system_is_sent_as_blocks(
        #[case] system: Value,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(sent_system(json!({"system": system})), expected);
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

    fn sent_messages(
        messages: Value,
        supports_mid_conversation_system: bool,
        drops_unsupported_params: bool,
    ) -> Value {
        serde_json::to_value(normalize_system_role_messages(
            request(json!({"messages": messages})),
            supports_mid_conversation_system,
            drops_unsupported_params,
            BetaProvider::AzureAi,
        ))
        .unwrap()
    }

    #[rstest]
    #[case::missing_content(json!({"role": "system"}))]
    #[case::null_content(json!({"role": "system", "content": null}))]
    #[case::extra_keys(json!({"role": "system", "clear_at": "next_user_message", "name": "n"}))]
    fn a_contentless_system_turn_is_a_system_turn_with_no_content(#[case] contentless: Value) {
        assert_eq!(
            sent_messages(
                json!([
                    {"role": "system", "content": "first"},
                    contentless,
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]},
                    contentless,
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
                    contentless
                ]),
                false,
                false,
            ),
            json!({
                "model": "test-model",
                "max_tokens": 64,
                "system": [{"type": "text", "text": "first"}],
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
                    {"role": "user", "content": [{"type": "text", "text": CONVERTED_SYSTEM_NOTE}]},
                    {"role": "user", "content": [{"type": "text", "text": CONVERTED_SYSTEM_NOTE}]}
                ]
            })
        );
    }

    #[rstest]
    fn a_contentless_system_turn_stays_as_sent_on_a_model_that_accepts_it() {
        let later = json!([
            {"role": "user", "content": "hi"},
            {"role": "system", "clear_at": "next_user_message"}
        ]);
        let sent = sent_messages(json!([{"role": "system"}, later[0], later[1]]), true, false);
        assert_eq!(sent["messages"], later);
        assert_eq!(sent.get("system"), None);
    }

    #[rstest]
    fn a_contentless_system_turn_that_loses_its_gated_field_is_dropped() {
        assert_eq!(
            sent_messages(
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "system", "clear_at": "next_user_message"},
                    {"role": "user", "content": "go on"}
                ]),
                false,
                true,
            )["messages"],
            json!([
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "go on"}
            ])
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

    #[rstest]
    #[case::unparsed_block_beside_a_scoped_one(
        json!({"role": "user", "content": [
            {"type": "text", "text": 5},
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "scope": "global", "ttl": "1h"}}
        ]}),
        json!({"role": "user", "content": [
            {"type": "text", "text": 5},
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ]})
    )]
    #[case::unparsed_scoped_block(
        json!({"role": "user", "content": [{"type": "text", "text": 5, "cache_control": {"scope": "global"}}]}),
        json!({"role": "user", "content": [{"type": "text", "text": 5, "cache_control": {}}]})
    )]
    #[case::non_object_cache_control(
        json!({"content": [{"type": "text", "text": "hi", "cache_control": "global"}]}),
        json!({"content": [{"type": "text", "text": "hi", "cache_control": "global"}]})
    )]
    #[case::text_content(
        json!({"role": 5, "content": "hi", "cache_control": {"scope": "global"}}),
        json!({"role": 5, "content": "hi", "cache_control": {"scope": "global"}})
    )]
    fn an_unparsed_message_loses_its_cache_control_scope(
        #[case] message: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            serde_json::to_value(strip_cache_control_scope(request(
                json!({"messages": [message]})
            )))
            .unwrap()["messages"],
            json!([expected])
        );
    }

    fn uncacheable_block() -> Value {
        json!({"type": "text", "text": "Be terse.", "cache_control": "ephemeral"})
    }

    #[rstest]
    #[case::unparsed_block(
        json!([{"role": "system", "content": [uncacheable_block()]}, {"role": "user", "content": "hi"}]),
        json!([{"type": "text", "text": "first"}, uncacheable_block()])
    )]
    #[case::non_list_content(
        json!([{"role": "system", "content": 5}, {"role": "user", "content": "hi"}]),
        json!([{"type": "text", "text": "first"}, 5])
    )]
    #[case::object_content(
        json!([{"role": "system", "content": {"type": "text", "text": "Be terse."}}, {"role": "user", "content": "hi"}]),
        json!([{"type": "text", "text": "first"}, {"type": "text", "text": "Be terse."}])
    )]
    fn a_leading_system_turn_that_did_not_parse_is_hoisted_as_sent(
        #[case] messages: Value,
        #[case] system: Value,
    ) {
        let sent = serde_json::to_value(normalize_system_role_messages(
            request(json!({"system": "first", "messages": messages})),
            true,
            false,
            BetaProvider::AzureAi,
        ))
        .unwrap();
        assert_eq!(sent["system"], system);
        assert_eq!(sent["messages"], json!([{"role": "user", "content": "hi"}]));
    }

    #[rstest]
    fn a_later_system_turn_that_did_not_parse_becomes_a_user_turn_as_sent() {
        assert_eq!(
            sent_messages(
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]},
                    {"role": "system", "content": [uncacheable_block()]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}, {"type": "text", "text": "x", "id": 5}]}
                ]),
                false,
                false,
            )["messages"],
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}, {"type": "text", "text": "x", "id": 5}]},
                {"role": "user", "content": [{"type": "text", "text": CONVERTED_SYSTEM_NOTE}, uncacheable_block()]}
            ])
        );
    }

    #[rstest]
    fn a_system_list_that_did_not_parse_loses_its_billing_and_scope() {
        let system = json!([
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
            {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral", "scope": "global"}},
            {"type": "text", "text": "x", "id": 5}
        ]);
        let sent = serde_json::to_value(strip_cache_control_scope(normalize_system_role_messages(
            request(json!({"system": system})),
            true,
            false,
            BetaProvider::AzureAi,
        )))
        .unwrap();
        assert_eq!(
            sent["system"],
            json!([
                {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "x", "id": 5}
            ])
        );
    }
}
