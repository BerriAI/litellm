use litellm_llms_types::{
    formats::messages::{
        CacheControl, ContentBlock, ContentBlockType, Message, MessageContent, MessageRole,
        MessagesOptionalParams, MessagesRequest, SystemPrompt,
    },
    recognized::Recognized,
    serde_compat::Nullable,
};

const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";
pub const CONVERTED_SYSTEM_NOTE: &str = "Operator note (not from the user): the following was originally a mid-conversation system-role reminder.";

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
            system: (!folded_system.is_empty()).then_some(SystemPrompt::Blocks(folded_system)),
            ..request.params
        },
        ..request
    }
}

fn opens_with_tool_results(message: &Message) -> bool {
    message.role == MessageRole::User
        && matches!(&message.content, MessageContent::Blocks(blocks) if blocks
            .first()
            .is_some_and(|block| block.is_type(ContentBlockType::ToolResult)))
}

fn system_role_message_as_user(message: Message) -> Message {
    if message.role != MessageRole::System {
        return message;
    }
    Message {
        role: MessageRole::User,
        content: MessageContent::Blocks(
            [ContentBlock::text(CONVERTED_SYSTEM_NOTE.to_string())]
                .into_iter()
                .chain(content_into_blocks(message.content))
                .collect(),
        ),
        extra: Default::default(),
    }
}

pub fn convert_mid_conversation_system_turns(messages: Vec<Message>) -> Vec<Message> {
    if !messages
        .iter()
        .any(|message| message.role == MessageRole::System)
    {
        return messages;
    }
    let runs: Vec<&[Message]> = messages
        .chunk_by(|left, right| {
            (left.role == MessageRole::System) == (right.role == MessageRole::System)
        })
        .collect();
    let first_system_run = usize::from(runs[0][0].role != MessageRole::System);
    runs[..first_system_run]
        .iter()
        .flat_map(|run| run.iter().cloned())
        .chain(runs[first_system_run..].chunks(2).flat_map(|pair| {
            let system_run = pair[0];
            let follower_run = pair.get(1).copied().unwrap_or_default();
            let tool_result_count =
                usize::from(follower_run.first().is_some_and(opens_with_tool_results));
            follower_run[..tool_result_count]
                .iter()
                .chain(system_run)
                .chain(&follower_run[tool_result_count..])
                .cloned()
        }))
        .map(system_role_message_as_user)
        .collect()
}

pub fn normalize_system_role_messages(
    request: MessagesRequest,
    supports_mid_conversation_system: bool,
) -> MessagesRequest {
    let leading_count = request
        .messages
        .iter()
        .take_while(|message| message.role == MessageRole::System)
        .count();
    let mut leading = request.messages;
    let remaining = leading.split_off(leading_count);
    let messages = if supports_mid_conversation_system {
        remaining
    } else {
        convert_mid_conversation_system_turns(remaining)
    };
    let system: Vec<ContentBlock> = system_into_blocks(request.params.system)
        .into_iter()
        .chain(
            leading
                .into_iter()
                .flat_map(|message| content_into_blocks(message.content)),
        )
        .collect();
    let system = without_billing_metadata(SystemPrompt::Blocks(system));
    MessagesRequest {
        messages,
        params: MessagesOptionalParams {
            system,
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

fn without_scope(cache_control: CacheControl) -> CacheControl {
    CacheControl {
        scope: None,
        ..cache_control
    }
}

fn block_without_scope(block: ContentBlock) -> ContentBlock {
    ContentBlock {
        cache_control: block
            .cache_control
            .map(|cache_control| match cache_control {
                Nullable::Value(cache_control) => Nullable::Value(without_scope(cache_control)),
                other => other,
            }),
        ..block
    }
}

pub(crate) fn map_request_blocks(
    request: MessagesRequest,
    system_block: fn(ContentBlock) -> ContentBlock,
    message_block: fn(ContentBlock) -> ContentBlock,
) -> MessagesRequest {
    MessagesRequest {
        messages: request
            .messages
            .into_iter()
            .map(|message| Message {
                content: match message.content {
                    MessageContent::Blocks(blocks) => {
                        MessageContent::Blocks(blocks.into_iter().map(message_block).collect())
                    }
                    text => text,
                },
                ..message
            })
            .collect(),
        params: MessagesOptionalParams {
            system: request.params.system.map(|system| match system {
                SystemPrompt::Blocks(blocks) => {
                    SystemPrompt::Blocks(blocks.into_iter().map(system_block).collect())
                }
                text => text,
            }),
            ..request.params
        },
        ..request
    }
}

/// Removes `cache_control.scope`, which hosts other than the first-party API reject, from
/// the top-level `cache_control` and every `system` and message block.
pub fn strip_cache_control_scope(request: MessagesRequest) -> MessagesRequest {
    let request = map_request_blocks(request, block_without_scope, block_without_scope);
    MessagesRequest {
        params: MessagesOptionalParams {
            cache_control: request
                .params
                .cache_control
                .map(|cache_control| match cache_control {
                    Recognized::Known(cache_control) => {
                        Recognized::Known(without_scope(cache_control))
                    }
                    other => other,
                }),
            ..request.params
        },
        ..request
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;

    fn text_message(role: MessageRole, text: impl Into<String>) -> Message {
        Message {
            role,
            content: MessageContent::Text(text.into()),
            extra: Default::default(),
        }
    }

    #[fixture]
    fn operator_note() -> ContentBlock {
        ContentBlock::text(
            "Operator note (not from the user): the following was originally a mid-conversation system-role reminder.",
        )
    }

    #[fixture]
    fn tool_result() -> Message {
        Message {
            role: MessageRole::User,
            content: MessageContent::Blocks(vec![ContentBlock {
                block_type: Some(Nullable::Value(ContentBlockType::ToolResult)),
                tool_use_id: Some(Nullable::Value("toolu_1".into())),
                payload: litellm_llms_types::formats::messages::ContentBlockPayload {
                    content: Some(Recognized::Known(
                        litellm_llms_types::formats::messages::BlockContent::Text("Rainy".into()),
                    )),
                    ..Default::default()
                },
                ..ContentBlock::default()
            }]),
            extra: serde_json::Map::from_iter([("future".into(), json!({"opaque": null}))]),
        }
    }

    mod mid_conversation_system {
        use super::*;

        #[rstest]
        fn converts_system_to_user_in_place(operator_note: ContentBlock) {
            let user = text_message(MessageRole::User, "hi");
            let assistant = text_message(MessageRole::Assistant, "Hi.");
            let block: ContentBlock = serde_json::from_value(json!({
                "type": "text", "text": "Keep it short.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
                "future": {"nested": [1, null]}
            }))
            .unwrap();
            let system = Message {
                role: MessageRole::System,
                content: MessageContent::Blocks(vec![block.clone()]),
                extra: serde_json::Map::from_iter([("discarded".into(), json!(true))]),
            };
            assert_eq!(
                convert_mid_conversation_system_turns(vec![
                    user.clone(),
                    system,
                    assistant.clone()
                ]),
                vec![
                    user,
                    Message {
                        role: MessageRole::User,
                        content: MessageContent::Blocks(vec![operator_note, block]),
                        extra: Default::default(),
                    },
                    assistant
                ]
            );
        }

        #[rstest]
        fn wraps_string_content(operator_note: ContentBlock) {
            let user = text_message(MessageRole::User, "hi");
            assert_eq!(
                convert_mid_conversation_system_turns(vec![
                    user.clone(),
                    text_message(MessageRole::System, "Keep it short.")
                ]),
                vec![
                    user,
                    Message {
                        role: MessageRole::User,
                        content: MessageContent::Blocks(vec![
                            operator_note,
                            ContentBlock::text("Keep it short.")
                        ]),
                        extra: Default::default(),
                    }
                ]
            );
        }

        #[rstest]
        fn moves_system_after_tool_result(tool_result: Message, operator_note: ContentBlock) {
            let assistant: Message = serde_json::from_value(json!({
                "role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}]
            })).unwrap();
            let result = convert_mid_conversation_system_turns(vec![
                assistant.clone(),
                text_message(MessageRole::System, "Use the corrected result."),
                tool_result.clone(),
            ]);
            assert_eq!(
                result,
                vec![
                    assistant,
                    tool_result,
                    Message {
                        role: MessageRole::User,
                        content: MessageContent::Blocks(vec![
                            operator_note,
                            ContentBlock::text("Use the corrected result.")
                        ]),
                        extra: Default::default(),
                    }
                ]
            );
        }

        #[rstest]
        #[case::text(json!({"role": "user", "content": "plain"}))]
        #[case::tool_result_after_text(json!({"role": "user", "content": [
            {"type": "text", "text": "first"}, {"type": "tool_result", "tool_use_id": "toolu_1"}
        ]}))]
        #[case::empty_blocks(json!({"role": "user", "content": []}))]
        #[case::assistant_tool_result(json!({"role": "assistant", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1"}
        ]}))]
        fn preserves_order_when_follower_does_not_open_with_user_tool_results(
            #[case] follower: Value,
            operator_note: ContentBlock,
        ) {
            let follower: Message = serde_json::from_value(follower).unwrap();
            assert_eq!(
                convert_mid_conversation_system_turns(vec![
                    text_message(MessageRole::System, "reminder"),
                    follower.clone()
                ]),
                vec![
                    Message {
                        role: MessageRole::User,
                        content: MessageContent::Blocks(vec![
                            operator_note,
                            ContentBlock::text("reminder")
                        ]),
                        extra: Default::default(),
                    },
                    follower
                ]
            );
        }

        #[rstest]
        fn long_system_run_preserves_order_without_splitting_tool_results(
            tool_result: Message,
            operator_note: ContentBlock,
        ) {
            let user = text_message(MessageRole::User, "hi");
            let messages: Vec<_> =
                [user.clone()]
                    .into_iter()
                    .chain((0..2000).map(|index| {
                        text_message(MessageRole::System, format!("reminder {index}"))
                    }))
                    .chain([tool_result.clone()])
                    .collect();
            let result = convert_mid_conversation_system_turns(messages);
            assert_eq!(result.len(), 2002);
            assert_eq!(result[0], user);
            assert_eq!(result[1], tool_result);
            for (index, message) in result[2..].iter().enumerate() {
                assert_eq!(message.role, MessageRole::User);
                assert_eq!(
                    message.content,
                    MessageContent::Blocks(vec![
                        operator_note.clone(),
                        ContentBlock::text(format!("reminder {index}"))
                    ])
                );
            }
        }

        #[rstest]
        #[case::empty(vec![])]
        #[case::ordinary(vec![text_message(MessageRole::User, "hi"), text_message(MessageRole::Assistant, "hello")])]
        #[case::unknown_role(vec![text_message(MessageRole::Other("System".into()), "case-sensitive")])]
        fn history_without_system_roles_is_unchanged(#[case] messages: Vec<Message>) {
            assert_eq!(
                convert_mid_conversation_system_turns(messages.clone()),
                messages
            );
        }

        #[rstest]
        #[case::capable(true)]
        #[case::incapable(false)]
        fn normalization_hoists_only_leading_system_and_preserves_cached_history(
            #[case] supports_mid_conversation_system: bool,
            operator_note: ContentBlock,
        ) {
            let leading = text_message(MessageRole::System, "initial");
            let user = Message {
                role: MessageRole::User,
                content: MessageContent::Blocks(vec![ContentBlock {
                    cache_control: Some(Nullable::Value(CacheControl {
                        cache_type: Some(Nullable::Value("ephemeral".into())),
                        ttl: Some(Nullable::Value("1h".into())),
                        ..CacheControl::default()
                    })),
                    ..ContentBlock::text("hi")
                }]),
                extra: Default::default(),
            };
            let assistant = text_message(MessageRole::Assistant, "hello");
            let later = text_message(MessageRole::System, "reminder");
            let request = MessagesRequest {
                model: "m".into(),
                messages: vec![leading, user.clone(), assistant.clone(), later.clone()],
                params: MessagesOptionalParams {
                    system: Some(SystemPrompt::Text("existing".into())),
                    ..MessagesOptionalParams::default()
                },
            };
            let normalized =
                normalize_system_role_messages(request, supports_mid_conversation_system);
            assert_eq!(
                normalized.params.system,
                Some(SystemPrompt::Blocks(vec![
                    ContentBlock::text("existing"),
                    ContentBlock::text("initial")
                ]))
            );
            let expected_later = if supports_mid_conversation_system {
                later
            } else {
                Message {
                    role: MessageRole::User,
                    content: MessageContent::Blocks(vec![
                        operator_note,
                        ContentBlock::text("reminder"),
                    ]),
                    extra: Default::default(),
                }
            };
            assert_eq!(normalized.messages, vec![user, assistant, expected_later]);
            assert_eq!(
                normalize_system_role_messages(
                    normalized.clone(),
                    supports_mid_conversation_system
                ),
                normalized
            );
        }

        #[rstest]
        fn normalization_strips_billing_from_hoisted_and_existing_system() {
            let request = MessagesRequest {
                model: "m".into(),
                messages: vec![
                    text_message(
                        MessageRole::System,
                        "x-anthropic-billing-header: cc_version=1",
                    ),
                    text_message(MessageRole::User, "hi"),
                ],
                params: MessagesOptionalParams {
                    system: Some(SystemPrompt::Blocks(vec![
                        ContentBlock::text("x-anthropic-billing-header: cc_version=2"),
                        ContentBlock::text("keep this"),
                    ])),
                    ..MessagesOptionalParams::default()
                },
            };
            let result = normalize_system_role_messages(request, false);
            assert_eq!(
                result.params.system,
                Some(SystemPrompt::Blocks(vec![ContentBlock::text("keep this")]))
            );
            assert_eq!(result.messages, vec![text_message(MessageRole::User, "hi")]);
        }
    }

    fn system_after_strip(system: Value) -> Value {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 1,
            "system": system,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        serde_json::to_value(strip_billing_metadata(request)).unwrap()["system"].clone()
    }

    #[test]
    fn cache_control_scope_is_stripped_everywhere_and_idempotently() {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 1,
            "cache_control": {"type": "ephemeral", "scope": "global"},
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "a", "cache_control": {"type": "ephemeral", "scope": "global"}},
                    {"type": "text", "text": "b"}
                ]},
                {"role": "assistant", "content": "plain"}
            ]
        }))
        .unwrap();
        let once = strip_cache_control_scope(request);
        let value = serde_json::to_value(&once).unwrap();
        assert_eq!(value["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(
            value["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            value["messages"][0]["content"][1],
            json!({"type": "text", "text": "b"})
        );
        assert_eq!(value["messages"][1]["content"], json!("plain"));
        assert_eq!(strip_cache_control_scope(once.clone()), once);
    }

    #[rstest]
    #[case::tool_definition(json!({"tools": [{
        "name": "lookup", "cache_control": {"type": "ephemeral", "scope": "global"}
    }]}))]
    #[case::nested_tool_result(json!({"messages": [{"role": "user", "content": [{
        "type": "tool_result", "content": [{"type": "text", "text": "result", "cache_control": {
            "type": "ephemeral", "scope": "global"
        }}]
    }]}]}))]
    #[case::application_input(json!({"messages": [{"role": "assistant", "content": [{
        "type": "tool_use", "input": {"cache_control": {"scope": "application-data"}}
    }]}]}))]
    #[case::unrecognized_top_level_cache(json!({"cache_control": {
        "type": false, "scope": "global"
    }}))]
    fn scope_stripping_preserves_sites_outside_its_existing_contract(#[case] fields: Value) {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let request: MessagesRequest = serde_json::from_value(Value::Object(
            [
                ("model".to_string(), json!("m")),
                ("messages".to_string(), json!([])),
            ]
            .into_iter()
            .chain(fields)
            .collect(),
        ))
        .unwrap();
        assert_eq!(strip_cache_control_scope(request.clone()), request);
    }

    #[rstest]
    #[case::billing_string(json!("x-anthropic-billing-header: cc_version=1"), Value::Null)]
    #[case::empty_string(json!(""), Value::Null)]
    #[case::plain_string(json!("be terse"), json!("be terse"))]
    #[case::only_billing_blocks(
        json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}]),
        Value::Null
    )]
    #[case::billing_block_among_others(
        json!([
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
            {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}
        ]),
        json!([{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}])
    )]
    #[case::prefix_not_at_start(
        json!([{"type": "text", "text": "see x-anthropic-billing-header: here"}]),
        json!([{"type": "text", "text": "see x-anthropic-billing-header: here"}])
    )]
    fn billing_metadata_is_stripped_from_system(#[case] system: Value, #[case] expected: Value) {
        assert_eq!(system_after_strip(system), expected);
    }
}
