use std::collections::BTreeSet;

use litellm_llms_types::{
    formats::messages::{ContentBlock, ContentBlockType, Message, MessageContent},
    providers::anthropic::{AnthropicBeta, BetaProvider, BetaSet},
    recognized::Recognized,
};
use serde_json::{Map, Value};

use crate::base_llm::messages::mid_conversation_system::{is_system_role_message, message_role};

const ASSISTANT_ROLE: &str = "assistant";
const SYSTEM_ROLE: &str = "system";
const CLEAR_AT: &str = "clear_at";
const CONTENT: &str = "content";
const TYPE: &str = "type";
const OUTPUT_CONFIG: &str = "output_config";
const TOOL_CHANGE_BLOCK_TYPES: [&str; 2] = ["tool_addition", "tool_removal"];

/// The mid-conversation system fields of Python's `BETA_GATED_FIELDS`, named as Python's warning
/// names them and declared in the order it sorts those names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GatedField {
    Emptied,
    ClearAt,
    ToolChanges,
    OutputConfig,
}

impl GatedField {
    fn name(self) -> &'static str {
        match self {
            Self::Emptied => "messages[].system (emptied by the strip)",
            Self::ClearAt => "messages[].system.clear_at",
            Self::ToolChanges => "messages[].system.content[].tool_addition|tool_removal",
            Self::OutputConfig => "messages[].system.output_config",
        }
    }

    fn betas(self) -> &'static [AnthropicBeta] {
        match self {
            Self::Emptied => &[],
            Self::ClearAt => &[AnthropicBeta::MidConversationSystemClearAt20260821],
            Self::ToolChanges => &[
                AnthropicBeta::InlineTools20260915,
                AnthropicBeta::MidConversationToolChanges20260701,
            ],
            Self::OutputConfig => &[
                AnthropicBeta::MidConversationOutputConfig20260701,
                AnthropicBeta::PerTurnControl20260701,
            ],
        }
    }

    fn gated(self, active: &BetaSet) -> bool {
        !self.betas().iter().any(|beta| active.contains(beta))
    }

    fn described(self) -> String {
        match self.betas() {
            [] => self.name().to_owned(),
            betas => format!(
                "{} (unlocked by anthropic-beta {})",
                self.name(),
                betas
                    .iter()
                    .map(AnthropicBeta::as_ref)
                    .collect::<Vec<_>>()
                    .join(" or ")
            ),
        }
    }
}

/// Python's `strip_beta_gated_system_messages` and `log_beta_gated_strips`: under drop_params,
/// every `role: "system"` message loses the fields no beta in `active` unlocks, and one left with
/// neither content nor `output_config` is dropped. The messages stay as sent when nothing is
/// gated, or when dropping an emptied turn would leave the conversation ending on an assistant
/// turn.
pub fn strip_beta_gated_system_messages(
    messages: Vec<Recognized<Message>>,
    active: &BetaSet,
    drops_unsupported_params: bool,
    model: &str,
    provider: BetaProvider,
) -> Vec<Recognized<Message>> {
    if !drops_unsupported_params {
        return messages;
    }
    let stripped: Vec<(Option<Recognized<Message>>, Vec<GatedField>)> = messages
        .iter()
        .cloned()
        .map(|message| match message {
            Recognized::Known(message) if is_system_role_message(&message) => {
                let (kept, gated) = strip_system_message(message, active);
                (kept.map(Recognized::Known), gated)
            }
            Recognized::Unrecognized(Value::Object(fields))
                if fields.get("role").and_then(Value::as_str) == Some(SYSTEM_ROLE) =>
            {
                let (kept, gated) = strip_unrecognized_system_message(fields, active);
                (
                    kept.map(|fields| Recognized::Unrecognized(Value::Object(fields))),
                    gated,
                )
            }
            other => (Some(other), Vec::new()),
        })
        .collect();
    let gated: BTreeSet<GatedField> = stripped
        .iter()
        .flat_map(|(_, gated)| gated.iter().copied())
        .collect();
    if gated.is_empty() {
        return messages;
    }
    let kept: Vec<Recognized<Message>> = stripped
        .into_iter()
        .filter_map(|(message, _)| message)
        .collect();
    if ends_on_assistant(&kept) && !ends_on_assistant(&messages) {
        return messages;
    }
    litellm_tracing::warn!(
        "Dropped {} for model={model} provider={provider} because drop_params is on and the provider would reject them on this request, either since this provider leg does not send that anthropic-beta (see anthropic_beta_headers_config.json) or since the model turns mid-conversation system messages into user messages.",
        gated
            .into_iter()
            .map(GatedField::described)
            .collect::<Vec<_>>()
            .join(", ")
    );
    kept
}

fn strip_system_message(message: Message, active: &BetaSet) -> (Option<Message>, Vec<GatedField>) {
    let clear_at_gated = message.extra.contains_key(CLEAR_AT) && GatedField::ClearAt.gated(active);
    let output_config_gated =
        message.extra.contains_key(OUTPUT_CONFIG) && GatedField::OutputConfig.gated(active);
    let tool_changes_gated =
        message.blocks().iter().any(is_tool_change_block) && GatedField::ToolChanges.gated(active);
    let gated: Vec<GatedField> = [
        (clear_at_gated, GatedField::ClearAt),
        (output_config_gated, GatedField::OutputConfig),
        (tool_changes_gated, GatedField::ToolChanges),
    ]
    .into_iter()
    .filter_map(|(gated, field)| gated.then_some(field))
    .collect();
    if gated.is_empty() {
        return (Some(message), gated);
    }
    let Message {
        role,
        content,
        extra,
    } = message;
    let content = match content {
        MessageContent::Blocks(blocks) if tool_changes_gated => MessageContent::Blocks(
            blocks
                .into_iter()
                .filter(|block| !is_tool_change_block(block))
                .collect(),
        ),
        content => content,
    };
    let extra: serde_json::Map<String, serde_json::Value> = extra
        .into_iter()
        .filter(|(key, _)| {
            !((clear_at_gated && key == CLEAR_AT) || (output_config_gated && key == OUTPUT_CONFIG))
        })
        .collect();
    let emptied = match &content {
        MessageContent::Text(text) => text.is_empty(),
        MessageContent::Blocks(blocks) => blocks.is_empty(),
    };
    if emptied && !extra.contains_key(OUTPUT_CONFIG) {
        return (None, [gated, vec![GatedField::Emptied]].concat());
    }
    (
        Some(Message {
            role,
            content,
            extra,
        }),
        gated,
    )
}

/// The same strip over a `role: "system"` message that did not parse, such as one sent without
/// content: Python reads the fields off any mapping, and a missing or `null` content counts as
/// empty.
fn strip_unrecognized_system_message(
    fields: Map<String, Value>,
    active: &BetaSet,
) -> (Option<Map<String, Value>>, Vec<GatedField>) {
    let clear_at_gated = fields.contains_key(CLEAR_AT) && GatedField::ClearAt.gated(active);
    let output_config_gated =
        fields.contains_key(OUTPUT_CONFIG) && GatedField::OutputConfig.gated(active);
    let tool_changes_gated = matches!(
        fields.get(CONTENT),
        Some(Value::Array(blocks)) if blocks.iter().any(is_raw_tool_change_block)
    ) && GatedField::ToolChanges.gated(active);
    let gated: Vec<GatedField> = [
        (clear_at_gated, GatedField::ClearAt),
        (output_config_gated, GatedField::OutputConfig),
        (tool_changes_gated, GatedField::ToolChanges),
    ]
    .into_iter()
    .filter_map(|(gated, field)| gated.then_some(field))
    .collect();
    if gated.is_empty() {
        return (Some(fields), gated);
    }
    let rewritten: Map<String, Value> = fields
        .into_iter()
        .filter(|(key, _)| {
            !((clear_at_gated && key == CLEAR_AT) || (output_config_gated && key == OUTPUT_CONFIG))
        })
        .map(|(key, value)| match value {
            Value::Array(blocks) if tool_changes_gated && key == CONTENT => (
                key,
                Value::Array(
                    blocks
                        .into_iter()
                        .filter(|block| !is_raw_tool_change_block(block))
                        .collect(),
                ),
            ),
            value => (key, value),
        })
        .collect();
    let emptied = match rewritten.get(CONTENT) {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty(),
        Some(Value::Array(blocks)) => blocks.is_empty(),
        Some(_) => false,
    };
    if emptied && !rewritten.contains_key(OUTPUT_CONFIG) {
        return (None, [gated, vec![GatedField::Emptied]].concat());
    }
    (Some(rewritten), gated)
}

fn is_raw_tool_change_block(block: &Value) -> bool {
    block
        .get(TYPE)
        .and_then(Value::as_str)
        .is_some_and(|block_type| TOOL_CHANGE_BLOCK_TYPES.contains(&block_type))
}

fn is_tool_change_block(block: &ContentBlock) -> bool {
    matches!(
        &block.block_type,
        Some(ContentBlockType::Other(block_type))
            if TOOL_CHANGE_BLOCK_TYPES.contains(&block_type.as_str())
    )
}

fn ends_on_assistant(messages: &[Recognized<Message>]) -> bool {
    messages.last().and_then(message_role) == Some(ASSISTANT_ROLE)
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::{
        formats::messages::Message,
        providers::anthropic::{BetaProvider, BetaSet},
        recognized::Recognized,
    };
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::strip_beta_gated_system_messages;

    fn strip(
        messages: Vec<Recognized<Message>>,
        active: &BetaSet,
        drops_unsupported_params: bool,
    ) -> Vec<Recognized<Message>> {
        strip_beta_gated_system_messages(
            messages,
            active,
            drops_unsupported_params,
            "test-model",
            BetaProvider::AzureAi,
        )
    }

    fn messages(value: Value) -> Vec<Recognized<Message>> {
        serde_json::from_value(value).unwrap()
    }

    fn betas(header: &str) -> BetaSet {
        header.parse().unwrap_or_else(|never| match never {})
    }

    fn conversation(system: Value) -> Value {
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            system,
            {"role": "user", "content": "go on"}
        ])
    }

    fn tool_addition() -> Value {
        json!({"type": "tool_addition", "tool": {"name": "t", "input_schema": {"type": "object"}}})
    }

    #[rstest]
    #[case::clear_at_without_its_beta(
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}),
        "",
        json!({"role": "system", "content": "note"})
    )]
    #[case::clear_at_with_its_beta(
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}),
        "mid-conversation-system-clear-at-2026-08-21",
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}})
    )]
    #[case::output_config_without_a_beta(
        json!({"role": "system", "content": "note", "output_config": {"effort": "low"}}),
        "",
        json!({"role": "system", "content": "note"})
    )]
    #[case::output_config_with_per_turn_control(
        json!({"role": "system", "content": "note", "output_config": {"effort": "low"}}),
        "per-turn-control-2026-07-01",
        json!({"role": "system", "content": "note", "output_config": {"effort": "low"}})
    )]
    #[case::output_config_with_the_mid_conversation_beta(
        json!({"role": "system", "content": "note", "output_config": {"effort": "low"}}),
        "mid-conversation-output-config-2026-07-01",
        json!({"role": "system", "content": "note", "output_config": {"effort": "low"}})
    )]
    #[case::tool_change_without_a_beta(
        json!({"role": "system", "content": [{"type": "text", "text": "note"}, tool_addition()]}),
        "context-1m-2025-08-07",
        json!({"role": "system", "content": [{"type": "text", "text": "note"}]})
    )]
    #[case::tool_removal_without_a_beta(
        json!({"role": "system", "content": [{"type": "text", "text": "note"}, {"type": "tool_removal", "tool": {"type": "tool_reference", "tool_name": "t"}}]}),
        "",
        json!({"role": "system", "content": [{"type": "text", "text": "note"}]})
    )]
    #[case::tool_change_with_the_inline_tools_beta(
        json!({"role": "system", "content": [{"type": "text", "text": "note"}, tool_addition()]}),
        "inline-tools-2026-09-15",
        json!({"role": "system", "content": [{"type": "text", "text": "note"}, tool_addition()]})
    )]
    #[case::emptied_turn_keeping_output_config(
        json!({"role": "system", "content": [tool_addition()], "output_config": {"effort": "low"}}),
        "per-turn-control-2026-07-01",
        json!({"role": "system", "content": [], "output_config": {"effort": "low"}})
    )]
    #[case::contentless_turn_keeping_output_config(
        json!({"role": "system", "clear_at": {"type": "turns"}, "output_config": {"effort": "low"}}),
        "per-turn-control-2026-07-01",
        json!({"role": "system", "output_config": {"effort": "low"}})
    )]
    #[case::unparsed_content_losing_a_tool_change(
        json!({"role": "system", "content": [{"type": "text", "text": 5}, tool_addition()]}),
        "",
        json!({"role": "system", "content": [{"type": "text", "text": 5}]})
    )]
    fn gated_system_fields_are_dropped(
        #[case] system: Value,
        #[case] active: &str,
        #[case] expected: Value,
    ) {
        assert_eq!(
            strip(messages(conversation(system)), &betas(active), true),
            messages(conversation(expected))
        );
    }

    #[rstest]
    #[case::clear_at(json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}))]
    #[case::tool_change(json!({"role": "system", "content": [tool_addition()]}))]
    fn without_drop_params_the_messages_are_sent_as_is(#[case] system: Value) {
        assert_eq!(
            strip(
                messages(conversation(system.clone())),
                &BetaSet::default(),
                false
            ),
            messages(conversation(system))
        );
    }

    #[rstest]
    #[case::only_tool_changes(json!([tool_addition()]))]
    #[case::empty_text(json!(""))]
    fn an_emptied_system_turn_is_dropped(#[case] content: Value) {
        assert_eq!(
            strip(
                messages(conversation(
                    json!({"role": "system", "content": content, "clear_at": {"type": "turns", "value": 1}})
                )),
                &BetaSet::default(),
                true
            ),
            messages(json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "go on"}
            ]))
        );
    }

    #[rstest]
    #[case::missing_content(json!({"role": "system", "clear_at": {"type": "turns"}}))]
    #[case::null_content(json!({"role": "system", "content": null, "clear_at": {"type": "turns"}}))]
    #[case::only_unparsed_tool_changes(
        json!({"role": "system", "content": [{"type": "tool_removal", "text": 5}], "output_config": {"effort": "low"}})
    )]
    fn an_emptied_unparsed_system_turn_is_dropped(#[case] system: Value) {
        assert_eq!(
            strip(messages(conversation(system)), &BetaSet::default(), true),
            messages(json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "go on"}
            ]))
        );
    }

    #[rstest]
    fn an_emptied_turn_that_would_leave_an_unparsed_assistant_prefill_keeps_every_message() {
        let sent = messages(json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null},
            {"role": "system", "content": "", "clear_at": {"type": "turns"}}
        ]));
        assert_eq!(strip(sent.clone(), &BetaSet::default(), true), sent);
    }

    #[rstest]
    fn an_emptied_turn_that_would_leave_an_assistant_prefill_keeps_every_message() {
        let sent = messages(json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": [tool_addition()], "clear_at": {"type": "turns", "value": 1}}
        ]));
        assert_eq!(strip(sent.clone(), &BetaSet::default(), true), sent);
    }

    #[rstest]
    fn non_system_messages_keep_their_fields() {
        let sent = messages(json!([
            {"role": "user", "content": [tool_addition()], "clear_at": {"type": "turns", "value": 1}}
        ]));
        assert_eq!(strip(sent.clone(), &BetaSet::default(), true), sent);
    }
}
