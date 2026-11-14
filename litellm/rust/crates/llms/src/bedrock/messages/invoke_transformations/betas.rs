use std::collections::BTreeSet;

use litellm_http::request::{with_header, without_headers};
use litellm_llms_types::providers::anthropic::{AnthropicBeta, BETA_HEADER, BetaProvider};
use serde_json::Value;

use crate::{
    Error, ErrorDetail,
    base_llm::{auth::Headers, messages::context::MessagesModelCapabilities},
};

use super::request::{Body, python_truthy};

pub(super) const BETA_GATE_EMPTY: BTreeSet<String> = BTreeSet::new();

const PER_TURN_CONTROL: &str = "per-turn-control-2026-07-01";
const DANGEROUS_TOOL_USE: &str = "dangerous-tool-use-2026-09-03";
const TOOL_SEARCH_TOOL: &str = "tool-search-tool-2025-10-19";
const ADVANCED_TOOL_USE: &str = "advanced-tool-use-2025-11-20";
const TOOL_EXAMPLES: &str = "tool-examples-2025-10-29";
const COMPACT_2026_09_04: &str = "compact-2026-09-04";
const THINKING_BINDING_CONTROLS: &str = "thinking-binding-controls-2026-08-01";
const THINKING_DISPLAY_UPDATES: &str = "thinking-display-updates-2026-08-18";
const MID_CONVERSATION_SYSTEM_CLEAR_AT: &str = "mid-conversation-system-clear-at-2026-08-21";
const MID_CONVERSATION_OUTPUT_CONFIG: &str = "mid-conversation-output-config-2026-07-01";
const MID_CONVERSATION_TOOL_CHANGES: &str = "mid-conversation-tool-changes-2026-07-01";
const INLINE_TOOLS: &str = "inline-tools-2026-09-15";
const PROGRAMMATIC_CALLER: &str = "code_execution_20250825";
const TOOL_SEARCH_TYPES: [&str; 2] = [
    "tool_search_tool_regex_20251119",
    "tool_search_tool_bm25_20251119",
];
const CONTEXT_MANAGEMENT_EDIT_BETAS: [(&str, &str); 2] = [
    ("compact_20260112", "compact-2026-01-12"),
    ("clear_tool_uses_20250919", "context-management-2025-06-27"),
];
const COMPUTER_TOOL_BETAS: [(&str, &str); 3] = [
    ("computer_20251124", "computer-use-2025-11-24"),
    ("computer_20250124", "computer-use-2025-01-24"),
    ("computer_20241022", "computer-use-2024-10-22"),
];
const OPUS_4_5_EFFORT_PATTERNS: [&str; 2] = ["opus-4-5", "opus_4_5"];

fn on_bedrock(beta: &str) -> Option<String> {
    beta.parse::<AnthropicBeta>()
        .unwrap_or_else(|never| match never {})
        .on(BetaProvider::Bedrock)
        .map(|beta| beta.as_str().to_string())
}

fn keep_set(capabilities: &MessagesModelCapabilities) -> BTreeSet<&'static str> {
    if capabilities.bedrock.supports_per_turn_control {
        BTreeSet::from([PER_TURN_CONTROL])
    } else {
        BTreeSet::new()
    }
}

fn header_value<'a>(headers: &'a Headers, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

pub(super) fn filtered_beta_header(headers: Headers, keep: &BTreeSet<&'static str>) -> Headers {
    let Some(existing) = header_value(&headers, BETA_HEADER).map(str::to_string) else {
        return headers;
    };
    let values: Vec<&str> = existing
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    let filtered: BTreeSet<String> = values
        .iter()
        .filter(|value| !keep.contains(*value))
        .filter_map(|value| on_bedrock(value))
        .collect();
    let mut kept: Vec<String> = filtered.into_iter().collect();
    values
        .iter()
        .filter(|value| keep.contains(*value))
        .for_each(|value| {
            if !kept.iter().any(|existing| existing == value) {
                kept.push((*value).to_string());
            }
        });
    let headers = without_headers(headers, &[BETA_HEADER]);
    if kept.is_empty() {
        return headers;
    }
    with_header(headers, BETA_HEADER, kept.join(","))
}

fn user_betas(headers: &Headers) -> BTreeSet<String> {
    let Some(header) = header_value(headers, BETA_HEADER).filter(|value| !value.is_empty()) else {
        return BTreeSet::new();
    };
    if header.starts_with('[')
        && header.ends_with(']')
        && let Ok(Value::Array(values)) = serde_json::from_str::<Value>(header)
    {
        return values
            .iter()
            .map(|value| match value {
                Value::String(text) => text.trim().to_string(),
                other => other.to_string(),
            })
            .filter(|value| !value.is_empty())
            .collect();
    }
    header
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn tools(value: Option<&Value>) -> &[Value] {
    match value {
        Some(Value::Array(tools)) => tools,
        _ => &[],
    }
}

fn tool_type(tool: &Value) -> Option<&str> {
    tool.get("type").and_then(Value::as_str)
}

fn lists_programmatic_caller(value: Option<&Value>) -> bool {
    matches!(value, Some(Value::Array(callers)) if callers.iter().any(|caller| caller.as_str() == Some(PROGRAMMATIC_CALLER)))
}

fn programmatic_tool_calling_used(tools: &[Value]) -> bool {
    tools.iter().any(|tool| {
        lists_programmatic_caller(tool.get("allowed_callers"))
            || lists_programmatic_caller(
                tool.get("function")
                    .and_then(|function| function.get("allowed_callers")),
            )
    })
}

fn non_empty_list(value: Option<&Value>) -> bool {
    matches!(value, Some(Value::Array(items)) if !items.is_empty())
}

fn input_examples_used(tools: &[Value]) -> bool {
    tools.iter().any(|tool| {
        non_empty_list(tool.get("input_examples"))
            || non_empty_list(
                tool.get("function")
                    .and_then(|function| function.get("input_examples")),
            )
    })
}

fn eager_input_streaming_flag(tool: &Value) -> Result<Option<bool>, Error> {
    let Value::Object(tool) = tool else {
        return Ok(None);
    };
    let strict_bool = |value: Option<&Value>| match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(flag)) => Ok(Some(*flag)),
        Some(_) => Err(Error::InvalidRequest(ErrorDetail::Expected {
            subject: "eager_input_streaming",
            expected: "a boolean",
        })),
    };
    let top_level = strict_bool(tool.get("eager_input_streaming"))?;
    let function = match tool.get("function") {
        None | Some(Value::Null) => None,
        Some(Value::Object(function)) => strict_bool(function.get("eager_input_streaming"))?,
        Some(_) => {
            return Err(Error::InvalidRequest(ErrorDetail::Expected {
                subject: "eager_input_streaming",
                expected: "a boolean",
            }));
        }
    };
    Ok(top_level.or(function))
}

fn eager_input_streaming_used(tools: &[Value]) -> Result<bool, Error> {
    tools.iter().try_fold(false, |used, tool| {
        Ok(eager_input_streaming_flag(tool)? == Some(true) || used)
    })
}

fn effort_used(params: &Body, model: &str, capabilities: &MessagesModelCapabilities) -> bool {
    if params.is_empty() || capabilities.supports_adaptive_thinking {
        return false;
    }
    let lowered = model.to_lowercase();
    let opus_4_5_effort = OPUS_4_5_EFFORT_PATTERNS
        .iter()
        .any(|pattern| lowered.contains(pattern))
        && params
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .is_some_and(|effort| !effort.is_empty());
    let output_config_effort = params
        .get("output_config")
        .and_then(|output_config| output_config.get("effort"))
        .and_then(Value::as_str)
        .is_some_and(|effort| !effort.is_empty());
    opus_4_5_effort || output_config_effort
}

fn system_messages(messages: &[Value]) -> impl Iterator<Item = &Value> {
    messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
}

fn mid_conversation_tool_change_used(messages: &[Value]) -> bool {
    system_messages(messages).any(|message| {
        tools(message.get("content")).iter().any(|block| {
            matches!(tool_type(block), Some("tool_addition" | "tool_removal"))
                && block.get("tool").and_then(tool_type) == Some("tool_reference")
        })
    })
}

fn thinking_betas(thinking: Option<&Value>) -> impl Iterator<Item = &'static str> {
    let display = super::request::is_display_updates(thinking).then_some(THINKING_DISPLAY_UPDATES);
    let binding =
        matches!(thinking, Some(Value::Object(thinking)) if thinking.contains_key("block_binding"))
            .then_some(THINKING_BINDING_CONTROLS);
    display.into_iter().chain(binding)
}

fn carries_signed_compaction_block(messages: &[Value]) -> bool {
    messages
        .iter()
        .flat_map(|message| tools(message.get("content")))
        .any(|block| {
            tool_type(block) == Some("compaction")
                && block
                    .get("signature")
                    .and_then(Value::as_str)
                    .is_some_and(|signature| !signature.is_empty())
        })
}

fn filter_context_management(body: &mut Body, betas: &mut BTreeSet<String>) {
    let Some(Value::Object(context_management)) = body.get("context_management") else {
        return;
    };
    let Some(Value::Array(edits)) = context_management.get("edits") else {
        body.shift_remove("context_management");
        return;
    };
    let retained: Vec<(Value, &'static str)> = edits
        .iter()
        .filter_map(|edit| {
            let edit_type = edit.as_object()?.get("type")?.as_str()?;
            CONTEXT_MANAGEMENT_EDIT_BETAS
                .iter()
                .find(|(supported, _)| *supported == edit_type)
                .map(|(_, beta)| (edit.clone(), *beta))
        })
        .collect();
    if retained.is_empty() {
        body.shift_remove("context_management");
        return;
    }
    let mut filtered = context_management.clone();
    betas.extend(retained.iter().map(|(_, beta)| (*beta).to_string()));
    filtered.insert(
        "edits".into(),
        Value::Array(retained.into_iter().map(|(edit, _)| edit).collect()),
    );
    body.insert("context_management".into(), Value::Object(filtered));
}

pub(super) fn bedrock_invoke_betas(
    body: &mut Body,
    params: &Body,
    headers: Headers,
    model: &str,
    capabilities: &MessagesModelCapabilities,
    injected_thinking: bool,
) -> Result<(Vec<String>, Headers), Error> {
    let keep = keep_set(capabilities);
    let headers = filtered_beta_header(headers, &keep);
    let user = user_betas(&headers);
    let mut betas = user.clone();
    let outgoing_tools = tools(body.get("tools")).to_vec();
    let messages = tools(body.get("messages")).to_vec();
    if effort_used(params, model, capabilities) {
        betas.insert("effort-2025-11-24".into());
    }
    if let Some(beta) = outgoing_tools
        .iter()
        .find_map(|tool| tool_type(tool).filter(|kind| kind.starts_with("computer_")))
        .and_then(|kind| COMPUTER_TOOL_BETAS.iter().find(|(tool, _)| *tool == kind))
        .map(|(_, beta)| *beta)
    {
        betas.insert(beta.into());
    }
    if params.get("mcp_servers").is_some_and(python_truthy) {
        betas.insert("mcp-client-2025-04-04".into());
    }
    if messages.iter().any(|message| {
        message
            .as_object()
            .is_some_and(|message| message.contains_key("output_config"))
    }) {
        betas.insert(MID_CONVERSATION_OUTPUT_CONFIG.into());
    }
    betas.extend(thinking_betas(body.get("thinking")).map(str::to_string));
    if mid_conversation_tool_change_used(&messages) {
        betas.insert(MID_CONVERSATION_TOOL_CHANGES.into());
    }
    if injected_thinking {
        betas.insert("interleaved-thinking-2025-05-14".into());
    }
    if eager_input_streaming_used(&outgoing_tools)? {
        betas.insert("fine-grained-tool-streaming-2025-05-14".into());
    }
    if params
        .get("safeguards")
        .is_some_and(|safeguards| !safeguards.is_null())
    {
        betas.insert(DANGEROUS_TOOL_USE.into());
    } else {
        betas.remove(DANGEROUS_TOOL_USE);
    }
    filter_context_management(body, &mut betas);
    if body.get("compaction").is_some_and(Value::is_object)
        || carries_signed_compaction_block(&messages)
    {
        betas.insert(COMPACT_2026_09_04.into());
    }
    let tool_search_used = outgoing_tools
        .iter()
        .any(|tool| tool_type(tool).is_some_and(|kind| TOOL_SEARCH_TYPES.contains(&kind)));
    if tool_search_used
        && !(programmatic_tool_calling_used(&outgoing_tools)
            || input_examples_used(&outgoing_tools))
    {
        betas.remove(ADVANCED_TOOL_USE);
        if capabilities.bedrock.supports_tool_search {
            betas.insert(TOOL_SEARCH_TOOL.into());
        }
    }
    if betas.contains(TOOL_SEARCH_TOOL) {
        betas.insert(TOOL_EXAMPLES.into());
    }
    let model_supported = user.iter().filter(|beta| keep.contains(beta.as_str()));
    let filtered: BTreeSet<String> = betas
        .iter()
        .filter_map(|beta| on_bedrock(beta))
        .chain(model_supported.cloned())
        .collect();
    let dropped: Vec<&String> = user
        .iter()
        .filter(|beta| !keep.contains(beta.as_str()) && on_bedrock(beta).is_none())
        .collect();
    if !dropped.is_empty() {
        litellm_tracing::warn!(
            "Bedrock Invoke: dropping unsupported anthropic-beta values from client headers: {dropped:?}. Bedrock has no mapping entry for these; forwarding them would cause a 400."
        );
    }
    Ok((filtered.into_iter().collect(), headers))
}

fn unlocked(active: &BTreeSet<String>, betas: &[&str]) -> bool {
    betas.iter().any(|beta| active.contains(*beta))
}

fn is_tool_change_block(block: &Value) -> bool {
    matches!(tool_type(block), Some("tool_addition" | "tool_removal"))
}

fn stripped_system_message(message: &Value, active: &BTreeSet<String>) -> Option<Option<Value>> {
    let fields = message.as_object()?;
    if fields.get("role").and_then(Value::as_str) != Some("system") {
        return None;
    }
    let clear_at_gated =
        fields.contains_key("clear_at") && !unlocked(active, &[MID_CONVERSATION_SYSTEM_CLEAR_AT]);
    let output_config_gated = fields.contains_key("output_config")
        && !unlocked(active, &[MID_CONVERSATION_OUTPUT_CONFIG, PER_TURN_CONTROL]);
    let blocks = match fields.get("content") {
        Some(Value::Array(blocks)) => Some(blocks),
        _ => None,
    };
    let tool_changes_gated = blocks.is_some_and(|blocks| blocks.iter().any(is_tool_change_block))
        && !unlocked(active, &[MID_CONVERSATION_TOOL_CHANGES, INLINE_TOOLS]);
    if !clear_at_gated && !output_config_gated && !tool_changes_gated {
        return None;
    }
    let mut rewritten = fields.clone();
    if clear_at_gated {
        rewritten.shift_remove("clear_at");
    }
    if output_config_gated {
        rewritten.shift_remove("output_config");
    }
    if tool_changes_gated && let Some(blocks) = blocks {
        rewritten.insert(
            "content".into(),
            Value::Array(
                blocks
                    .iter()
                    .filter(|block| !is_tool_change_block(block))
                    .cloned()
                    .collect(),
            ),
        );
    }
    let emptied = match rewritten.get("content") {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty(),
        Some(Value::Array(blocks)) => blocks.is_empty(),
        Some(_) => false,
    };
    Some((!emptied || rewritten.contains_key("output_config")).then_some(Value::Object(rewritten)))
}

fn ends_on_assistant(messages: &[Value]) -> bool {
    messages
        .last()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("assistant")
}

pub(super) fn strip_beta_gated_system_messages(
    messages: Vec<Value>,
    active: &BTreeSet<String>,
) -> Vec<Value> {
    let mut any_stripped = false;
    let kept: Vec<Value> = messages
        .iter()
        .filter_map(|message| match stripped_system_message(message, active) {
            None => Some(message.clone()),
            Some(rewritten) => {
                any_stripped = true;
                rewritten
            }
        })
        .collect();
    if !any_stripped || (ends_on_assistant(&kept) && !ends_on_assistant(&messages)) {
        return messages;
    }
    kept
}

pub(super) fn strip_beta_gated_request(body: &mut Body, betas: &[String]) {
    let active: BTreeSet<String> = betas.iter().cloned().collect();
    if let Some(Value::Object(thinking)) = body.get("thinking") {
        let binding_gated = thinking.contains_key("block_binding")
            && !unlocked(&active, &[THINKING_BINDING_CONTROLS]);
        let display_gated = thinking.get("display").and_then(Value::as_str) == Some("updates")
            && !unlocked(&active, &[THINKING_DISPLAY_UPDATES]);
        if binding_gated || display_gated {
            let mut stripped = thinking.clone();
            if binding_gated {
                stripped.shift_remove("block_binding");
            }
            if display_gated {
                stripped.shift_remove("display");
            }
            body.insert("thinking".into(), Value::Object(stripped));
        }
    }
    if let Some(Value::Array(messages)) = body.get_mut("messages") {
        let taken = std::mem::take(messages);
        *messages = strip_beta_gated_system_messages(taken, &active);
    }
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::messages::MessagesRequest;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        base_llm::messages::{
            context::{BedrockCapabilities, MessagesTransformContext},
            transformation::BaseMessagesConfig,
        },
        bedrock::messages::invoke_transformations::anthropic_claude_transformation::AmazonAnthropicClaudeMessagesConfig,
    };

    fn sent(
        params: Value,
        capabilities: MessagesModelCapabilities,
        header: Option<&str>,
        drop_params: bool,
    ) -> (Value, Headers) {
        let mut request = json!({
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "max_tokens": 128,
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(params.as_object().unwrap().clone());
        let request: MessagesRequest = serde_json::from_value(request).unwrap();
        let context = MessagesTransformContext {
            drops_unsupported_params: drop_params,
            ..MessagesTransformContext::new(capabilities, drop_params)
        };
        let headers = header
            .map(|value| vec![("anthropic-beta".to_string(), value.to_string())])
            .unwrap_or_default();
        let (request, headers) = AmazonAnthropicClaudeMessagesConfig
            .transform_request_with_headers(request, &context, headers)
            .unwrap();
        (serde_json::to_value(request).unwrap(), headers)
    }

    fn betas(
        params: Value,
        capabilities: MessagesModelCapabilities,
        header: Option<&str>,
    ) -> Value {
        sent(params, capabilities, header, false)
            .0
            .get("anthropic_beta")
            .cloned()
            .unwrap_or(Value::Null)
    }

    fn bedrock(bedrock: BedrockCapabilities) -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            bedrock,
            ..MessagesModelCapabilities::default()
        }
    }

    #[rstest]
    #[case::unlisted_beta_dropped("advisor-tool-2026-03-01,context-1m-2025-08-07", json!(["context-1m-2025-08-07"]))]
    #[case::aliased_beta_renamed("advanced-tool-use-2025-11-20", json!(["tool-search-tool-2025-10-19"]))]
    #[case::unknown_beta_dropped("not-a-real-beta-2099-01-01", Value::Null)]
    #[case::dangerous_tool_use_dropped_without_safeguards(
        "dangerous-tool-use-2026-09-03,interleaved-thinking-2025-05-14",
        Value::Null
    )]
    fn user_betas_go_through_the_bedrock_column(#[case] header: &str, #[case] expected: Value) {
        assert_eq!(
            betas(
                json!({}),
                MessagesModelCapabilities::default(),
                Some(header)
            ),
            expected
        );
    }

    #[rstest]
    #[case::client_sends_beta("dangerous-tool-use-2026-09-03,interleaved-thinking-2025-05-14")]
    #[case::client_omits_beta("interleaved-thinking-2025-05-14")]
    fn safeguards_carry_the_dangerous_tool_use_beta(#[case] header: &str) {
        let (body, _) = sent(
            json!({"safeguards": {"mode": "block"}}),
            MessagesModelCapabilities::default(),
            Some(header),
            false,
        );
        assert_eq!(
            body["anthropic_beta"],
            json!(["dangerous-tool-use-2026-09-03"])
        );
        assert_eq!(body["safeguards"], json!({"mode": "block"}));
    }

    #[rstest]
    #[case::flagged(true, json!(["per-turn-control-2026-07-01"]), Some("per-turn-control-2026-07-01"))]
    #[case::not_flagged(false, Value::Null, None)]
    fn per_turn_control_is_kept_only_where_the_model_flags_it(
        #[case] supports_per_turn_control: bool,
        #[case] expected: Value,
        #[case] header: Option<&str>,
    ) {
        let (body, headers) = sent(
            json!({}),
            bedrock(BedrockCapabilities {
                supports_per_turn_control,
                ..BedrockCapabilities::default()
            }),
            Some("per-turn-control-2026-07-01"),
            false,
        );
        assert_eq!(
            body.get("anthropic_beta").cloned().unwrap_or(Value::Null),
            expected
        );
        assert_eq!(
            headers
                .iter()
                .find(|(name, _)| name == "anthropic-beta")
                .map(|(_, value)| value.as_str()),
            header
        );
    }

    #[rstest]
    #[case::supported(true, json!(["tool-search-tool-2025-10-19"]))]
    #[case::unsupported(false, Value::Null)]
    fn tool_search_beta_follows_the_model_map(
        #[case] supports_tool_search: bool,
        #[case] expected: Value,
    ) {
        assert_eq!(
            betas(
                json!({"tools": [{"type": "tool_search_tool_regex_20251119", "name": "tool_search"}]}),
                bedrock(BedrockCapabilities {
                    supports_tool_search,
                    ..BedrockCapabilities::default()
                }),
                None,
            ),
            expected
        );
    }

    #[test]
    fn tool_search_with_programmatic_calling_keeps_the_users_beta() {
        assert_eq!(
            betas(
                json!({"tools": [
                    {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
                    {"name": "t", "input_schema": {"type": "object"}, "allowed_callers": ["code_execution_20250825"]},
                ]}),
                MessagesModelCapabilities::default(),
                Some("advanced-tool-use-2025-11-20"),
            ),
            json!(["tool-search-tool-2025-10-19"])
        );
    }

    #[rstest]
    #[case::computer_20250124("computer_20250124", json!(["computer-use-2025-01-24"]))]
    #[case::computer_20251124("computer_20251124", json!(["computer-use-2025-11-24"]))]
    #[case::computer_20241022_is_rejected("computer_20241022", Value::Null)]
    fn computer_tools_add_their_beta(#[case] tool_type: &str, #[case] expected: Value) {
        assert_eq!(
            betas(
                json!({"tools": [{"type": tool_type, "name": "computer", "display_width_px": 1, "display_height_px": 1}]}),
                MessagesModelCapabilities::default(),
                None,
            ),
            expected
        );
    }

    #[test]
    fn output_config_effort_adds_the_effort_beta_on_non_adaptive_models() {
        assert_eq!(
            betas(
                json!({"output_config": {"effort": "high"}}),
                MessagesModelCapabilities {
                    supports_output_config: true,
                    ..MessagesModelCapabilities::default()
                },
                None,
            ),
            json!(["effort-2025-11-24"])
        );
    }

    #[rstest]
    #[case::display_updates(json!({"type": "adaptive", "display": "updates"}), json!(["thinking-display-updates-2026-08-18"]))]
    #[case::block_binding(json!({"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}), json!(["thinking-binding-controls-2026-08-01"]))]
    fn thinking_controls_add_their_betas(#[case] thinking: Value, #[case] expected: Value) {
        let capabilities = MessagesModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            ..MessagesModelCapabilities::default()
        };
        assert_eq!(
            betas(json!({"thinking": thinking}), capabilities, None),
            expected
        );
    }

    #[test]
    fn mid_conversation_output_config_keeps_its_auto_added_beta() {
        let (body, _) = sent(
            json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
                {"role": "system", "content": "Answer briefly", "output_config": {"effort": "high"}},
                {"role": "user", "content": [{"type": "text", "text": "Go on"}]},
            ]}),
            MessagesModelCapabilities {
                supports_mid_conversation_system: true,
                ..MessagesModelCapabilities::default()
            },
            None,
            true,
        );
        assert_eq!(
            body["anthropic_beta"],
            json!(["mid-conversation-output-config-2026-07-01"])
        );
        assert_eq!(
            body["messages"][2]["output_config"],
            json!({"effort": "high"})
        );
    }

    #[test]
    fn mid_conversation_tool_changes_add_their_beta() {
        let (body, _) = sent(
            json!({"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "tool_name": "x"}}]},
                {"role": "user", "content": "Go on"},
            ]}),
            MessagesModelCapabilities {
                supports_mid_conversation_system: true,
                ..MessagesModelCapabilities::default()
            },
            None,
            true,
        );
        assert_eq!(
            body["anthropic_beta"],
            json!(["mid-conversation-tool-changes-2026-07-01"])
        );
    }

    #[rstest]
    #[case::clear_at_stripped(
        json!({"role": "system", "content": "note", "clear_at": 3}),
        Some(json!({"role": "system", "content": "note"})),
    )]
    #[case::emptied_turn_dropped(
        json!({"role": "system", "content": [{"type": "tool_removal", "tool": {"type": "custom"}}]}),
        None,
    )]
    fn beta_gated_system_fields_are_stripped_under_drop_params(
        #[case] system: Value,
        #[case] expected: Option<Value>,
    ) {
        let messages = vec![
            json!({"role": "user", "content": "a"}),
            system,
            json!({"role": "user", "content": "b"}),
        ];
        let stripped = strip_beta_gated_system_messages(messages, &BETA_GATE_EMPTY);
        assert_eq!(
            stripped
                .get(1)
                .filter(|message| message["role"] == "system")
                .cloned(),
            expected
        );
    }

    #[test]
    fn a_strip_that_would_end_on_an_assistant_turn_is_undone() {
        let messages = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": "b"}),
            json!({"role": "system", "content": [], "clear_at": 1}),
        ];
        assert_eq!(
            strip_beta_gated_system_messages(messages.clone(), &BETA_GATE_EMPTY),
            messages
        );
    }

    #[rstest]
    #[case::kept_with_its_beta(&["thinking-binding-controls-2026-08-01"], json!({"type": "adaptive", "block_binding": {}}))]
    #[case::stripped_without_it(&[], json!({"type": "adaptive"}))]
    fn block_binding_needs_its_beta(#[case] active: &[&str], #[case] expected: Value) {
        let mut body: Body = json!({"thinking": {"type": "adaptive", "block_binding": {}}})
            .as_object()
            .unwrap()
            .clone();
        strip_beta_gated_request(
            &mut body,
            &active
                .iter()
                .map(|beta| beta.to_string())
                .collect::<Vec<_>>(),
        );
        assert_eq!(body["thinking"], expected);
    }

    #[test]
    fn the_header_keeps_kept_betas_after_the_sorted_filtered_ones() {
        let headers = filtered_beta_header(
            vec![("anthropic-beta".into(), "per-turn-control-2026-07-01, effort-2025-11-24,files-api-2025-04-14,context-1m-2025-08-07".into())],
            &BTreeSet::from([PER_TURN_CONTROL]),
        );
        assert_eq!(
            headers,
            vec![(
                "anthropic-beta".to_string(),
                "context-1m-2025-08-07,effort-2025-11-24,per-turn-control-2026-07-01".to_string()
            )]
        );
    }
}
