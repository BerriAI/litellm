use litellm_types::llms::anthropic_messages::anthropic_request::{
    AnthropicMessage, ContentBlock, MessageContent,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::anthropic::ANTHROPIC_OAUTH_TOKEN_PREFIX;

pub const ANTHROPIC_OAUTH_BETA_HEADER: &str = "oauth-2025-04-20";
pub const ANTHROPIC_ADVISOR_TOOL_TYPE: &str = "advisor_20260301";
pub const ANTHROPIC_TOOL_SEARCH_TOOL_TYPES: [&str; 2] = [
    "tool_search_tool_regex_20251119",
    "tool_search_tool_bm25_20251119",
];
pub const ENCRYPTED_REASONING_SIGNATURE_PREFIX: &str = "litellm_encrypted_reasoning:";
const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

pub mod beta {
    pub const CONTEXT_MANAGEMENT_2025_06_27: &str = "context-management-2025-06-27";
    pub const COMPACT_2026_01_12: &str = "compact-2026-01-12";
    pub const COMPACT_2026_09_04: &str = "compact-2026-09-04";
    pub const STRUCTURED_OUTPUT: &str = "structured-outputs-2025-11-13";
    pub const ADVANCED_TOOL_USE_2025_11_20: &str = "advanced-tool-use-2025-11-20";
    pub const FAST_MODE_2026_02_01: &str = "fast-mode-2026-02-01";
    pub const ADVISOR_TOOL_2026_03_01: &str = "advisor-tool-2026-03-01";
    pub const PER_TURN_CONTROL_2026_07_01: &str = "per-turn-control-2026-07-01";
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl EffortLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::Xhigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportedEffortTiers {
    #[serde(default)]
    pub minimal: bool,
    #[serde(default)]
    pub low: bool,
    #[serde(default)]
    pub medium: bool,
    #[serde(default)]
    pub high: bool,
    #[serde(default)]
    pub xhigh: bool,
    #[serde(default)]
    pub max: bool,
}

impl SupportedEffortTiers {
    pub fn any(self) -> bool {
        self.minimal || self.low || self.medium || self.high || self.xhigh || self.max
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicModelCapabilities {
    #[serde(default)]
    pub supports_reasoning: bool,
    #[serde(default)]
    pub supports_adaptive_thinking: bool,
    #[serde(default)]
    pub thinking_always_on: bool,
    #[serde(default)]
    pub supports_legacy_thinking: bool,
    #[serde(default)]
    pub supports_output_config: bool,
    #[serde(default = "default_true")]
    pub supports_sampling_params: bool,
    #[serde(default)]
    pub supports_speed: bool,
    #[serde(default)]
    pub effort_tiers: SupportedEffortTiers,
}

fn default_true() -> bool {
    true
}

impl Default for AnthropicModelCapabilities {
    fn default() -> Self {
        Self {
            supports_reasoning: false,
            supports_adaptive_thinking: false,
            thinking_always_on: false,
            supports_legacy_thinking: false,
            supports_output_config: false,
            supports_sampling_params: true,
            supports_speed: false,
            effort_tiers: SupportedEffortTiers::default(),
        }
    }
}

impl AnthropicModelCapabilities {
    pub fn supports_effort_tier(&self, level: EffortLevel) -> bool {
        match level {
            EffortLevel::Low => self.effort_tiers.low,
            EffortLevel::Medium => self.effort_tiers.medium,
            EffortLevel::High => self.effort_tiers.high,
            EffortLevel::Xhigh => self.effort_tiers.xhigh,
            EffortLevel::Max => self.effort_tiers.max,
        }
    }

    pub fn supports_effort_param(&self) -> bool {
        self.supports_output_config || self.effort_tiers.any()
    }

    pub fn effort_level_rejection(&self, effort: &str, model: &str) -> Option<String> {
        match effort {
            "max" if !(self.supports_adaptive_thinking || self.effort_tiers.max) => Some(format!(
                "effort='max' is not supported by this model. Got model: {model}"
            )),
            "xhigh" if !self.effort_tiers.xhigh => Some(format!(
                "effort='xhigh' is not supported by this model. Got model: {model}"
            )),
            _ => None,
        }
    }
}

pub fn is_anthropic_oauth_key(value: &str) -> bool {
    value
        .strip_prefix("Bearer ")
        .unwrap_or(value)
        .starts_with(ANTHROPIC_OAUTH_TOKEN_PREFIX)
}

pub fn split_beta_values(header: Option<&str>) -> impl Iterator<Item = String> + '_ {
    header
        .into_iter()
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|piece| !piece.is_empty())
        .map(str::to_string)
}

pub fn join_beta_values(values: impl IntoIterator<Item = String>) -> String {
    let mut values: Vec<String> = values.into_iter().collect();
    values.sort();
    values.dedup();
    values.join(",")
}

pub fn is_tool_search_used(tools: Option<&[Value]>) -> bool {
    tools.into_iter().flatten().any(|tool| {
        tool.get("type")
            .and_then(Value::as_str)
            .is_some_and(|tool_type| ANTHROPIC_TOOL_SEARCH_TOOL_TYPES.contains(&tool_type))
    })
}

pub fn has_advisor_tool(tools: Option<&[Value]>) -> bool {
    tools
        .into_iter()
        .flatten()
        .any(|tool| tool.get("type").and_then(Value::as_str) == Some(ANTHROPIC_ADVISOR_TOOL_TYPE))
}

pub fn requires_native_compaction_beta(
    compaction: Option<&Value>,
    messages: &[AnthropicMessage],
) -> bool {
    compaction.is_some()
        || messages
            .iter()
            .flat_map(AnthropicMessage::blocks)
            .any(|block| {
                block.is_type("compaction")
                    && block.signature.as_deref().is_some_and(|s| !s.is_empty())
            })
}

fn is_blank(text: Option<&str>) -> bool {
    text.is_none_or(|text| text.trim().is_empty())
}

fn is_empty_text_block(block: &ContentBlock) -> bool {
    block.is_type("text") && is_blank(block.text.as_deref())
}

pub fn is_empty_thinking_block(block: &ContentBlock) -> bool {
    block.is_type("thinking") && is_blank(block.thinking.as_deref())
}

fn retain_blocks(
    messages: Vec<AnthropicMessage>,
    keep: impl Fn(&ContentBlock) -> bool,
) -> Vec<AnthropicMessage> {
    messages
        .into_iter()
        .filter_map(|message| match message.content {
            MessageContent::Text(_) => Some(message),
            MessageContent::Blocks(ref blocks) => {
                let kept: Vec<ContentBlock> =
                    blocks.iter().filter(|block| keep(block)).cloned().collect();
                if kept.len() == blocks.len() {
                    return Some(message);
                }
                (!kept.is_empty()).then(|| message.with_blocks(kept))
            }
        })
        .collect()
}

pub fn is_anthropic_invalid_thinking_block_error(error_text: &str) -> bool {
    let lower = error_text.to_lowercase();
    lower.contains("thinking")
        && ((lower.contains("signature")
            && (lower.contains("invalid") || lower.contains("valid string")))
            || lower.contains("must contain thinking"))
}

pub fn strip_thinking_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| {
        !block.is_type("thinking") && !block.is_type("redacted_thinking")
    })
}

pub fn strip_empty_content_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| {
        !is_empty_text_block(block) && !is_empty_thinking_block(block)
    })
}

pub fn normalize_anthropic_tool_use_id(raw_id: &str) -> String {
    let base = raw_id
        .split_once(THOUGHT_SIGNATURE_SEPARATOR)
        .map_or(raw_id, |(base, _)| base);
    let sanitized: String = base
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "tool_use_id".to_string()
    } else {
        sanitized
    }
}

fn normalized_if_changed(raw_id: Option<&str>) -> Option<String> {
    let raw_id = raw_id?;
    let normalized = normalize_anthropic_tool_use_id(raw_id);
    (normalized != raw_id).then_some(normalized)
}

fn sanitize_tool_use_id_block(block: ContentBlock) -> ContentBlock {
    match block.block_type.as_deref() {
        Some("tool_use" | "server_tool_use") => match normalized_if_changed(block.id.as_deref()) {
            Some(id) => ContentBlock {
                id: Some(id),
                ..block
            },
            None => block,
        },
        Some("tool_result") => match normalized_if_changed(block.tool_use_id.as_deref()) {
            Some(tool_use_id) => ContentBlock {
                tool_use_id: Some(tool_use_id),
                ..block
            },
            None => block,
        },
        _ => block,
    }
}

fn map_blocks(
    messages: Vec<AnthropicMessage>,
    rewrite: impl Fn(Vec<ContentBlock>) -> Vec<ContentBlock>,
) -> Vec<AnthropicMessage> {
    messages
        .into_iter()
        .map(|message| match message.content {
            MessageContent::Blocks(blocks) => AnthropicMessage {
                content: MessageContent::Blocks(rewrite(blocks)),
                ..message
            },
            MessageContent::Text(_) => message,
        })
        .collect()
}

pub fn sanitize_tool_use_ids(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    map_blocks(messages, |blocks| {
        blocks.into_iter().map(sanitize_tool_use_id_block).collect()
    })
}

pub fn strip_provider_specific_fields(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    map_blocks(messages, |blocks| {
        blocks
            .into_iter()
            .map(|block| ContentBlock {
                provider_specific_fields: None,
                ..block
            })
            .collect()
    })
}

pub fn is_encrypted_reasoning_block(block: &ContentBlock) -> bool {
    let field = match block.block_type.as_deref() {
        Some("thinking") => block.signature.as_deref(),
        Some("redacted_thinking") => block.data.as_deref(),
        _ => None,
    };
    field.is_some_and(|value| value.starts_with(ENCRYPTED_REASONING_SIGNATURE_PREFIX))
}

pub fn strip_encrypted_reasoning_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| !is_encrypted_reasoning_block(block))
}

fn is_advisor_use(block: &ContentBlock) -> bool {
    block.is_type("server_tool_use")
        && block.name.as_deref() == Some("advisor")
        && block.id.as_deref().is_some_and(|id| !id.is_empty())
}

pub fn strip_advisor_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    messages
        .into_iter()
        .map(|message| {
            if message.role != "assistant" {
                return message;
            }
            let MessageContent::Blocks(blocks) = &message.content else {
                return message;
            };
            let advisor_ids: Vec<&str> = blocks
                .iter()
                .filter(|block| is_advisor_use(block))
                .filter_map(|block| block.id.as_deref())
                .collect();
            if advisor_ids.is_empty() {
                return message;
            }
            let kept: Vec<ContentBlock> = blocks
                .iter()
                .filter(|block| {
                    let is_result = block.is_type("advisor_tool_result")
                        && block
                            .tool_use_id
                            .as_deref()
                            .is_some_and(|id| advisor_ids.contains(&id));
                    !is_advisor_use(block) && !is_result
                })
                .cloned()
                .collect();
            message.with_blocks(kept)
        })
        .collect()
}

#[derive(Deserialize)]
struct ReplayedWebSearchResult {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    snippet: String,
    #[serde(default)]
    encrypted_content: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ReplayedWebSearchContent {
    #[serde(rename = "web_search_tool_result_error")]
    Error {
        #[serde(default)]
        error_code: String,
    },
}

enum WebSearchResults {
    Results(Vec<ReplayedWebSearchResult>),
    Error(String),
}

fn flattenable_web_search_results(block: &ContentBlock) -> Option<(&str, WebSearchResults)> {
    if !block.is_type("web_search_tool_result") {
        return None;
    }
    let tool_use_id = block.tool_use_id.as_deref()?;
    let results = match block.content.as_ref()? {
        Value::Array(items) => {
            let results = items
                .iter()
                .map(|item| {
                    (item.get("type").and_then(Value::as_str) == Some("web_search_result"))
                        .then(|| {
                            serde_json::from_value::<ReplayedWebSearchResult>(item.clone()).ok()
                        })
                        .flatten()
                })
                .collect::<Option<Vec<_>>>()?;
            if results
                .iter()
                .any(|result| !result.encrypted_content.is_empty())
            {
                return None;
            }
            WebSearchResults::Results(results)
        }
        error @ Value::Object(_) => match serde_json::from_value(error.clone()).ok()? {
            ReplayedWebSearchContent::Error { error_code } => WebSearchResults::Error(error_code),
        },
        _ => return None,
    };
    Some((tool_use_id, results))
}

fn render_web_search_results(query: &str, results: &WebSearchResults) -> String {
    let header = if query.is_empty() {
        "Web search results:".to_string()
    } else {
        format!("Web search results for '{query}':")
    };
    match results {
        WebSearchResults::Error(code) => {
            let code = if code.is_empty() { "unavailable" } else { code };
            format!("{header}\n\nSearch failed: {code}")
        }
        WebSearchResults::Results(results) if results.is_empty() => {
            format!("{header}\n\nNo results were returned.")
        }
        WebSearchResults::Results(results) => {
            let body = results
                .iter()
                .map(|result| {
                    [
                        (!result.title.is_empty()).then(|| format!("Title: {}", result.title)),
                        (!result.url.is_empty()).then(|| format!("URL: {}", result.url)),
                        (!result.snippet.is_empty())
                            .then(|| format!("Snippet: {}", result.snippet)),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("\n")
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            if body.is_empty() {
                header
            } else {
                format!("{header}\n\n{body}")
            }
        }
    }
}

fn server_tool_use_query(block: &ContentBlock) -> Option<(&str, &str)> {
    if !block.is_type("server_tool_use") {
        return None;
    }
    let id = block.id.as_deref()?;
    let query = match block.input.as_ref() {
        None => "",
        Some(Value::Object(input)) => match input.get("query") {
            None => "",
            Some(query) => query.as_str()?,
        },
        Some(_) => return None,
    };
    Some((id, query))
}

fn flatten_web_search_results_in_blocks(blocks: Vec<ContentBlock>) -> Vec<ContentBlock> {
    let flattenable_ids: Vec<&str> = blocks
        .iter()
        .filter_map(flattenable_web_search_results)
        .map(|(tool_use_id, _)| tool_use_id)
        .collect();
    if flattenable_ids.is_empty() {
        return blocks;
    }
    let queries: Vec<(&str, &str)> = blocks.iter().filter_map(server_tool_use_query).collect();
    blocks
        .iter()
        .filter_map(|block| {
            if let Some((tool_use_id, results)) = flattenable_web_search_results(block) {
                let query = queries
                    .iter()
                    .rfind(|(id, _)| *id == tool_use_id)
                    .map_or("", |(_, query)| query);
                return Some(ContentBlock::text(render_web_search_results(
                    query, &results,
                )));
            }
            if let Some((id, _)) = server_tool_use_query(block)
                && flattenable_ids.contains(&id)
            {
                return None;
            }
            Some(block.clone())
        })
        .collect()
}

pub fn flatten_unencrypted_web_search_results(
    messages: Vec<AnthropicMessage>,
) -> Vec<AnthropicMessage> {
    map_blocks(messages, flatten_web_search_results_in_blocks)
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    const ALL_LEVELS: [EffortLevel; 5] = [
        EffortLevel::Low,
        EffortLevel::Medium,
        EffortLevel::High,
        EffortLevel::Xhigh,
        EffortLevel::Max,
    ];

    fn apply(
        sanitizer: fn(Vec<AnthropicMessage>) -> Vec<AnthropicMessage>,
        messages: Value,
    ) -> Value {
        let parsed: Vec<AnthropicMessage> = serde_json::from_value(messages).unwrap();
        serde_json::to_value(sanitizer(parsed)).unwrap()
    }

    fn block(value: Value) -> ContentBlock {
        serde_json::from_value(value).unwrap()
    }

    fn history(messages: Value) -> Vec<AnthropicMessage> {
        serde_json::from_value(messages).unwrap()
    }

    fn tools(value: Option<Value>) -> Option<Vec<Value>> {
        value.map(|tools| tools.as_array().unwrap().clone())
    }

    fn tagged(encrypted: &str) -> String {
        format!("{ENCRYPTED_REASONING_SIGNATURE_PREFIX}{encrypted}")
    }

    fn tiers(
        minimal: bool,
        low: bool,
        medium: bool,
        high: bool,
        xhigh: bool,
        max: bool,
    ) -> SupportedEffortTiers {
        SupportedEffortTiers {
            minimal,
            low,
            medium,
            high,
            xhigh,
            max,
        }
    }

    fn replayed_search_turn(results: Value) -> Value {
        json!([
            {"role": "user", "content": "when was Rome founded?"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "when"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": results},
                {"type": "text", "text": "753 BC."}
            ]}
        ])
    }

    #[fixture]
    fn unmapped() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities::default()
    }

    #[rstest]
    #[case::empty_text(json!({"type": "thinking", "thinking": ""}), true)]
    #[case::whitespace_only(json!({"type": "thinking", "thinking": " \n\t "}), true)]
    #[case::null_text(json!({"type": "thinking", "thinking": null}), true)]
    #[case::missing_text(json!({"type": "thinking"}), true)]
    #[case::empty_text_despite_signature(json!({"type": "thinking", "thinking": "", "signature": "sig_abc"}), true)]
    #[case::real_thinking(json!({"type": "thinking", "thinking": "plan", "signature": "sig"}), false)]
    #[case::padded_real_thinking(json!({"type": "thinking", "thinking": "  plan  "}), false)]
    #[case::redacted_thinking_is_a_different_type(json!({"type": "redacted_thinking", "data": "opaque"}), false)]
    #[case::empty_text_block(json!({"type": "text", "text": ""}), false)]
    #[case::untyped_block(json!({"thinking": ""}), false)]
    fn empty_thinking_block_detection(#[case] input: Value, #[case] expected: bool) {
        assert_eq!(is_empty_thinking_block(&block(input)), expected);
    }

    #[rstest]
    #[case::empty_text_beside_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]}])
    )]
    #[case::whitespace_text_beside_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "  \n "},
            {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]}])
    )]
    #[case::null_text(
        json!([{"role": "user", "content": [
            {"type": "text", "text": null},
            {"type": "tool_result", "tool_use_id": "x", "content": "y"}
        ]}]),
        json!([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}])
    )]
    #[case::missing_text(
        json!([{"role": "user", "content": [
            {"type": "text"},
            {"type": "tool_result", "tool_use_id": "x", "content": "y"}
        ]}]),
        json!([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}])
    )]
    #[case::empty_signed_thinking_beside_tool_use(
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "sig_abc"},
                {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}}
            ]}
        ]),
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}}
            ]}
        ])
    )]
    #[case::whitespace_thinking_beside_real_and_redacted_thinking(
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": " \n "},
            {"type": "thinking", "thinking": "real plan", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "real plan", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"}
        ]}])
    )]
    #[case::blank_text_beside_real_thinking(
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
            {"type": "text", "text": ""}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "plan", "signature": "sig"}]}])
    )]
    #[case::message_left_without_blocks_is_dropped(
        json!([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": ""}]}
        ]),
        json!([{"role": "user", "content": "hello"}])
    )]
    fn strip_empty_content_blocks_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_empty_content_blocks, input), expected);
    }

    #[rstest]
    #[case::non_empty_text(json!([{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]))]
    #[case::padded_text(json!([{"role": "assistant", "content": [{"type": "text", "text": "  hi  "}]}]))]
    #[case::empty_string_content(json!([{"role": "user", "content": ""}]))]
    #[case::textless_non_text_block(json!([{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
    ]}]))]
    #[case::encrypted_reasoning_left_for_the_responses_bridge(json!([{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")},
        {"type": "redacted_thinking", "data": tagged("gAAAA_2")},
        {"type": "text", "text": "The answer."}
    ]}]))]
    fn strip_empty_content_blocks_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(strip_empty_content_blocks, input.clone()), input);
    }

    #[rstest]
    #[case::replayed_provider_id("functions.Bash:0", "functions_Bash_0")]
    #[case::thought_signature_suffix("call_abc123__thought__CiIBDDnWx+/a==", "call_abc123")]
    #[case::splits_at_first_thought_separator("call_1__thought__a__thought__b", "call_1")]
    #[case::valid_id("toolu_01-A_b", "toolu_01-A_b")]
    #[case::non_ascii_letter("café", "caf_")]
    #[case::only_invalid_characters("::", "__")]
    #[case::empty("", "tool_use_id")]
    #[case::thought_signature_only("__thought__CiIB", "tool_use_id")]
    fn normalize_anthropic_tool_use_id_cases(#[case] raw: &str, #[case] expected: &str) {
        assert_eq!(normalize_anthropic_tool_use_id(raw), expected);
    }

    #[rstest]
    #[case::tool_use_and_its_result(
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]}
        ]),
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"}]}
        ])
    )]
    #[case::server_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srv.1", "name": "web_search", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}}
        ]}])
    )]
    #[case::tool_use_rewrites_only_its_id(
        json!([{"role": "assistant", "content": [
            {"type": "tool_use", "id": "a.b", "tool_use_id": "c.d", "name": "Bash", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "tool_use", "id": "a_b", "tool_use_id": "c.d", "name": "Bash", "input": {}}
        ]}])
    )]
    #[case::tool_result_rewrites_only_its_tool_use_id(
        json!([{"role": "user", "content": [
            {"type": "tool_result", "id": "a.b", "tool_use_id": "c.d", "content": "ok"}
        ]}]),
        json!([{"role": "user", "content": [
            {"type": "tool_result", "id": "a.b", "tool_use_id": "c_d", "content": "ok"}
        ]}])
    )]
    fn sanitize_tool_use_ids_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(sanitize_tool_use_ids, input), expected);
    }

    #[rstest]
    #[case::valid_ids(json!([
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}]}
    ]))]
    #[case::id_mentioned_in_text(json!([{"role": "user", "content": [{"type": "text", "text": "id: functions.Bash:0"}]}]))]
    #[case::tool_use_without_id(json!([{"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "input": {}}]}]))]
    #[case::tool_result_without_tool_use_id(json!([{"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}]))]
    #[case::string_content(json!([{"role": "user", "content": "functions.Bash:0"}]))]
    fn sanitize_tool_use_ids_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(sanitize_tool_use_ids, input.clone()), input);
    }

    #[rstest]
    #[case::thinking_block(
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hm", "signature": "s", "provider_specific_fields": {"a": 1}}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "s"}]}])
    )]
    #[case::every_block_of_every_message(
        json!([
            {"role": "assistant", "content": [
                {"type": "text", "text": "a", "provider_specific_fields": {"x": 1}},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}, "provider_specific_fields": {"y": 2}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "provider_specific_fields": {}}
            ]}
        ]),
        json!([
            {"role": "assistant", "content": [
                {"type": "text", "text": "a"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
        ])
    )]
    fn strip_provider_specific_fields_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_provider_specific_fields, input), expected);
    }

    #[rstest]
    #[case::string_content(json!([{"role": "user", "content": "provider_specific_fields"}]))]
    #[case::blocks_without_the_field(json!([{"role": "assistant", "content": [{"type": "text", "text": "a"}]}]))]
    fn strip_provider_specific_fields_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(strip_provider_specific_fields, input.clone()), input);
    }

    #[rstest]
    #[case::tagged_thinking_signature(json!({"type": "thinking", "thinking": "x", "signature": tagged("g")}), true)]
    #[case::tagged_redacted_data(json!({"type": "redacted_thinking", "data": tagged("g")}), true)]
    #[case::bare_tag_signature(json!({"type": "thinking", "thinking": "x", "signature": tagged("")}), true)]
    #[case::bare_tag_data(json!({"type": "redacted_thinking", "data": tagged("")}), true)]
    #[case::anthropic_signature(json!({"type": "thinking", "thinking": "x", "signature": "ErcBCkgIValid"}), false)]
    #[case::anthropic_data(json!({"type": "redacted_thinking", "data": "EmwKAhgBEgy"}), false)]
    #[case::unsigned_thinking(json!({"type": "thinking", "thinking": "x"}), false)]
    #[case::tag_in_text_block(json!({"type": "text", "text": tagged("g")}), false)]
    #[case::tag_in_thinking_data(json!({"type": "thinking", "thinking": "x", "data": tagged("g")}), false)]
    #[case::tag_in_redacted_signature(
        json!({"type": "redacted_thinking", "data": "EmwKAhgBEgy", "signature": tagged("g")}),
        false
    )]
    #[case::tag_not_at_start(json!({"type": "thinking", "thinking": "x", "signature": format!("x{}", tagged("g"))}), false)]
    fn encrypted_reasoning_block_detection(#[case] input: Value, #[case] expected: bool) {
        assert_eq!(is_encrypted_reasoning_block(&block(input)), expected);
    }

    #[rstest]
    #[case::only_the_bridge_tagged_blocks(
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")},
                {"type": "redacted_thinking", "data": tagged("gAAAA_2")}
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_3")},
                {"type": "thinking", "thinking": "native", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
                {"type": "text", "text": "The answer."}
            ]}
        ]),
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "native", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
                {"type": "text", "text": "The answer."}
            ]}
        ])
    )]
    #[case::bridge_turn_keeps_its_text(
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")},
                {"type": "redacted_thinking", "data": tagged("gAAAA_2")},
                {"type": "text", "text": "The answer."}
            ]},
            {"role": "user", "content": "And the next one?"}
        ]),
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [{"type": "text", "text": "The answer."}]},
            {"role": "user", "content": "And the next one?"}
        ])
    )]
    fn strip_encrypted_reasoning_blocks_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_encrypted_reasoning_blocks, input), expected);
    }

    #[rstest]
    #[case::anthropic_signed_blocks(json!([
        {"role": "user", "content": "Solve it."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
            {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
            {"type": "text", "text": "The answer."}
        ]}
    ]))]
    #[case::string_content(json!([{"role": "user", "content": tagged("g")}]))]
    fn strip_encrypted_reasoning_blocks_leaves_untouched(#[case] input: Value) {
        assert_eq!(
            apply(strip_encrypted_reasoning_blocks, input.clone()),
            input
        );
    }

    #[rstest]
    #[case::advisor_exchange_between_texts(
        json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "server_tool_use", "id": "srvtoolu_abc123", "name": "advisor", "input": {}},
                {"type": "advisor_tool_result", "tool_use_id": "srvtoolu_abc123",
                 "content": {"type": "advisor_result", "text": "Use channels."}},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ]),
        json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ])
    )]
    #[case::only_results_of_this_turns_advisor_calls(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "adv_1", "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "tool_use_id": "adv_1", "content": "advice"},
            {"type": "advisor_tool_result", "tool_use_id": "other", "content": "kept"},
            {"type": "tool_result", "tool_use_id": "adv_1", "content": "kept"},
            {"type": "text", "text": "answer"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "advisor_tool_result", "tool_use_id": "other", "content": "kept"},
            {"type": "tool_result", "tool_use_id": "adv_1", "content": "kept"},
            {"type": "text", "text": "answer"}
        ]}])
    )]
    #[case::advisor_call_without_result(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "adv_1", "name": "advisor", "input": {}},
            {"type": "text", "text": "answer"}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}])
    )]
    #[case::advisor_only_turn_keeps_an_empty_block_list(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "adv_1", "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "tool_use_id": "adv_1", "content": "advice"}
        ]}]),
        json!([{"role": "assistant", "content": []}])
    )]
    fn strip_advisor_blocks_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_advisor_blocks, input), expected);
    }

    #[rstest]
    #[case::no_advisor_blocks(json!([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Hi there"},
            {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"location": "SF"}}
        ]}
    ]))]
    #[case::user_turn(json!([{"role": "user", "content": [
        {"type": "server_tool_use", "id": "adv_2", "name": "advisor", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "adv_2", "content": "advice"}
    ]}]))]
    #[case::other_server_tool(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "s1", "content": "advice"}
    ]}]))]
    #[case::client_tool_named_advisor(json!([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "advisor", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "t1", "content": "advice"}
    ]}]))]
    #[case::advisor_call_with_empty_id(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "", "name": "advisor", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "", "content": "advice"}
    ]}]))]
    #[case::advisor_call_without_id(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "name": "advisor", "input": {}}
    ]}]))]
    #[case::string_content(json!([{"role": "assistant", "content": "advisor"}]))]
    fn strip_advisor_blocks_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(strip_advisor_blocks, input.clone()), input);
    }

    #[rstest]
    #[case::results_keep_their_evidence(
        json!([
            {"role": "user", "content": "latest version?"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "latest version"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                    {"type": "web_search_result", "url": "https://example.com/releases", "title": "Releases",
                     "page_age": null, "encrypted_content": "", "snippet": "Latest release v1.95.0"}
                ]},
                {"type": "text", "text": "v1.95.0"}
            ]}
        ]),
        json!([
            {"role": "user", "content": "latest version?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Web search results for 'latest version':\n\nTitle: Releases\nURL: https://example.com/releases\nSnippet: Latest release v1.95.0"},
                {"type": "text", "text": "v1.95.0"}
            ]}
        ])
    )]
    #[case::each_result_lists_only_its_present_fields(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "title": "A"},
                {"type": "web_search_result", "snippet": "b"},
                {"type": "web_search_result", "url": "https://c"}
            ]}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nTitle: A\n\nSnippet: b\n\nURL: https://c"}
        ]}])
    )]
    #[case::result_without_fields_renders_the_header_only(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [{"type": "web_search_result"}]}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "text", "text": "Web search results for 'q':"}]}])
    )]
    #[case::resultless_search(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "who won"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            {"type": "text", "text": "I could not find that."}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'who won':\n\nNo results were returned."},
            {"type": "text", "text": "I could not find that."}
        ]}])
    )]
    #[case::failed_search(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
             "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nSearch failed: max_uses_exceeded"}
        ]}])
    )]
    #[case::failed_search_without_error_code(
        json!([{"role": "assistant", "content": [
            {"type": "web_search_tool_result", "tool_use_id": "e1", "content": {"type": "web_search_tool_result_error"}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nSearch failed: unavailable"}
        ]}])
    )]
    #[case::server_tool_use_without_query(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search"},
            {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "web_search_tool_result", "tool_use_id": "s2", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nNo results were returned."},
            {"type": "text", "text": "Web search results:\n\nNo results were returned."}
        ]}])
    )]
    #[case::genuine_results_in_the_same_turn_stay(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "rust"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://r", "title": "Rust", "snippet": "fast"}
            ]},
            {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "real"}},
            {"type": "web_search_tool_result", "tool_use_id": "s2", "content": [
                {"type": "web_search_result", "url": "https://a", "title": "A", "snippet": "b", "encrypted_content": "enc"}
            ]},
            {"type": "text", "text": "done"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'rust':\n\nTitle: Rust\nURL: https://r\nSnippet: fast"},
            {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "real"}},
            {"type": "web_search_tool_result", "tool_use_id": "s2", "content": [
                {"type": "web_search_result", "url": "https://a", "title": "A", "snippet": "b", "encrypted_content": "enc"}
            ]},
            {"type": "text", "text": "done"}
        ]}])
    )]
    #[case::other_blocks_sharing_the_tool_use_id_stay(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "tool_result", "tool_use_id": "s1", "content": "x"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nNo results were returned."},
            {"type": "tool_result", "tool_use_id": "s1", "content": "x"}
        ]}])
    )]
    #[case::query_lookup_stays_within_the_message(
        json!([
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
            ]}
        ]),
        json!([
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Web search results:\n\nNo results were returned."}
            ]}
        ])
    )]
    #[case::result_without_any_field_keeps_its_slot(
        json!([{"role": "assistant", "content": [
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://a", "title": "A"},
                {"type": "web_search_result"},
                {"type": "web_search_result", "title": "B"}
            ]}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nTitle: A\nURL: https://a\n\n\n\nTitle: B"}
        ]}])
    )]
    #[case::non_string_query_keeps_its_server_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": 123}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": 123}},
            {"type": "text", "text": "Web search results:\n\nNo results were returned."}
        ]}])
    )]
    #[case::non_object_input_keeps_its_server_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": "q"},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": "q"},
            {"type": "text", "text": "Web search results:\n\nNo results were returned."}
        ]}])
    )]
    #[case::repeated_tool_use_id_renders_each_block_from_its_own_results(
        json!([{"role": "assistant", "content": [
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": {"type": "web_search_tool_result_error", "error_code": "max_uses"}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nNo results were returned."},
            {"type": "text", "text": "Web search results:\n\nSearch failed: max_uses"}
        ]}])
    )]
    #[case::encrypted_block_sharing_a_replayed_id_stays(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://a", "encrypted_content": "enc"}
            ]}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nNo results were returned."},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://a", "encrypted_content": "enc"}
            ]}
        ]}])
    )]
    #[case::last_query_wins_for_a_repeated_server_tool_use_id(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "first"}},
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "second"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'second':\n\nNo results were returned."}
        ]}])
    )]
    fn flatten_unencrypted_web_search_results_rewrites(
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            apply(flatten_unencrypted_web_search_results, input),
            expected
        );
    }

    #[rstest]
    #[case::anthropic_issued_results(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
            {"type": "web_search_result", "url": "https://example.com", "title": "Example",
             "page_age": null, "encrypted_content": "EqgfCioIARgBIiQ4"}
        ]}
    ]}]))]
    #[case::any_encrypted_result_marks_the_block_genuine(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "url": "https://a", "encrypted_content": ""},
            {"type": "web_search_result", "url": "https://b", "encrypted_content": "enc"}
        ]}
    ]}]))]
    #[case::result_without_tool_use_id(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "content": []}
    ]}]))]
    #[case::foreign_item_in_results(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "url": "https://a"},
            {"type": "text", "text": "x"}
        ]}
    ]}]))]
    #[case::result_with_null_url(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "url": null, "title": "A"}
        ]}
    ]}]))]
    #[case::string_result_content(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": "oops"}
    ]}]))]
    #[case::object_content_that_is_not_an_error(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": {"type": "web_search_result", "url": "https://a"}}
    ]}]))]
    #[case::string_content(json!([{"role": "assistant", "content": "web_search_tool_result"}]))]
    fn flatten_unencrypted_web_search_results_leaves_untouched(#[case] input: Value) {
        assert_eq!(
            apply(flatten_unencrypted_web_search_results, input.clone()),
            input
        );
    }

    #[rstest]
    #[case::anthropic_invalid_signature(
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages.3.content.3: Invalid `signature` in `thinking` block"},"request_id":"req_1"}"#
    )]
    #[case::bedrock_signature_not_a_string(
        r#"{"message":"messages.2.content.0.thinking.signature.str: Input should be a valid string"}"#
    )]
    #[case::vertex_signature_not_a_string(
        "messages.4.content.1.thinking.signature.str: Input should be a valid string"
    )]
    #[case::empty_thinking_text(
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0.thinking: each thinking block must contain thinking"}}"#
    )]
    #[case::shouting("MESSAGES.0.CONTENT.0: INVALID `SIGNATURE` IN `THINKING` BLOCK")]
    fn invalid_thinking_block_errors_are_recognized(#[case] error_text: &str) {
        assert!(is_anthropic_invalid_thinking_block_error(error_text));
    }

    #[rstest]
    #[case::empty("")]
    #[case::rate_limit("rate limit exceeded")]
    #[case::unrelated_invalid_request("invalid_request_error: model not found")]
    #[case::signature_without_invalid("thinking signature is malformed")]
    #[case::invalid_signature_without_thinking(
        "messages.0.content.0: Invalid `signature` in `tool_use` block"
    )]
    #[case::must_contain_without_thinking("each text block must contain text")]
    fn other_errors_are_not_invalid_thinking_block_errors(#[case] error_text: &str) {
        assert!(!is_anthropic_invalid_thinking_block_error(error_text));
    }

    #[rstest]
    #[case::thinking_beside_text(
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {"type": "text", "text": "hello"}
            ]}
        ]),
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
        ])
    )]
    #[case::redacted_thinking_beside_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}])
    )]
    #[case::message_left_without_blocks_is_dropped(
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "plan", "signature": "sig"}]}
        ]),
        json!([{"role": "user", "content": "hi"}])
    )]
    #[case::history_without_thinking_is_kept(
        json!([{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]),
        json!([{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}])
    )]
    fn strip_thinking_blocks_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_thinking_blocks, input), expected);
    }

    #[rstest]
    #[case::with_results(json!([{"type": "web_search_result", "url": "u", "title": "Rome", "snippet": "s", "page_age": null}]))]
    #[case::without_results(json!([]))]
    #[case::failed_search(json!({"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}))]
    fn flatten_unencrypted_web_search_results_is_idempotent(#[case] results: Value) {
        let input = replayed_search_turn(results);
        let once = apply(flatten_unencrypted_web_search_results, input.clone());
        let twice = apply(flatten_unencrypted_web_search_results, once.clone());
        assert_ne!(once, input);
        assert_eq!(twice, once);
    }

    #[rstest]
    #[case::no_existing_header(None, "b", "b")]
    #[case::empty_existing_header(Some(""), "b", "b")]
    #[case::whitespace_existing_header(Some("  "), "b", "b")]
    #[case::sorted_after_merge(Some("c,a"), "b", "a,b,c")]
    #[case::already_present(Some("a,b"), "a", "a,b")]
    #[case::trimmed_and_deduplicated(Some("b, a ,b"), "c", "a,b,c")]
    #[case::blank_pieces_skipped(Some("a,,b"), "c", "a,b,c")]
    fn beta_values_merge_sorted_and_deduplicated(
        #[case] existing: Option<&str>,
        #[case] new_beta: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            join_beta_values(split_beta_values(existing).chain([new_beta.to_string()])),
            expected
        );
    }

    #[rstest]
    #[case::raw_token("sk-ant-oat01-abc123", true)]
    #[case::bearer_token("Bearer sk-ant-oat02-xyz789", true)]
    #[case::bare_prefix(ANTHROPIC_OAUTH_TOKEN_PREFIX, true)]
    #[case::api_key("sk-ant-api01-abc123", false)]
    #[case::bearer_api_key("Bearer sk-ant-api01-abc123", false)]
    #[case::empty("", false)]
    #[case::uppercase_prefix("sk-ant-OAT01-abc123", false)]
    #[case::shouting_prefix("SK-ANT-OAT01-abc123", false)]
    #[case::lowercase_bearer("bearer sk-ant-oat01-abc123", false)]
    #[case::bearer_stripped_once("Bearer Bearer sk-ant-oat01-abc123", false)]
    #[case::prefix_not_at_start(" sk-ant-oat01-abc123", false)]
    fn anthropic_oauth_key_detection(#[case] value: &str, #[case] expected: bool) {
        assert_eq!(is_anthropic_oauth_key(value), expected);
    }

    #[rstest]
    #[case::regex_tool(Some(json!([{"type": ANTHROPIC_TOOL_SEARCH_TOOL_TYPES[0], "name": "tool_search_tool_regex"}])), true)]
    #[case::bm25_tool(Some(json!([{"type": ANTHROPIC_TOOL_SEARCH_TOOL_TYPES[1], "name": "tool_search_tool_bm25"}])), true)]
    #[case::after_other_tools(
        Some(json!([{"name": "get_weather", "input_schema": {}}, {"type": ANTHROPIC_TOOL_SEARCH_TOOL_TYPES[1]}])),
        true
    )]
    #[case::function_tool(Some(json!([{"type": "function", "function": {"name": "get_weather"}}])), false)]
    #[case::name_without_type(Some(json!([{"name": ANTHROPIC_TOOL_SEARCH_TOOL_TYPES[0]}])), false)]
    #[case::empty_tools(Some(json!([])), false)]
    #[case::no_tools(None, false)]
    fn tool_search_detection(#[case] input: Option<Value>, #[case] expected: bool) {
        assert_eq!(is_tool_search_used(tools(input).as_deref()), expected);
    }

    #[rstest]
    #[case::advisor_tool(Some(json!([{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "name": "advisor"}])), true)]
    #[case::after_other_tools(Some(json!([{"name": "f", "input_schema": {}}, {"type": ANTHROPIC_ADVISOR_TOOL_TYPE}])), true)]
    #[case::tool_named_advisor(Some(json!([{"name": "advisor", "input_schema": {}}])), false)]
    #[case::other_server_tool(Some(json!([{"type": "web_search_20250305", "name": "web_search"}])), false)]
    #[case::empty_tools(Some(json!([])), false)]
    #[case::no_tools(None, false)]
    fn advisor_tool_detection(#[case] input: Option<Value>, #[case] expected: bool) {
        assert_eq!(has_advisor_tool(tools(input).as_deref()), expected);
    }

    #[rstest]
    #[case::param_without_history(Some(json!({})), json!([]), true)]
    #[case::param_with_unsigned_history(
        Some(json!({"trigger": 1})),
        json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c"}]}]),
        true
    )]
    #[case::signed_block(None, json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c", "signature": "s"}]}]), true)]
    #[case::signed_block_later_in_history(
        None,
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "compaction", "content": "c", "signature": "s"}]}
        ]),
        true
    )]
    #[case::unsigned_block(None, json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c"}]}]), false)]
    #[case::empty_signature(None, json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c", "signature": ""}]}]), false)]
    #[case::signed_non_compaction_block(
        None,
        json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}]}]),
        false
    )]
    #[case::string_content(None, json!([{"role": "user", "content": "compaction"}]), false)]
    #[case::neither(None, json!([]), false)]
    fn native_compaction_beta_requirement(
        #[case] compaction: Option<Value>,
        #[case] messages: Value,
        #[case] expected: bool,
    ) {
        assert_eq!(
            requires_native_compaction_beta(compaction.as_ref(), &history(messages)),
            expected
        );
    }

    #[rstest]
    #[case::low(EffortLevel::Low, "low")]
    #[case::medium(EffortLevel::Medium, "medium")]
    #[case::high(EffortLevel::High, "high")]
    #[case::xhigh(EffortLevel::Xhigh, "xhigh")]
    #[case::max(EffortLevel::Max, "max")]
    fn effort_level_names_agree_across_str_parse_and_serde(
        #[case] level: EffortLevel,
        #[case] name: &str,
    ) {
        assert_eq!(level.as_str(), name);
        assert_eq!(EffortLevel::parse(name), Some(level));
        assert_eq!(serde_json::to_value(level).unwrap(), json!(name));
        assert_eq!(
            serde_json::from_value::<EffortLevel>(json!(name)).unwrap(),
            level
        );
    }

    #[rstest]
    #[case::unknown("ultra")]
    #[case::minimal_is_not_an_output_config_level("minimal")]
    #[case::uppercase("HIGH")]
    #[case::empty("")]
    fn effort_level_parse_rejects(#[case] value: &str) {
        assert_eq!(EffortLevel::parse(value), None);
    }

    #[rstest]
    #[case::minimal_only(tiers(true, false, false, false, false, false), [false, false, false, false, false])]
    #[case::low_only(tiers(false, true, false, false, false, false), [true, false, false, false, false])]
    #[case::medium_only(tiers(false, false, true, false, false, false), [false, true, false, false, false])]
    #[case::high_only(tiers(false, false, false, true, false, false), [false, false, true, false, false])]
    #[case::xhigh_only(tiers(false, false, false, false, true, false), [false, false, false, true, false])]
    #[case::max_only(tiers(false, false, false, false, false, true), [false, false, false, false, true])]
    fn supports_effort_tier_reads_the_matching_flag(
        #[case] effort_tiers: SupportedEffortTiers,
        #[case] expected: [bool; 5],
        unmapped: AnthropicModelCapabilities,
    ) {
        let capabilities = AnthropicModelCapabilities {
            effort_tiers,
            ..unmapped
        };
        assert_eq!(
            ALL_LEVELS.map(|level| capabilities.supports_effort_tier(level)),
            expected
        );
    }

    #[rstest]
    #[case::unmapped(false, false, false, SupportedEffortTiers::default(), false)]
    #[case::reasoning_and_adaptive_thinking_alone(
        true,
        true,
        false,
        SupportedEffortTiers::default(),
        false
    )]
    #[case::output_config_without_tiers(false, false, true, SupportedEffortTiers::default(), true)]
    #[case::minimal_tier(
        false,
        false,
        false,
        tiers(true, false, false, false, false, false),
        true
    )]
    #[case::low_tier(
        false,
        false,
        false,
        tiers(false, true, false, false, false, false),
        true
    )]
    #[case::medium_tier(
        false,
        false,
        false,
        tiers(false, false, true, false, false, false),
        true
    )]
    #[case::high_tier(
        false,
        false,
        false,
        tiers(false, false, false, true, false, false),
        true
    )]
    #[case::xhigh_tier(
        false,
        false,
        false,
        tiers(false, false, false, false, true, false),
        true
    )]
    #[case::max_tier(
        false,
        false,
        false,
        tiers(false, false, false, false, false, true),
        true
    )]
    fn supports_effort_param_cases(
        #[case] supports_reasoning: bool,
        #[case] supports_adaptive_thinking: bool,
        #[case] supports_output_config: bool,
        #[case] effort_tiers: SupportedEffortTiers,
        #[case] expected: bool,
        unmapped: AnthropicModelCapabilities,
    ) {
        let capabilities = AnthropicModelCapabilities {
            supports_reasoning,
            supports_adaptive_thinking,
            supports_output_config,
            effort_tiers,
            ..unmapped
        };
        assert_eq!(capabilities.supports_effort_param(), expected);
    }

    #[rstest]
    #[case::max_on_adaptive_thinking_model(true, SupportedEffortTiers::default(), "max", None)]
    #[case::max_on_max_tier_model(
        false,
        tiers(false, false, false, false, false, true),
        "max",
        None
    )]
    #[case::max_on_output_config_only_model(
        false,
        SupportedEffortTiers::default(),
        "max",
        Some("effort='max' is not supported by this model. Got model: claude-test")
    )]
    #[case::max_on_xhigh_tier_model(
        false,
        tiers(false, false, false, false, true, false),
        "max",
        Some("effort='max' is not supported by this model. Got model: claude-test")
    )]
    #[case::xhigh_on_xhigh_tier_model(
        false,
        tiers(false, false, false, false, true, false),
        "xhigh",
        None
    )]
    #[case::xhigh_on_adaptive_thinking_model(
        true,
        SupportedEffortTiers::default(),
        "xhigh",
        Some("effort='xhigh' is not supported by this model. Got model: claude-test")
    )]
    #[case::xhigh_on_max_tier_model(
        false,
        tiers(false, false, false, false, false, true),
        "xhigh",
        Some("effort='xhigh' is not supported by this model. Got model: claude-test")
    )]
    #[case::high_on_unmapped_model(false, SupportedEffortTiers::default(), "high", None)]
    #[case::low_on_unmapped_model(false, SupportedEffortTiers::default(), "low", None)]
    #[case::unknown_level_is_left_to_other_validation(
        false,
        SupportedEffortTiers::default(),
        "ultra",
        None
    )]
    fn effort_level_rejection_cases(
        #[case] supports_adaptive_thinking: bool,
        #[case] effort_tiers: SupportedEffortTiers,
        #[case] effort: &str,
        #[case] expected: Option<&str>,
        unmapped: AnthropicModelCapabilities,
    ) {
        let capabilities = AnthropicModelCapabilities {
            supports_output_config: true,
            supports_adaptive_thinking,
            effort_tiers,
            ..unmapped
        };
        assert_eq!(
            capabilities
                .effort_level_rejection(effort, "claude-test")
                .as_deref(),
            expected
        );
    }

    #[rstest]
    fn unmapped_model_has_no_reasoning_features_but_accepts_sampling_params(
        unmapped: AnthropicModelCapabilities,
    ) {
        assert_eq!(
            unmapped,
            AnthropicModelCapabilities {
                supports_reasoning: false,
                supports_adaptive_thinking: false,
                thinking_always_on: false,
                supports_legacy_thinking: false,
                supports_output_config: false,
                supports_sampling_params: true,
                supports_speed: false,
                effort_tiers: tiers(false, false, false, false, false, false),
            }
        );
        assert_eq!(
            serde_json::from_value::<AnthropicModelCapabilities>(json!({})).unwrap(),
            unmapped
        );
    }

    #[rstest]
    #[case::sampling_params_removed(
        json!({"supports_sampling_params": false}),
        AnthropicModelCapabilities { supports_sampling_params: false, ..AnthropicModelCapabilities::default() }
    )]
    #[case::fast_mode(
        json!({"supports_speed": true}),
        AnthropicModelCapabilities { supports_speed: true, ..AnthropicModelCapabilities::default() }
    )]
    #[case::partial_effort_tiers(
        json!({"supports_reasoning": true, "effort_tiers": {"xhigh": true}}),
        AnthropicModelCapabilities {
            supports_reasoning: true,
            effort_tiers: tiers(false, false, false, false, true, false),
            ..AnthropicModelCapabilities::default()
        }
    )]
    fn capabilities_fill_missing_flags_with_unmapped_defaults(
        #[case] input: Value,
        #[case] expected: AnthropicModelCapabilities,
    ) {
        assert_eq!(
            serde_json::from_value::<AnthropicModelCapabilities>(input).unwrap(),
            expected
        );
    }
}
