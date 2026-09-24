//! Anthropic knowledge shared by every route that speaks the Messages API: model
//! capability flags, beta header values, and the message sanitizers Python runs in
//! `litellm/llms/anthropic/common_utils.py`.

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

/// Known `anthropic-beta` values, as Python's `ANTHROPIC_BETA_HEADER_VALUES` lists them.
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

/// The `output_config.effort` levels, in ascending order.
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

/// Which `reasoning_effort` tiers the cost map flags as `supports_<tier>_reasoning_effort`.
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

/// The cost-map facts about a model that the Messages transformation branches on. The
/// host resolves them the way Python's `AnthropicModelInfo._supports_model_capability`
/// does, under the caller's provider and with the fallback generalizations applied.
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
    /// Whether `temperature`, `top_p` and `top_k` are accepted. Claude 4.7+ removed them.
    #[serde(default = "default_true")]
    pub supports_sampling_params: bool,
    /// Whether the model takes `speed` (fast mode), a direct Anthropic API feature.
    #[serde(default)]
    pub supports_speed: bool,
    #[serde(default)]
    pub effort_tiers: SupportedEffortTiers,
}

fn default_true() -> bool {
    true
}

impl Default for AnthropicModelCapabilities {
    /// An unmapped model: no reasoning features, sampling params still accepted.
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

    /// Whether the model accepts `output_config.effort` at all, as Python's
    /// `_model_supports_effort_param` decides it.
    pub fn supports_effort_param(&self) -> bool {
        self.supports_output_config || self.effort_tiers.any()
    }

    /// Python's `_validate_effort_for_model`: `None` when the level is allowed, else the
    /// 400 message.
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

/// Merge one more value into a comma separated `anthropic-beta` header, sorted and
/// deduplicated like Python's `_merge_beta_headers`.
pub fn merge_beta_headers(existing: Option<&str>, new_beta: &str) -> String {
    join_beta_values(split_beta_values(existing).chain(std::iter::once(new_beta.to_string())))
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

/// Whether the request already carries native compaction: a `compaction` param or a
/// signed `compaction` block in history.
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

/// A `thinking` block with no thinking text. Anthropic rejects it regardless of any
/// signature; `redacted_thinking` blocks are a different type and never match.
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

/// Drop empty `text` and `thinking` blocks. A message whose block list empties out is
/// dropped with them, since Anthropic rejects an empty content array.
pub fn strip_empty_content_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| {
        !is_empty_text_block(block) && !is_empty_thinking_block(block)
    })
}

/// Rewrite a `tool_use` / `tool_result` id into Anthropic's `^[a-zA-Z0-9_-]+$` alphabet.
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

/// Rewrite tool ids replayed from another provider (`functions.Bash:0`) into ids Anthropic
/// accepts.
pub fn sanitize_tool_use_ids(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    map_blocks(messages, |blocks| {
        blocks.into_iter().map(sanitize_tool_use_id_block).collect()
    })
}

/// Drop `provider_specific_fields` from every block; it is LiteLLM's own annotation.
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

/// A thinking or redacted_thinking block carrying Responses API encrypted reasoning that
/// only the bridge which minted it can read back.
pub fn is_encrypted_reasoning_block(block: &ContentBlock) -> bool {
    let field = match block.block_type.as_deref() {
        Some("thinking") => block.signature.as_deref(),
        Some("redacted_thinking") => block.data.as_deref(),
        _ => None,
    };
    field.is_some_and(|value| value.starts_with(ENCRYPTED_REASONING_SIGNATURE_PREFIX))
}

/// Drop the bridge-tagged reasoning blocks Anthropic cannot verify; its own signed
/// blocks stay.
pub fn strip_encrypted_reasoning_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| !is_encrypted_reasoning_block(block))
}

fn is_advisor_use(block: &ContentBlock) -> bool {
    block.is_type("server_tool_use")
        && block.name.as_deref() == Some("advisor")
        && block.id.as_deref().is_some_and(|id| !id.is_empty())
}

/// Drop `server_tool_use(name=advisor)` and `advisor_tool_result` blocks from assistant
/// turns. Anthropic rejects advisor history when the advisor tool is not in `tools`.
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

/// The results of a replayed `web_search_tool_result` block that carries no
/// `encrypted_content`, else `None` for anything Anthropic itself issued.
fn flattenable_web_search_results(block: &ContentBlock) -> Option<WebSearchResults> {
    if !block.is_type("web_search_tool_result") || block.tool_use_id.is_none() {
        return None;
    }
    match block.content.as_ref()? {
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
            Some(WebSearchResults::Results(results))
        }
        error @ Value::Object(_) => match serde_json::from_value(error.clone()).ok()? {
            ReplayedWebSearchContent::Error { error_code } => {
                Some(WebSearchResults::Error(error_code))
            }
        },
        _ => None,
    }
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
                .filter(|entry| !entry.is_empty())
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
    let query = block
        .input
        .as_ref()
        .and_then(|input| input.get("query"))
        .and_then(Value::as_str)
        .unwrap_or("");
    Some((id, query))
}

fn flatten_web_search_results_in_blocks(blocks: Vec<ContentBlock>) -> Vec<ContentBlock> {
    let flattenable: Vec<(&str, WebSearchResults)> = blocks
        .iter()
        .filter_map(|block| {
            let results = flattenable_web_search_results(block)?;
            Some((block.tool_use_id.as_deref()?, results))
        })
        .collect();
    if flattenable.is_empty() {
        return blocks;
    }
    let queries: Vec<(&str, &str)> = blocks.iter().filter_map(server_tool_use_query).collect();
    let rewritten: Vec<ContentBlock> = blocks
        .iter()
        .filter_map(|block| {
            if let Some((tool_use_id, results)) = block
                .tool_use_id
                .as_deref()
                .and_then(|id| flattenable.iter().find(|(flat_id, _)| *flat_id == id))
                .filter(|_| block.is_type("web_search_tool_result"))
            {
                let query = queries
                    .iter()
                    .find(|(id, _)| id == tool_use_id)
                    .map_or("", |(_, query)| query);
                return Some(ContentBlock::text(render_web_search_results(
                    query, results,
                )));
            }
            if let Some((id, _)) = server_tool_use_query(block)
                && flattenable.iter().any(|(flat_id, _)| *flat_id == id)
            {
                return None;
            }
            Some(block.clone())
        })
        .collect();
    rewritten
}

/// Rewrite replayed `web_search_tool_result` blocks that carry no `encrypted_content`
/// (LiteLLM synthesized them) into plain text so Anthropic does not reject the history.
pub fn flatten_unencrypted_web_search_results(
    messages: Vec<AnthropicMessage>,
) -> Vec<AnthropicMessage> {
    map_blocks(messages, flatten_web_search_results_in_blocks)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn message(role: &str, content: Value) -> AnthropicMessage {
        serde_json::from_value(json!({"role": role, "content": content})).unwrap()
    }

    fn wire(messages: &[AnthropicMessage]) -> Value {
        serde_json::to_value(messages).unwrap()
    }

    #[test]
    fn strip_empty_content_blocks_drops_blank_text_and_thinking_and_empty_messages() {
        let messages = vec![
            message(
                "assistant",
                json!([
                    {"type": "text", "text": "  "},
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                ]),
            ),
            message("user", json!([{"type": "text", "text": ""}])),
            message("user", json!("plain")),
        ];
        assert_eq!(
            wire(&strip_empty_content_blocks(messages)),
            json!([
                {"role": "assistant", "content": [
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                ]},
                {"role": "user", "content": "plain"}
            ])
        );
    }

    #[rstest]
    #[case("functions.Bash:0", "functions_Bash_0")]
    #[case("call_1__thought__abc", "call_1")]
    #[case("toolu_01", "toolu_01")]
    #[case("::", "__")]
    #[case("", "tool_use_id")]
    fn normalize_tool_use_id_matches_python(#[case] raw: &str, #[case] expected: &str) {
        assert_eq!(normalize_anthropic_tool_use_id(raw), expected);
    }

    #[test]
    fn sanitize_tool_use_ids_rewrites_use_and_result_ids_only() {
        let messages = vec![
            message(
                "assistant",
                json!([{"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}]),
            ),
            message(
                "user",
                json!([{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]),
            ),
            message(
                "user",
                json!([{"type": "text", "text": "id: functions.Bash:0"}]),
            ),
        ];
        let sanitized = wire(&sanitize_tool_use_ids(messages));
        assert_eq!(sanitized[0]["content"][0]["id"], "functions_Bash_0");
        assert_eq!(
            sanitized[1]["content"][0]["tool_use_id"],
            "functions_Bash_0"
        );
        assert_eq!(sanitized[2]["content"][0]["text"], "id: functions.Bash:0");
    }

    #[test]
    fn strip_provider_specific_fields_leaves_other_keys() {
        let messages = vec![message(
            "assistant",
            json!([{"type": "thinking", "thinking": "hm", "signature": "s", "provider_specific_fields": {"a": 1}}]),
        )];
        assert_eq!(
            wire(&strip_provider_specific_fields(messages)),
            json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "s"}]}])
        );
    }

    #[test]
    fn strip_encrypted_reasoning_blocks_keeps_anthropic_signed_blocks() {
        let messages = vec![
            message(
                "assistant",
                json!([
                    {"type": "thinking", "thinking": "a", "signature": "litellm_encrypted_reasoning:xyz"},
                    {"type": "redacted_thinking", "data": "litellm_encrypted_reasoning:xyz"},
                    {"type": "thinking", "thinking": "b", "signature": "anthropic-sig"},
                ]),
            ),
            message(
                "assistant",
                json!([{"type": "thinking", "thinking": "a", "signature": "litellm_encrypted_reasoning:xyz"}]),
            ),
        ];
        assert_eq!(
            wire(&strip_encrypted_reasoning_blocks(messages)),
            json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "b", "signature": "anthropic-sig"}]}])
        );
    }

    #[test]
    fn strip_advisor_blocks_removes_matched_pairs_from_assistant_turns_only() {
        let messages = vec![
            message(
                "assistant",
                json!([
                    {"type": "server_tool_use", "id": "adv_1", "name": "advisor", "input": {}},
                    {"type": "advisor_tool_result", "tool_use_id": "adv_1", "content": "advice"},
                    {"type": "advisor_tool_result", "tool_use_id": "other", "content": "kept"},
                    {"type": "text", "text": "answer"}
                ]),
            ),
            message(
                "user",
                json!([{"type": "server_tool_use", "id": "adv_2", "name": "advisor", "input": {}}]),
            ),
        ];
        let stripped = wire(&strip_advisor_blocks(messages));
        assert_eq!(
            stripped[0]["content"],
            json!([
                {"type": "advisor_tool_result", "tool_use_id": "other", "content": "kept"},
                {"type": "text", "text": "answer"}
            ])
        );
        assert_eq!(stripped[1]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn flatten_web_search_results_rewrites_unencrypted_results_and_drops_their_server_tool_use() {
        let messages = vec![message(
            "assistant",
            json!([
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "rust"}},
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                    {"type": "web_search_result", "url": "https://r", "title": "Rust", "snippet": "fast"},
                    {"type": "web_search_result", "url": "https://q", "title": "", "snippet": ""}
                ]},
                {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "real"}},
                {"type": "web_search_tool_result", "tool_use_id": "s2", "content": [
                    {"type": "web_search_result", "url": "https://a", "title": "A", "snippet": "b", "encrypted_content": "enc"}
                ]},
                {"type": "text", "text": "done"}
            ]),
        )];
        assert_eq!(
            wire(&flatten_unencrypted_web_search_results(messages))[0]["content"],
            json!([
                {"type": "text", "text": "Web search results for 'rust':\n\nTitle: Rust\nURL: https://r\nSnippet: fast\n\nURL: https://q"},
                {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "real"}},
                {"type": "web_search_tool_result", "tool_use_id": "s2", "content": [
                    {"type": "web_search_result", "url": "https://a", "title": "A", "snippet": "b", "encrypted_content": "enc"}
                ]},
                {"type": "text", "text": "done"}
            ])
        );
    }

    #[test]
    fn flatten_web_search_results_renders_errors_and_empty_results() {
        let messages = vec![message(
            "assistant",
            json!([
                {"type": "web_search_tool_result", "tool_use_id": "e1", "content": {"type": "web_search_tool_result_error", "error_code": "max_uses"}},
                {"type": "web_search_tool_result", "tool_use_id": "e2", "content": []}
            ]),
        )];
        assert_eq!(
            wire(&flatten_unencrypted_web_search_results(messages))[0]["content"],
            json!([
                {"type": "text", "text": "Web search results:\n\nSearch failed: max_uses"},
                {"type": "text", "text": "Web search results:\n\nNo results were returned."}
            ])
        );
    }

    #[test]
    fn beta_header_merging_sorts_and_dedupes() {
        assert_eq!(merge_beta_headers(None, "b"), "b");
        assert_eq!(merge_beta_headers(Some("b, a ,b"), "c"), "a,b,c");
    }

    #[rstest]
    #[case("sk-ant-oat01-x", true)]
    #[case("Bearer sk-ant-oat01-x", true)]
    #[case("sk-ant-api03-x", false)]
    fn oauth_key_detection(#[case] value: &str, #[case] expected: bool) {
        assert_eq!(is_anthropic_oauth_key(value), expected);
    }

    #[test]
    fn compaction_beta_requires_param_or_signed_block() {
        let signed = vec![message(
            "assistant",
            json!([{"type": "compaction", "content": "c", "signature": "s"}]),
        )];
        let unsigned = vec![message(
            "assistant",
            json!([{"type": "compaction", "content": "c"}]),
        )];
        assert!(requires_native_compaction_beta(None, &signed));
        assert!(!requires_native_compaction_beta(None, &unsigned));
        assert!(requires_native_compaction_beta(Some(&json!({})), &unsigned));
    }

    #[test]
    fn capability_effort_gates_match_python() {
        let opus_4_5 = AnthropicModelCapabilities {
            supports_reasoning: true,
            supports_output_config: true,
            ..Default::default()
        };
        assert!(opus_4_5.supports_effort_param());
        assert!(opus_4_5.effort_level_rejection("high", "m").is_none());
        assert!(opus_4_5.effort_level_rejection("xhigh", "m").is_some());
        assert!(opus_4_5.effort_level_rejection("max", "m").is_some());
        let adaptive = AnthropicModelCapabilities {
            supports_adaptive_thinking: true,
            ..Default::default()
        };
        assert!(adaptive.effort_level_rejection("max", "m").is_none());
        assert!(adaptive.effort_level_rejection("xhigh", "m").is_some());
    }

    #[test]
    fn capabilities_deserialize_with_python_defaults() {
        let parsed: AnthropicModelCapabilities = serde_json::from_value(json!({})).unwrap();
        assert!(parsed.supports_sampling_params);
        assert!(!parsed.supports_reasoning);
        let flagged: AnthropicModelCapabilities = serde_json::from_value(
            json!({"effort_tiers": {"xhigh": true}, "supports_sampling_params": false}),
        )
        .unwrap();
        assert!(flagged.effort_tiers.xhigh);
        assert!(!flagged.supports_sampling_params);
    }
}
