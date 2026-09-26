use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use strum::IntoStaticStr;

use crate::{llms::openai::ReasoningEffort, recognized::Recognized};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub block_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            block_type: Some("text".to_string()),
            text: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn is_type(&self, block_type: &str) -> bool {
        self.block_type.as_deref() == Some(block_type)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub cache_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl EffortLevel {
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl From<EffortLevel> for ReasoningEffort {
    fn from(level: EffortLevel) -> Self {
        match level {
            EffortLevel::Low => Self::Low,
            EffortLevel::Medium => Self::Medium,
            EffortLevel::High => Self::High,
            EffortLevel::Xhigh => Self::Xhigh,
            EffortLevel::Max => Self::Max,
        }
    }
}

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Speed {
    Fast,
    Standard,
}

impl Speed {
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

/// The tools whose presence changes how the request is sent. Every other tool, custom or
/// server, deserializes as `Recognized::Unrecognized` and passes through verbatim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicTool {
    #[serde(rename = "advisor_20260301")]
    Advisor {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "tool_search_tool_regex_20251119")]
    ToolSearchRegex {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "tool_search_tool_bm25_20251119")]
    ToolSearchBm25 {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContextEdit {
    #[serde(rename = "compact_20260112")]
    Compact {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "clear_tool_uses_20250919")]
    ClearToolUses {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextManagement {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edits: Option<Vec<Recognized<ContextEdit>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Recognized<EffortLevel>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl OutputConfig {
    pub fn is_empty(&self) -> bool {
        self.effort.is_none() && self.format.is_none() && self.extra.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    Summarized,
    Omitted,
    Updates,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnabledThinking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<Recognized<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<Recognized<ThinkingDisplay>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveThinking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<Recognized<ThinkingDisplay>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DisabledThinking {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ThinkingConfig {
    Enabled(EnabledThinking),
    Adaptive(AdaptiveThinking),
    Disabled(DisabledThinking),
}

impl ThinkingConfig {
    pub fn enabled(budget_tokens: u64) -> Self {
        Self::Enabled(EnabledThinking {
            budget_tokens: Some(Recognized::Known(budget_tokens)),
            ..EnabledThinking::default()
        })
    }

    pub fn adaptive(display: Option<ThinkingDisplay>) -> Self {
        Self::Adaptive(AdaptiveThinking {
            display: display.map(Recognized::Known),
            ..AdaptiveThinking::default()
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(flatten)]
    pub params: AnthropicMessagesOptionalParams,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesOptionalParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Recognized<AnthropicTool>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Recognized<ThinkingConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Recognized<ContextManagement>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Recognized<OutputConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<Recognized<Speed>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Recognized<ReasoningEffort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl AnthropicMessage {
    pub fn blocks(&self) -> &[ContentBlock] {
        match &self.content {
            MessageContent::Blocks(blocks) => blocks,
            MessageContent::Text(_) => &[],
        }
    }

    pub fn with_blocks(self, blocks: Vec<ContentBlock>) -> Self {
        Self {
            content: MessageContent::Blocks(blocks),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn round_trip<T: serde::de::DeserializeOwned + Serialize>(value: &Value) -> Value {
        let parsed: T = serde_json::from_value(value.clone()).unwrap();
        serde_json::to_value(parsed).unwrap()
    }

    #[rstest]
    #[case::text(json!({"type": "text", "text": "hi"}))]
    #[case::text_with_citations_and_cache_control(json!({
        "type": "text",
        "text": "hi",
        "citations": [{"type": "char_location", "cited_text": "x"}],
        "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global", "future": 1}
    }))]
    #[case::image(json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}))]
    #[case::thinking(json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}))]
    #[case::redacted_thinking(json!({"type": "redacted_thinking", "data": "opaque"}))]
    #[case::tool_use(json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"q": [1, null]}}))]
    #[case::tool_result_with_text(json!({"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok", "is_error": false}))]
    #[case::tool_result_with_blocks(json!({"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "ok"}]}))]
    #[case::web_search_result_with_nulls(json!({
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "url": "u", "page_age": null, "encrypted_content": ""}]
    }))]
    #[case::provider_specific_fields(json!({"type": "tool_use", "id": "t", "name": "f", "input": {}, "provider_specific_fields": {"x": 1}}))]
    #[case::untyped(json!({"unknown": {"nested": true}}))]
    fn content_block_round_trips_unchanged(#[case] block: Value) {
        assert_eq!(round_trip::<ContentBlock>(&block), block);
    }

    #[test]
    fn request_splits_required_fields_from_optional_params() {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": true,
            "safeguards": [{"type": "dangerous_tool_use"}]
        });
        let request: AnthropicMessagesRequest = serde_json::from_value(body.clone()).unwrap();

        assert_eq!(
            (
                request.params.max_tokens,
                request.params.stream,
                request
                    .params
                    .extra
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            ),
            (Some(16_u64), Some(true), vec!["safeguards"])
        );
        assert_eq!(serde_json::to_value(request).unwrap(), body);
    }

    #[test]
    fn text_constructor_serializes_as_a_text_block() {
        assert_eq!(
            serde_json::to_value(ContentBlock::text("hello")).unwrap(),
            json!({"type": "text", "text": "hello"})
        );
    }

    #[rstest]
    #[case::same_type(json!({"type": "tool_use"}), "tool_use", true)]
    #[case::other_type(json!({"type": "tool_result"}), "tool_use", false)]
    #[case::prefix_of_type(json!({"type": "tool_use"}), "tool", false)]
    #[case::no_type(json!({"text": "x"}), "text", false)]
    fn is_type_matches_the_exact_block_type(
        #[case] block: Value,
        #[case] block_type: &str,
        #[case] expected: bool,
    ) {
        let block: ContentBlock = serde_json::from_value(block).unwrap();
        assert_eq!(block.is_type(block_type), expected);
    }

    #[rstest]
    #[case::string_content(json!({"role": "user", "content": "hi"}), vec![])]
    #[case::block_content(
        json!({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}),
        vec![ContentBlock::text("a"), ContentBlock::text("b")],
    )]
    fn message_blocks_list_only_block_content(
        #[case] message: Value,
        #[case] expected: Vec<ContentBlock>,
    ) {
        let message: AnthropicMessage = serde_json::from_value(message).unwrap();
        assert_eq!(message.blocks(), expected.as_slice());
    }

    #[rstest]
    #[case::replaces_string_content(json!({"role": "assistant", "content": "old", "name": "kept"}))]
    #[case::replaces_block_content(json!({"role": "assistant", "content": [{"type": "text", "text": "old"}], "name": "kept"}))]
    fn with_blocks_replaces_content_and_keeps_the_rest(#[case] message: Value) {
        let message: AnthropicMessage = serde_json::from_value(message).unwrap();
        assert_eq!(
            serde_json::to_value(message.with_blocks(vec![ContentBlock::text("new")])).unwrap(),
            json!({"role": "assistant", "content": [{"type": "text", "text": "new"}], "name": "kept"})
        );
    }

    #[rstest]
    #[case::minimal(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))]
    #[case::reasoning_effort_compaction_and_unknown_fields(json!({
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "max_tokens": 8,
        "reasoning_effort": "high",
        "compaction": {"type": "auto"},
        "safeguards": [{"type": "dangerous_tool_use", "classifier_context": {"v": 1}}],
        "metadata": {"user_id": "u"}
    }))]
    #[case::typed_thinking_and_output_config(json!({
        "model": "m",
        "messages": [],
        "thinking": {"type": "enabled", "budget_tokens": 2048, "display": "omitted", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
        "output_config": {"effort": "xhigh", "format": {"type": "json_schema", "schema": {}}, "task_budget": {"type": "tokens", "total": 4096}}
    }))]
    #[case::unrecognized_values_are_kept_verbatim(json!({
        "model": "m",
        "messages": [],
        "reasoning_effort": "turbo",
        "thinking": {"type": "adaptive", "display": "loud"},
        "output_config": {"effort": 5}
    }))]
    #[case::unrecognized_shapes_are_kept_verbatim(json!({
        "model": "m",
        "messages": [],
        "reasoning_effort": 3,
        "thinking": {"type": "future", "budget_tokens": 1},
        "output_config": "bogus"
    }))]
    #[case::tools_speed_and_context_management(json!({
        "model": "m",
        "messages": [],
        "speed": "fast",
        "tools": [
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"type": "custom", "name": "f", "input_schema": {}},
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-6"},
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
            {"type": "tool_search_tool_bm25_20251119"}
        ],
        "context_management": {"edits": [
            {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}},
            {"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 3}},
            {"type": "clear_thinking_20251015"},
            {"type": "future_edit"},
            {}
        ], "future": true}
    }))]
    #[case::unrecognized_tools_speed_and_context_management_are_kept_verbatim(json!({
        "model": "m",
        "messages": [],
        "speed": "turbo",
        "tools": ["none", 5],
        "context_management": [{"type": "compaction", "compact_threshold": 5}]
    }))]
    fn request_round_trips_unchanged(#[case] request: Value) {
        assert_eq!(round_trip::<AnthropicMessagesRequest>(&request), request);
    }

    #[rstest]
    #[case::enabled(
        json!({"type": "enabled", "budget_tokens": 2048, "display": "omitted"}),
        ThinkingConfig::Enabled(EnabledThinking {
            budget_tokens: Some(Recognized::Known(2048)),
            display: Some(Recognized::Known(ThinkingDisplay::Omitted)),
            extra: Map::new(),
        })
    )]
    #[case::enabled_without_budget(
        json!({"type": "enabled"}),
        ThinkingConfig::Enabled(EnabledThinking::default())
    )]
    #[case::enabled_with_unrecognized_budget(
        json!({"type": "enabled", "budget_tokens": "lots"}),
        ThinkingConfig::Enabled(EnabledThinking {
            budget_tokens: Some(Recognized::Unrecognized(json!("lots"))),
            ..EnabledThinking::default()
        })
    )]
    #[case::adaptive_with_unrecognized_display(
        json!({"type": "adaptive", "display": "loud"}),
        ThinkingConfig::Adaptive(AdaptiveThinking {
            display: Some(Recognized::Unrecognized(json!("loud"))),
            extra: Map::new(),
        })
    )]
    #[case::disabled_keeps_extra_fields(
        json!({"type": "disabled", "future": true}),
        ThinkingConfig::Disabled(DisabledThinking {
            extra: Map::from_iter([("future".to_string(), json!(true))]),
        })
    )]
    fn thinking_config_parses_every_documented_type_leniently(
        #[case] thinking: Value,
        #[case] expected: ThinkingConfig,
    ) {
        assert_eq!(
            serde_json::from_value::<ThinkingConfig>(thinking).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::advisor(
        json!({"type": "advisor_20260301", "name": "advisor"}),
        Recognized::Known(AnthropicTool::Advisor { extra: Map::from_iter([("name".to_string(), json!("advisor"))]) })
    )]
    #[case::regex_tool_search(
        json!({"type": "tool_search_tool_regex_20251119"}),
        Recognized::Known(AnthropicTool::ToolSearchRegex { extra: Map::new() })
    )]
    #[case::bm25_tool_search(
        json!({"type": "tool_search_tool_bm25_20251119"}),
        Recognized::Known(AnthropicTool::ToolSearchBm25 { extra: Map::new() })
    )]
    #[case::custom_tool_without_a_type(
        json!({"name": "advisor", "input_schema": {}}),
        Recognized::Unrecognized(json!({"name": "advisor", "input_schema": {}}))
    )]
    #[case::other_server_tool(
        json!({"type": "web_search_20250305", "name": "web_search"}),
        Recognized::Unrecognized(json!({"type": "web_search_20250305", "name": "web_search"}))
    )]
    #[case::not_an_object(json!("advisor_20260301"), Recognized::Unrecognized(json!("advisor_20260301")))]
    fn tools_are_recognized_by_their_exact_type(
        #[case] tool: Value,
        #[case] expected: Recognized<AnthropicTool>,
    ) {
        assert_eq!(
            serde_json::from_value::<Recognized<AnthropicTool>>(tool).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::compact(
        json!({"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1}}),
        Recognized::Known(ContextEdit::Compact {
            extra: Map::from_iter([("trigger".to_string(), json!({"type": "input_tokens", "value": 1}))]),
        })
    )]
    #[case::clear_tool_uses(
        json!({"type": "clear_tool_uses_20250919"}),
        Recognized::Known(ContextEdit::ClearToolUses { extra: Map::new() })
    )]
    #[case::clear_thinking(
        json!({"type": "clear_thinking_20251015"}),
        Recognized::Known(ContextEdit::ClearThinking { extra: Map::new() })
    )]
    #[case::unknown_type(json!({"type": "future"}), Recognized::Unrecognized(json!({"type": "future"})))]
    #[case::no_type(json!({}), Recognized::Unrecognized(json!({})))]
    fn context_edits_are_recognized_by_their_exact_type(
        #[case] edit: Value,
        #[case] expected: Recognized<ContextEdit>,
    ) {
        assert_eq!(
            serde_json::from_value::<Recognized<ContextEdit>>(edit).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::edits(
        json!({"edits": [{"type": "compact_20260112"}]}),
        Recognized::Known(ContextManagement {
            edits: Some(vec![Recognized::Known(ContextEdit::Compact { extra: Map::new() })]),
            extra: Map::new(),
        })
    )]
    #[case::object_without_edits(
        json!({"future": 1}),
        Recognized::Known(ContextManagement {
            edits: None,
            extra: Map::from_iter([("future".to_string(), json!(1))]),
        })
    )]
    #[case::openai_list(json!([{"type": "compaction"}]), Recognized::Unrecognized(json!([{"type": "compaction"}])))]
    #[case::edits_not_a_list(json!({"edits": 5}), Recognized::Unrecognized(json!({"edits": 5})))]
    #[case::scalar(json!("compaction"), Recognized::Unrecognized(json!("compaction")))]
    fn context_management_is_known_only_as_an_edits_object(
        #[case] value: Value,
        #[case] expected: Recognized<ContextManagement>,
    ) {
        assert_eq!(
            serde_json::from_value::<Recognized<ContextManagement>>(value).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::fast(json!("fast"), Recognized::Known(Speed::Fast))]
    #[case::standard(json!("standard"), Recognized::Known(Speed::Standard))]
    #[case::unknown(json!("turbo"), Recognized::Unrecognized(json!("turbo")))]
    #[case::wrong_case(json!("Fast"), Recognized::Unrecognized(json!("Fast")))]
    #[case::not_a_string(json!(1), Recognized::Unrecognized(json!(1)))]
    fn speed_is_known_only_as_a_documented_value(
        #[case] value: Value,
        #[case] expected: Recognized<Speed>,
    ) {
        assert_eq!(
            serde_json::from_value::<Recognized<Speed>>(value).unwrap(),
            expected
        );
    }

    #[rstest]
    fn speed_names_match_the_wire(#[values(Speed::Fast, Speed::Standard)] speed: Speed) {
        assert_eq!(serde_json::to_value(speed).unwrap(), json!(speed.as_str()));
    }

    #[rstest]
    fn effort_level_names_match_the_wire(
        #[values(
            EffortLevel::Low,
            EffortLevel::Medium,
            EffortLevel::High,
            EffortLevel::Xhigh,
            EffortLevel::Max
        )]
        level: EffortLevel,
    ) {
        assert_eq!(serde_json::to_value(level).unwrap(), json!(level.as_str()));
        assert_eq!(
            serde_json::to_value(ReasoningEffort::from(level)).unwrap(),
            json!(level.as_str())
        );
    }
}
