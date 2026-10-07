use serde_json::{Map, Value};
use strum::IntoStaticStr;

use crate::formats::chat::ReasoningEffort;
use crate::recognized::Recognized;

use super::{
    CacheControl, ContainerReference, ContentBlock, McpServer, MessageContent, MessagesCompaction,
    MessagesMetadata, OutputFormat, Safeguard, SystemPrompt, ToolChoice, ToolDefinition,
};

#[macro_rules_attribute::apply(wire_type)]
pub struct Message {
    pub role: super::MessageRole,
    pub content: MessageContent,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Hash, IntoStaticStr, Eq)]
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

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, IntoStaticStr, Eq)]
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

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MessagesTool {
    Builtin(BuiltinMessagesTool),
    Custom(CustomTool),
}

impl MessagesTool {
    pub fn definition(&self) -> &ToolDefinition {
        match self {
            Self::Builtin(tool) => tool.definition(),
            Self::Custom(tool) => &tool.definition,
        }
    }

    pub fn map_definition(self, f: impl FnOnce(ToolDefinition) -> ToolDefinition) -> Self {
        match self {
            Self::Builtin(tool) => Self::Builtin(tool.map_definition(f)),
            Self::Custom(tool) => Self::Custom(CustomTool {
                definition: f(tool.definition),
            }),
        }
    }
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(try_from = "ToolDefinition")]
pub struct CustomTool {
    #[serde(flatten)]
    pub definition: ToolDefinition,
}

impl TryFrom<ToolDefinition> for CustomTool {
    type Error = serde::de::value::Error;

    fn try_from(definition: ToolDefinition) -> Result<Self, Self::Error> {
        if definition.extra.contains_key("type")
            || !matches!(definition.name, Some(Recognized::Known(_)))
        {
            return Err(serde::de::Error::custom(
                "expected a custom tool with a string name and no type",
            ));
        }
        Ok(Self { definition })
    }
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type")]
pub enum BuiltinMessagesTool {
    #[serde(rename = "advisor_20260301")]
    Advisor(ToolDefinition),
    #[serde(rename = "tool_search_tool_regex_20251119")]
    ToolSearchRegex(ToolDefinition),
    #[serde(rename = "tool_search_tool_bm25_20251119")]
    ToolSearchBm25(ToolDefinition),
    #[serde(rename = "custom")]
    Custom(ToolDefinition),
    #[serde(rename = "web_search_20250305")]
    WebSearch(ToolDefinition),
    #[serde(rename = "computer_20250124")]
    Computer(ToolDefinition),
    #[serde(rename = "bash_20250124")]
    Bash(ToolDefinition),
    #[serde(rename = "text_editor_20250728")]
    TextEditor(ToolDefinition),
    #[serde(rename = "code_execution_20250825")]
    CodeExecution(ToolDefinition),
    #[serde(rename = "web_search_20260209")]
    WebSearch20260209(ToolDefinition),
    #[serde(rename = "computer_20241022")]
    Computer20241022(ToolDefinition),
    #[serde(rename = "bash_20241022")]
    Bash20241022(ToolDefinition),
    #[serde(rename = "text_editor_20241022")]
    TextEditor20241022(ToolDefinition),
    #[serde(rename = "text_editor_20250124")]
    TextEditor20250124(ToolDefinition),
    #[serde(rename = "code_execution_20250522")]
    CodeExecution20250522(ToolDefinition),
    #[serde(rename = "memory_20250818")]
    Memory(ToolDefinition),
    #[serde(rename = "web_fetch_20250910")]
    WebFetch(ToolDefinition),
    #[serde(rename = "web_fetch_20260209")]
    WebFetch20260209(ToolDefinition),
    #[serde(rename = "web_fetch_20260309")]
    WebFetch20260309(ToolDefinition),
    #[serde(rename = "web_fetch_20260318")]
    WebFetch20260318(ToolDefinition),
    #[serde(rename = "web_search_20260318")]
    WebSearch20260318(ToolDefinition),
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120(ToolDefinition),
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521(ToolDefinition),
    #[serde(rename = "computer_20251124")]
    Computer20251124(ToolDefinition),
    #[serde(rename = "text_editor_20250429")]
    TextEditor20250429(ToolDefinition),
}

impl BuiltinMessagesTool {
    pub fn definition(&self) -> &ToolDefinition {
        match self {
            Self::Advisor(definition)
            | Self::ToolSearchRegex(definition)
            | Self::ToolSearchBm25(definition)
            | Self::Custom(definition)
            | Self::WebSearch(definition)
            | Self::Computer(definition)
            | Self::Bash(definition)
            | Self::TextEditor(definition)
            | Self::CodeExecution(definition)
            | Self::WebSearch20260209(definition)
            | Self::Computer20241022(definition)
            | Self::Bash20241022(definition)
            | Self::TextEditor20241022(definition)
            | Self::TextEditor20250124(definition)
            | Self::CodeExecution20250522(definition)
            | Self::Memory(definition)
            | Self::WebFetch(definition)
            | Self::WebFetch20260209(definition)
            | Self::WebFetch20260309(definition)
            | Self::WebFetch20260318(definition)
            | Self::WebSearch20260318(definition)
            | Self::CodeExecution20260120(definition)
            | Self::CodeExecution20260521(definition)
            | Self::Computer20251124(definition)
            | Self::TextEditor20250429(definition) => definition,
        }
    }

    pub fn map_definition(self, f: impl FnOnce(ToolDefinition) -> ToolDefinition) -> Self {
        match self {
            Self::Advisor(definition) => Self::Advisor(f(definition)),
            Self::ToolSearchRegex(definition) => Self::ToolSearchRegex(f(definition)),
            Self::ToolSearchBm25(definition) => Self::ToolSearchBm25(f(definition)),
            Self::Custom(definition) => Self::Custom(f(definition)),
            Self::WebSearch(definition) => Self::WebSearch(f(definition)),
            Self::Computer(definition) => Self::Computer(f(definition)),
            Self::Bash(definition) => Self::Bash(f(definition)),
            Self::TextEditor(definition) => Self::TextEditor(f(definition)),
            Self::CodeExecution(definition) => Self::CodeExecution(f(definition)),
            Self::WebSearch20260209(definition) => Self::WebSearch20260209(f(definition)),
            Self::Computer20241022(definition) => Self::Computer20241022(f(definition)),
            Self::Bash20241022(definition) => Self::Bash20241022(f(definition)),
            Self::TextEditor20241022(definition) => Self::TextEditor20241022(f(definition)),
            Self::TextEditor20250124(definition) => Self::TextEditor20250124(f(definition)),
            Self::CodeExecution20250522(definition) => Self::CodeExecution20250522(f(definition)),
            Self::Memory(definition) => Self::Memory(f(definition)),
            Self::WebFetch(definition) => Self::WebFetch(f(definition)),
            Self::WebFetch20260209(definition) => Self::WebFetch20260209(f(definition)),
            Self::WebFetch20260309(definition) => Self::WebFetch20260309(f(definition)),
            Self::WebFetch20260318(definition) => Self::WebFetch20260318(f(definition)),
            Self::WebSearch20260318(definition) => Self::WebSearch20260318(f(definition)),
            Self::CodeExecution20260120(definition) => Self::CodeExecution20260120(f(definition)),
            Self::CodeExecution20260521(definition) => Self::CodeExecution20260521(f(definition)),
            Self::Computer20251124(definition) => Self::Computer20251124(f(definition)),
            Self::TextEditor20250429(definition) => Self::TextEditor20250429(f(definition)),
        }
    }

    pub fn is_web_search(&self) -> bool {
        match self {
            Self::WebSearch(_) | Self::WebSearch20260209(_) | Self::WebSearch20260318(_) => true,
            Self::Advisor(_)
            | Self::ToolSearchRegex(_)
            | Self::ToolSearchBm25(_)
            | Self::Custom(_)
            | Self::Computer(_)
            | Self::Bash(_)
            | Self::TextEditor(_)
            | Self::CodeExecution(_)
            | Self::Computer20241022(_)
            | Self::Bash20241022(_)
            | Self::TextEditor20241022(_)
            | Self::TextEditor20250124(_)
            | Self::CodeExecution20250522(_)
            | Self::Memory(_)
            | Self::WebFetch(_)
            | Self::WebFetch20260209(_)
            | Self::WebFetch20260309(_)
            | Self::WebFetch20260318(_)
            | Self::CodeExecution20260120(_)
            | Self::CodeExecution20260521(_)
            | Self::Computer20251124(_)
            | Self::TextEditor20250429(_) => false,
        }
    }
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextTrigger {
    InputTokens {
        value: Recognized<i64>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type")]
pub enum ContextEdit {
    #[serde(rename = "compact_20260112")]
    Compact {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "crate::serde_compat::deserialize_present"
        )]
        trigger: Option<Recognized<ContextTrigger>>,
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

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ContextManagement {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edits: Option<Vec<Recognized<ContextEdit>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OutputConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Recognized<EffortLevel>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub format: Option<Recognized<OutputFormat>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl OutputConfig {
    pub fn is_empty(&self) -> bool {
        self.effort.is_none() && self.format.is_none() && self.extra.is_empty()
    }
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    Summarized,
    Omitted,
    Updates,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct EnabledThinking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<Recognized<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<Recognized<ThinkingDisplay>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct AdaptiveThinking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<Recognized<ThinkingDisplay>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct DisabledThinking {
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
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

#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(flatten)]
    pub params: MessagesOptionalParams,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MessagesOptionalParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub metadata: Option<Recognized<MessagesMetadata>>,
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
    pub tools: Option<Vec<Recognized<MessagesTool>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub tool_choice: Option<Recognized<ToolChoice>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Recognized<ThinkingConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub container: Option<Recognized<ContainerReference>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub mcp_servers: Option<Vec<Recognized<McpServer>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Recognized<ContextManagement>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub output_format: Option<Recognized<OutputFormat>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Recognized<OutputConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<Recognized<Speed>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Recognized<ReasoningEffort>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub compaction: Option<Recognized<MessagesCompaction>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub cache_control: Option<Recognized<CacheControl>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::serde_compat::deserialize_present"
    )]
    pub safeguards: Option<Recognized<Vec<Recognized<Safeguard>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Message {
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

    fn round_trip<T: serde::de::DeserializeOwned + serde::Serialize>(value: &Value) -> Value {
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
        let request: MessagesRequest = serde_json::from_value(body.clone()).unwrap();

        assert_eq!(request.params.max_tokens, Some(16));
        assert_eq!(request.params.stream, Some(true));
        assert_eq!(
            request.params.safeguards.as_ref().unwrap().known().unwrap()[0]
                .known()
                .unwrap()
                .safeguard_type,
            "dangerous_tool_use"
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
    #[case::string_content(json!({"role": "user", "content": "hi"}), vec![])]
    #[case::block_content(
        json!({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}),
        vec![ContentBlock::text("a"), ContentBlock::text("b")],
    )]
    fn message_blocks_list_only_block_content(
        #[case] message: Value,
        #[case] expected: Vec<ContentBlock>,
    ) {
        let message: Message = serde_json::from_value(message).unwrap();
        assert_eq!(message.blocks(), expected.as_slice());
    }

    #[rstest]
    #[case::replaces_string_content(json!({"role": "assistant", "content": "old", "name": "kept"}))]
    #[case::replaces_block_content(json!({"role": "assistant", "content": [{"type": "text", "text": "old"}], "name": "kept"}))]
    fn with_blocks_replaces_content_and_keeps_the_rest(#[case] message: Value) {
        let message: Message = serde_json::from_value(message).unwrap();
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
        assert_eq!(round_trip::<MessagesRequest>(&request), request);
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
        json!({"type": "advisor_20260301", "name": "advisor", "model": "model"}),
        MessagesTool::Builtin(BuiltinMessagesTool::Advisor(ToolDefinition {
            name: Some(Recognized::Known("advisor".into())),
            model: Some(Recognized::Known("model".into())),
            ..ToolDefinition::default()
        }))
    )]
    #[case::regex_tool_search(
        json!({"type": "tool_search_tool_regex_20251119"}),
        MessagesTool::Builtin(BuiltinMessagesTool::ToolSearchRegex(ToolDefinition::default()))
    )]
    #[case::bm25_tool_search(
        json!({"type": "tool_search_tool_bm25_20251119"}),
        MessagesTool::Builtin(BuiltinMessagesTool::ToolSearchBm25(ToolDefinition::default()))
    )]
    #[case::custom_tool_without_a_type(
        json!({"name": "f", "input_schema": {}}),
        MessagesTool::Custom(CustomTool { definition: ToolDefinition {
            name: Some(Recognized::Known("f".into())),
            input_schema: Some(Recognized::Known(crate::json_schema::JsonSchema::Object(Box::default()))),
            ..ToolDefinition::default()
        } })
    )]
    #[case::web_search(
        json!({"type": "web_search_20250305", "name": "web_search"}),
        MessagesTool::Builtin(BuiltinMessagesTool::WebSearch(ToolDefinition {
            name: Some(Recognized::Known("web_search".into())),
            ..ToolDefinition::default()
        }))
    )]
    fn tools_expose_their_definitions(#[case] wire: Value, #[case] expected: MessagesTool) {
        let parsed: Recognized<MessagesTool> = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(parsed, Recognized::Known(expected));
        assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
    }

    #[rstest]
    #[case::compact(
        json!({"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1}}),
        Recognized::Known(ContextEdit::Compact {
            trigger: Some(Recognized::Known(ContextTrigger::InputTokens {
                value: Recognized::Known(1),
                extra: Map::new(),
            })),
            extra: Map::new(),
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
            edits: Some(vec![Recognized::Known(ContextEdit::Compact { trigger: None, extra: Map::new() })]),
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

    #[rstest]
    #[case::omitted(None, json!({}))]
    #[case::null(Some(json!(null)), json!({"output_config":null}))]
    #[case::empty(Some(json!({})), json!({"output_config":{}}))]
    #[case::effort(Some(json!({"effort":"high"})), json!({"output_config":{"effort":"high"}}))]
    #[case::opaque(Some(json!({"format":{"type":"future","schema":null}})), json!({"output_config":{"format":{"type":"future","schema":null}}}))]
    fn per_turn_output_config_preserves_extension_presence(
        #[case] config: Option<Value>,
        #[case] extension: Value,
    ) {
        let message = Message {
            role: super::super::MessageRole::System,
            content: MessageContent::Blocks(vec![ContentBlock::text("# Environment")]),
            extra: config
                .map(|value| Map::from_iter([("output_config".into(), value)]))
                .unwrap_or_default(),
        };
        let wire = Value::Object(
            Map::from_iter([
                ("role".into(), json!("system")),
                (
                    "content".into(),
                    json!([{"type":"text","text":"# Environment"}]),
                ),
            ])
            .into_iter()
            .chain(extension.as_object().unwrap().clone())
            .collect(),
        );
        assert_eq!(serde_json::to_value(&message).unwrap(), wire);
        assert_eq!(serde_json::from_value::<Message>(wire).unwrap(), message);
    }

    #[rstest]
    fn structured_output_config_keeps_format_and_effort_together() {
        let config = OutputConfig {
            effort: Some(Recognized::Known(EffortLevel::Xhigh)),
            format: Some(Recognized::Known(OutputFormat {
                format_type: super::super::OutputFormatType::JsonSchema,
                schema: Some(Recognized::Known(crate::json_schema::JsonSchema::Object(
                    Box::default(),
                ))),
                strict: None,
                extra: Map::new(),
            })),
            extra: Map::from_iter([("task_budget".into(), json!({"type":"tokens","total":4096}))]),
        };
        let wire = json!({"effort":"xhigh","format":{"type":"json_schema","schema":{}},"task_budget":{"type":"tokens","total":4096}});
        assert_eq!(serde_json::to_value(&config).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<OutputConfig>(wire).unwrap(),
            config
        );
    }

    #[rstest]
    #[case::omitted(None, json!({"type":"adaptive"}))]
    #[case::summarized(Some(ThinkingDisplay::Summarized), json!({"type":"adaptive","display":"summarized"}))]
    #[case::omitted_display(Some(ThinkingDisplay::Omitted), json!({"type":"adaptive","display":"omitted"}))]
    #[case::updates(Some(ThinkingDisplay::Updates), json!({"type":"adaptive","display":"updates"}))]
    fn native_messages_thinking_display_contract(
        #[case] display: Option<ThinkingDisplay>,
        #[case] wire: Value,
    ) {
        let thinking = ThinkingConfig::adaptive(display);
        assert_eq!(serde_json::to_value(&thinking).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<ThinkingConfig>(wire).unwrap(),
            thinking
        );
    }

    #[rstest]
    #[case::string(json!("hi"))]
    #[case::scalar(json!(123))]
    #[case::list(json!([{"role":"user","content":"hello"}]))]
    #[case::null_content(json!({"role":"user","content":null}))]
    #[case::scalar_block(json!({"role":"user","content":["not a block"]}))]
    #[case::system_string_block(json!({"role":"system","content":["tool_addition"]}))]
    fn malformed_message_entry_is_rejected(#[case] entry: Value) {
        let wire = json!({"model":"model","messages":[entry,{"role":"user","content":"hello"}]});
        assert!(serde_json::from_value::<MessagesRequest>(wire).is_err());
    }

    #[rstest]
    fn non_string_text_is_rejected_at_request_boundary() {
        let wire = json!({"model":"model","messages":[{"role":"user","content":[
            {"type":"text","text":123},
            {"type":"tool_result","tool_use_id":"x","content":"y"}
        ]}]});
        assert!(serde_json::from_value::<MessagesRequest>(wire).is_err());
    }
}
