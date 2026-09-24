use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
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
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
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
    fn request_round_trips_unchanged(#[case] request: Value) {
        assert_eq!(round_trip::<AnthropicMessagesRequest>(&request), request);
    }
}
