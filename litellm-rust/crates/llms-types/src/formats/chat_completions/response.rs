use serde_json::{Map, Value};

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionToolCallFunctionChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionToolCallChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatCompletionToolCallFunctionChunk,
    pub index: i64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionThinkingBlock {
    Thinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    RedactedThinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
}

/// OpenAI `usage`, including the `prompt_tokens_details` split LiteLLM's Python
/// path reports so cost tracking sees the same numbers on either path.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub text_tokens: u64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatCompletionsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsChoiceMessage {
    pub role: String,
    // Whether an empty turn is `None` or `""` is the provider's choice, not a
    // shared invariant: Anthropic's transform ends on `merged_text or None`
    // while Converse assigns the joined string unconditionally. Each config
    // mirrors its own, so keep this optional and serialize it even when None.
    pub content: Option<String>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsChoice {
    pub index: u64,
    pub message: ChatCompletionsChoiceMessage,
    pub finish_reason: String,
}

/// The normalized response handed back to the host.
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionsChoice>,
    pub usage: ChatCompletionsUsage,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::provider(None)]
    #[case::reviewed(Some("reviewed-1".to_owned()))]
    fn response_round_trip_keeps_an_optional_host_id(#[case] id: Option<String>) {
        let response = ChatCompletionsResponse {
            id: id.clone(),
            created: 0,
            model: "test".into(),
            choices: Vec::new(),
            usage: ChatCompletionsUsage::default(),
        };
        let value = serde_json::to_value(&response).unwrap();

        assert_eq!(value.get("id"), id.as_ref().map(|id| json!(id)).as_ref());
        assert_eq!(
            serde_json::from_value::<ChatCompletionsResponse>(value).unwrap(),
            response
        );
    }
}
