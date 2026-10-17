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
        /// Which thinking block of the response a streamed piece belongs to.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<u64>,
    },
    RedactedThinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<u64>,
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
    /// The cache writes again, under the name Python's `Usage` gives them on a stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// The cache writes split by lifetime (`ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_token_details: Option<Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct CompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u64>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatCompletionsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Anthropic's server tool counts, `{"web_search_requests": n}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatCompletionsChoiceMessage {
    pub role: String,
    // Whether an empty turn is `None` or `""` is the provider's choice, not a
    // shared invariant: Anthropic's transform ends on `merged_text or None`
    // while Converse assigns the joined string unconditionally. Each config
    // mirrors its own, so keep this optional and serialize it even when None.
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionToolCallChunk>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<ChatCompletionThinkingBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsChoice {
    pub index: u64,
    pub message: ChatCompletionsChoiceMessage,
    pub finish_reason: String,
}

/// The normalized response handed back to the host.
///
/// There is deliberately no `id`: Python mints the `chatcmpl-…` id on the
/// `ModelResponse` it already created, and echoing the provider's own id here
/// would change it. Pinned by `response_carries_no_id` in the Anthropic chat transformation tests.
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsResponse {
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionsChoice>,
    pub usage: ChatCompletionsUsage,
}
