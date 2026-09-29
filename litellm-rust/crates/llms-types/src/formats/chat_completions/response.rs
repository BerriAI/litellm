use serde::{Deserialize, Serialize};

/// OpenAI `usage`, including the `prompt_tokens_details` split LiteLLM's Python
/// path reports so cost tracking sees the same numbers on either path.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub text_tokens: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsChoiceMessage {
    pub role: String,
    // Whether an empty turn is `None` or `""` is the provider's choice, not a
    // shared invariant: Anthropic's transform ends on `merged_text or None`
    // while Converse assigns the joined string unconditionally. Each config
    // mirrors its own, so keep this optional and serialize it even when None.
    pub content: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsResponse {
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionsChoice>,
    pub usage: ChatCompletionsUsage,
}
