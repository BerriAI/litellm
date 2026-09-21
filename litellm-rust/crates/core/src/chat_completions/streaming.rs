use litellm_host::machine::RouteMachine;
use litellm_llms::base_llm::base_model_iterator::ChatCompletions;

use crate::streaming::{StreamingCall, StreamingRoute};

pub struct ChatCompletionsCall {
    pub model: String,
    pub messages: Box<[litellm_types::llms::openai::ChatMessage]>,
    pub optional_params: std::collections::BTreeMap<String, serde_json::Value>,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<litellm_auth::SecretValue>,
    pub api_base: Option<String>,
    pub extra_headers: Box<[(String, String)]>,
    pub timeout: Option<std::time::Duration>,
}

pub struct Streaming;

impl StreamingCall for Streaming {
    type Contract = ChatCompletions;
    type Request = ChatCompletionsCall;
    type Response = litellm_types::utils::ChatCompletionsResponse;
}

pub fn chat_completions_stream_machine() -> RouteMachine<StreamingRoute<Streaming>> {
    todo!(
        "Select a typed native or translated stream and preserve existing provider acceptance gates until the streaming path is implemented"
    )
}
