use std::time::Duration;

pub use litellm_llms::base_llm::chat::transformation::DropParams;
use litellm_llms_types::formats::chat_completions::ChatMessage;
use serde_json::{Map, Value};

pub enum ClientSideCredentials {
    Any,
    Only(Box<[String]>),
}

impl ClientSideCredentials {
    pub fn admits(&self, name: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Only(names) => names.iter().any(|allowed| allowed == name),
        }
    }
}

pub enum OptionalParams {
    OpenAi {
        params: Map<String, Value>,
        client_side: ClientSideCredentials,
        drop_params: DropParams,
    },
    ProviderMapped(Map<String, Value>),
}

/// A `/chat/completions` call as it crosses into the core.
///
/// The host says which names `optional_params` carries: the gateway passes the
/// OpenAI body as received, and the Python bridge passes the output of Python's
/// `map_openai_params`. The core maps OpenAI names to the provider's, and owns
/// the conversation translation, the provider call, and the response
/// normalization.
pub struct ChatCompletionsRequest<'a> {
    pub model: &'a str,
    pub messages: Value,
    pub optional_params: OptionalParams,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct ChatCompletionsCall {
    pub model: String,
    pub messages: Value,
    pub optional_params: OptionalParams,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

impl From<ChatCompletionsRequest<'_>> for ChatCompletionsCall {
    fn from(request: ChatCompletionsRequest<'_>) -> Self {
        Self {
            model: request.model.into(),
            messages: request.messages,
            optional_params: request.optional_params,
            api_key: request.api_key.map(str::to_owned),
            api_base: request.api_base.map(str::to_owned),
            custom_llm_provider: request.custom_llm_provider.map(str::to_owned),
            extra_headers: request.extra_headers,
            timeout: request.timeout,
        }
    }
}

pub struct ResolvedChatCompletionsRequest {
    pub messages: Vec<ChatMessage>,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}
