use std::time::Duration;

use litellm_llms::{
    anthropic::common_utils::AnthropicModelCapabilities,
    base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// What the host knows about the deployment that the route cannot read off the body: the
/// model's cost-map flags and the caller's LiteLLM-level request settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesShaping {
    #[serde(default)]
    pub capabilities: AnthropicModelCapabilities,
    /// `litellm.drop_params` or the per-request `drop_params`.
    #[serde(default)]
    pub drop_params: bool,
    /// `litellm.reasoning_auto_summary` or `LITELLM_REASONING_AUTO_SUMMARY`.
    #[serde(default)]
    pub reasoning_auto_summary: bool,
    /// The deployment's `additional_drop_params`: dot paths removed from the request body.
    #[serde(default)]
    pub additional_drop_params: Vec<String>,
}

pub struct MessagesRequest<'a> {
    pub model: &'a str,
    pub body: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

pub struct ProviderMessagesRequest {
    pub provider: String,
    pub model: String,
    pub config: &'static dyn BaseAnthropicMessagesConfig,
    pub url: String,
    pub body: Value,
    pub upstream_headers: Vec<(String, String)>,
    pub timeout: Option<Duration>,
}
