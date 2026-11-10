use std::time::Duration;

use litellm_inference_messages::MessagesShaping;
use litellm_router_types::LitellmParams;

#[derive(Clone, Debug, Default)]
pub struct Deployment {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub litellm_params: LitellmParams,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}
