use std::time::Duration;

use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_inference::Connection;
use litellm_inference_messages::MessagesShaping;

#[derive(Clone, Debug, Default)]
pub struct Deployment {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

impl Deployment {
    pub fn connection(&self) -> Connection {
        Connection {
            api_key: self
                .api_key
                .clone()
                .map(|key| Sourced::new(SecretValue::new(key), InputSource::Deployment)),
            api_base: self
                .api_base
                .clone()
                .map(|base| Sourced::new(base, InputSource::Deployment)),
            extra_headers: None,
            timeout: self.timeout,
        }
    }
}
