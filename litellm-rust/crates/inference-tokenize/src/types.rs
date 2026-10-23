use std::time::Duration;

use litellm_auth::SecretValue;
use serde_json::{Map, Value};

#[derive(Clone, Default)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub extra_headers: Map<String, Value>,
    /// Shared deployment defaults; only supported tokenization options apply.
    pub deployment_params: Map<String, Value>,
    /// Explicit caller options override deployment defaults and are strictly validated.
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
}
