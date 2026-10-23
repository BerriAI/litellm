use std::{collections::BTreeMap, time::Duration};

use litellm_auth::{SecretValue, Sourced};
use serde_json::Value;

#[derive(Clone, Default)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<Sourced<SecretValue>>,
    pub api_base: Option<Sourced<String>>,
    pub extra_headers: BTreeMap<String, Sourced<Value>>,
    pub deployment_params: BTreeMap<String, Sourced<Value>>,
    pub optional_params: BTreeMap<String, Sourced<Value>>,
    pub timeout: Option<Duration>,
}
