use std::time::Duration;

use serde_json::{Map, Value};

#[derive(Clone, Debug, Default)]
pub struct RequestOptions {
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub extra_query: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    pub provider_connection: Map<String, Value>,
}
