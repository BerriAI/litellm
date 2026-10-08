use std::time::Duration;

use serde_json::{Map, Value};

#[derive(Default)]
pub struct Connection {
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}
