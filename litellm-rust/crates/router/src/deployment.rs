use std::time::Duration;

use litellm_core_utils::serde_compat::parse_pydantic_bool;
use litellm_inference_messages::{MessagesSettings, MessagesShaping};
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value};

pub const REQUEST_SETTINGS: [&str; 2] = ["drop_params", "additional_drop_params"];

#[derive(Clone, Debug, Default)]
pub struct Deployment {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub litellm_params: LitellmParams,
    pub timeout: Option<Duration>,
    pub messages_timeouts: MessagesTimeouts,
    pub drops_params_globally: bool,
    pub shaping: MessagesShaping,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MessagesTimeouts {
    pub timeout: Option<Duration>,
    pub stream_timeout: Option<Duration>,
}

impl MessagesTimeouts {
    pub fn for_call(&self, stream: bool, request_timeout: Option<Duration>) -> Option<Duration> {
        if stream {
            return self.stream_timeout.or(request_timeout).or(self.timeout);
        }
        self.timeout.or(request_timeout)
    }
}

impl Deployment {
    pub fn shaping_for(&self, request: &Map<String, Value>) -> MessagesShaping {
        let settings = &self.shaping.settings;
        let requested = request.get("drop_params").map(requests_drop);
        MessagesShaping {
            settings: MessagesSettings {
                drop_params: self.drops_params_globally
                    || requested.unwrap_or(settings.drop_params),
                drops_unsupported_params: self.drops_params_globally
                    || requested.unwrap_or(settings.drops_unsupported_params),
                additional_drop_params: request
                    .get("additional_drop_params")
                    .map_or_else(|| settings.additional_drop_params.clone(), drop_paths),
                ..settings.clone()
            },
            ..self.shaping.clone()
        }
    }
}

fn requests_drop(value: &Value) -> bool {
    match value {
        Value::Bool(on) => *on,
        Value::String(text) => parse_pydantic_bool(text) == Some(true),
        Value::Number(number) => number.as_f64() == Some(1.0),
        _ => false,
    }
}

pub(crate) fn drop_paths(value: &Value) -> Vec<String> {
    match value {
        Value::Array(paths) => paths
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}
