use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

/// Seconds since the Unix epoch, on one clock for every host.
pub fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub start_time: f64,
    pub end_time: f64,
}

/// The provider request as it is about to leave, offered to the host for rewriting.
#[derive(Clone, Debug, PartialEq)]
pub struct WireRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
    pub optional_params: Value,
    /// Body keys whose values are the caller's own inputs, unchanged by the route.
    pub caller_fields: Vec<String>,
    /// Optional-param names that carry credentials and must be redacted when logged.
    pub secret_fields: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawResponse {
    pub body: String,
}

/// Whether a failure surfaced inside the call, including a host op the call asked for,
/// or in a host step around it (preparing the arguments, finalizing the response).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureOrigin {
    Call,
    Host,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CallEvent {
    ResponseReceived {
        raw: RawResponse,
    },
    Succeeded {
        timing: Timing,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
    },
}
