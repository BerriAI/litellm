use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

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
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

/// What the route knows about the request it is sending, for a host that logs it. The
/// route owns these facts; a host reads them beside the wire request and never rewrites
/// them.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestContext {
    pub model: String,
    pub custom_llm_provider: String,
    /// The route's parameters before the provider transformation.
    pub optional_params: Value,
    /// Optional-param names that carry credentials and must be redacted when logged.
    pub secret_fields: Vec<String>,
    /// The credential the route resolved for the provider call.
    pub api_key: Option<litellm_auth::SecretValue>,
}

/// The caller's request as the route reads it before the provider transform, offered to
/// the host for rewriting. `fields` names everything an answer may set.
#[derive(Clone, Debug, PartialEq)]
pub struct PublicRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub messages: Value,
    pub params: Map<String, Value>,
    pub fields: &'static [&'static str],
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

/// What a machine reports while it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MachineEvent {
    ResponseReceived { raw: RawResponse },
}

/// What an in-process host observes: the machine's own events between the driver's
/// start and terminal ones.
#[derive(Clone, Debug, PartialEq)]
pub enum CallEvent {
    Started {
        start_time: f64,
    },
    Machine(MachineEvent),
    Succeeded {
        timing: Timing,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
    },
}
