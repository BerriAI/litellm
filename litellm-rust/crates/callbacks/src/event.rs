use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

/// One provider attempt of a logical call, as the loop that runs attempts identifies it.
/// `deployment` is the host's own handle; the event stream never sees model strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttemptInfo {
    /// 0-based across the whole logical call.
    pub index: u32,
    /// 0 is the primary group; n > 0 is fallback depth.
    pub group: u32,
    pub deployment: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CallEvent {
    AttemptStarted {
        attempt: AttemptInfo,
    },
    ResponseReceived {
        raw: RawResponse,
    },
    /// Per-attempt failure, the signal cooldown and health tables consume. User-facing
    /// failure callbacks fire once, on `Failed`, not here.
    AttemptFailed {
        attempt: AttemptInfo,
        /// The same deployment may be tried again after backoff.
        retryable: bool,
        /// The deployment should be taken out of rotation for this long.
        cooldown: Option<Duration>,
    },
    Succeeded {
        timing: Timing,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
    },
}
