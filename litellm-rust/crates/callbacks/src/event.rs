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
    pub passthrough_fields: Passthrough,
    /// Optional-param names that carry credentials and must be redacted when logged.
    pub secret_fields: Vec<String>,
}

/// Body keys whose values are the caller's inputs, unchanged by the route. The only way to
/// build one is to compare the two, so a route cannot name a key it rewrote.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Passthrough(Vec<String>);

impl Passthrough {
    pub fn unchanged(caller: &Map<String, Value>, body: &Value) -> Self {
        Self(
            caller
                .iter()
                .filter(|(name, value)| body.get(name.as_str()) == Some(*value))
                .map(|(name, _)| name.clone())
                .collect(),
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.iter().any(|field| field == name)
    }
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptInfo {
    /// Spans every attempt of one logical call: Python's `litellm_trace_id`. Each attempt's
    /// own call id is the host's to mint, as the `@client` wrapper does per call.
    pub trace_id: String,
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
    Succeeded {
        timing: Timing,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
    },
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::unchanged_scalar(json!({"pages": [0]}), json!({"pages": [0]}), &["pages"])]
    #[case::unchanged_explicit_null(json!({"pages": null}), json!({"pages": null}), &["pages"])]
    #[case::unchanged_nested_object(
        json!({"document": {"type": "document_url", "document_url": "https://a/b.pdf"}}),
        json!({"document": {"type": "document_url", "document_url": "https://a/b.pdf"}, "model": "m"}),
        &["document"]
    )]
    #[case::rewritten_value(
        json!({"document": {"type": "document_url", "document_url": "https://a/b.pdf"}}),
        json!({"document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}}),
        &[]
    )]
    #[case::dropped_nested_field(
        json!({"document": {"type": "image_url", "image_url": "https://a/b.png", "document_name": "b.png"}}),
        json!({"document": {"type": "image_url", "image_url": "https://a/b.png"}}),
        &[]
    )]
    #[case::added_nested_field(
        json!({"document": {"type": "image_url", "image_url": "https://a/b.png"}}),
        json!({"document": {"type": "image_url", "image_url": "https://a/b.png", "detail": "high"}}),
        &[]
    )]
    #[case::reordered_array(json!({"pages": [0, 1]}), json!({"pages": [1, 0]}), &[])]
    #[case::consumed_by_the_route(json!({"api_key": "k", "pages": [0]}), json!({"pages": [0]}), &["pages"])]
    #[case::added_by_the_route(json!({}), json!({"model": "m"}), &[])]
    #[case::non_object_body(json!({"pages": [0]}), json!([{"pages": [0]}]), &[])]
    fn passthrough_is_exactly_the_callers_unchanged_keys(
        #[case] caller: Value,
        #[case] body: Value,
        #[case] expected: &[&str],
    ) {
        let passthrough = Passthrough::unchanged(caller.as_object().unwrap(), &body);
        assert_eq!(passthrough.iter().collect::<Vec<_>>(), expected);
    }
}
