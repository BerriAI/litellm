use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub use super::redaction::BodySnapshot;
use super::redaction::{redact_headers, redact_url, snapshot_json};

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event")]
pub enum LogEvent {
    #[serde(rename = "provider.request")]
    Request(ProviderRequestEvent),
    #[serde(rename = "provider.response")]
    Response(ProviderResponseEvent),
    #[serde(rename = "provider.stream.started")]
    StreamStarted(ProviderStreamStartedEvent),
    #[serde(rename = "provider.stream.completed")]
    StreamCompleted(ProviderStreamCompletedEvent),
    #[serde(rename = "provider.error")]
    Error(ProviderErrorEvent),
}

pub struct RequestEventInput {
    pub model: String,
    pub stream: bool,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

pub struct ResponseEventInput {
    pub call_id: String,
    pub provider: String,
    pub status: u16,
    pub duration_ms: u128,
    pub headers: Vec<(String, String)>,
    pub body: ResponseBody,
}

pub struct ErrorEventInput {
    pub call_id: String,
    pub provider: String,
    pub duration_ms: u128,
    pub status: Option<u16>,
    pub kind: &'static str,
    pub message: String,
    pub body: Option<ResponseBody>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderRequestEvent {
    pub source: &'static str,
    pub call_id: String,
    pub provider: String,
    pub model: String,
    pub stream: bool,
    pub method: &'static str,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_truncated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_original_bytes: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderResponseEvent {
    pub source: &'static str,
    pub call_id: String,
    pub provider: String,
    pub status: u16,
    pub duration_ms: u128,
    pub headers: BTreeMap<String, String>,
    pub body: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_truncated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_original_bytes: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderStreamStartedEvent {
    pub source: &'static str,
    pub call_id: String,
    pub provider: String,
    pub status: u16,
    pub content_type: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderStreamCompletedEvent {
    pub source: &'static str,
    pub call_id: String,
    pub provider: String,
    pub duration_ms: u128,
    pub bytes_received: usize,
    pub frames_received: usize,
    pub events_decoded: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderErrorEvent {
    pub source: &'static str,
    pub call_id: String,
    pub provider: String,
    pub duration_ms: u128,
    pub status: Option<u16>,
    pub kind: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
}

pub(crate) fn request_event(
    call_id: String,
    provider: String,
    input: RequestEventInput,
) -> LogEvent {
    let snapshot = snapshot_json(input.body);
    LogEvent::Request(ProviderRequestEvent {
        source: "litellm-rust",
        call_id,
        provider,
        model: input.model,
        stream: input.stream,
        method: "POST",
        url: redact_url(&input.url),
        headers: redact_headers(&input.headers),
        body: snapshot.body,
        body_truncated: snapshot.body_truncated,
        body_original_bytes: snapshot.body_original_bytes,
    })
}

pub(crate) fn response_event(input: ResponseEventInput) -> LogEvent {
    let snapshot = input.body.snapshot();
    LogEvent::Response(ProviderResponseEvent {
        source: "litellm-rust",
        call_id: input.call_id,
        provider: input.provider,
        status: input.status,
        duration_ms: input.duration_ms,
        headers: redact_headers(&input.headers),
        body: snapshot.body,
        body_truncated: snapshot.body_truncated,
        body_original_bytes: snapshot.body_original_bytes,
    })
}

pub(crate) fn error_event(input: ErrorEventInput) -> LogEvent {
    LogEvent::Error(ProviderErrorEvent {
        source: "litellm-rust",
        call_id: input.call_id,
        provider: input.provider,
        duration_ms: input.duration_ms,
        status: input.status,
        kind: input.kind,
        message: input.message,
        body: input.body.map(|body| body.snapshot().body),
    })
}

pub(crate) fn stream_started(
    call_id: String,
    provider: String,
    status: u16,
    content_type: Option<String>,
) -> LogEvent {
    LogEvent::StreamStarted(ProviderStreamStartedEvent {
        source: "litellm-rust",
        call_id,
        provider,
        status,
        content_type,
    })
}

pub(crate) fn stream_completed(
    call_id: String,
    provider: String,
    duration_ms: u128,
    bytes_received: usize,
    frames_received: usize,
    events_decoded: usize,
) -> LogEvent {
    LogEvent::StreamCompleted(ProviderStreamCompletedEvent {
        source: "litellm-rust",
        call_id,
        provider,
        duration_ms,
        bytes_received,
        frames_received,
        events_decoded,
    })
}

#[derive(Clone, Debug)]
pub enum ResponseBody {
    Json(Value),
    Binary {
        media_type: Option<String>,
        bytes: usize,
    },
}

impl ResponseBody {
    fn snapshot(self) -> BodySnapshot {
        match self {
            Self::Json(value) => snapshot_json(value),
            Self::Binary { media_type, bytes } => BodySnapshot {
                body: serde_json::json!({"media_type": media_type, "bytes": bytes}),
                body_truncated: None,
                body_original_bytes: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_credentials_recursively() {
        let event = request_event(
            "call_01".to_string(),
            "anthropic".to_string(),
            RequestEventInput {
                model: "claude".to_string(),
                stream: false,
                url: "https://example.test?signature=secret&x=ok".to_string(),
                headers: vec![("Authorization".to_string(), "Bearer secret".to_string())],
                body: serde_json::json!({"nested": {"token": "secret"}, "prompt": "visible"}),
            },
        );
        let json = serde_json::to_string(&event).expect("serializes");
        assert!(!json.contains("secret"));
        assert!(json.contains("visible"));
        assert!(json.contains("[REDACTED]"));
    }

    #[test]
    fn redact_url_preserves_queryless_encoded_paths() {
        assert_eq!(
            redact_url("https://example.test/v1%3A0/invoke"),
            "https://example.test/v1%3A0/invoke"
        );
    }

    #[test]
    fn redact_url_preserves_non_secret_query_params() {
        let redacted = redact_url(
            "https://example.test/invoke?X-Amz-Signature=sig&X-Amz-Credential=cred&foo=bar",
        );
        assert!(redacted.contains("X-Amz-Signature=%5BREDACTED%5D"));
        assert!(redacted.contains("X-Amz-Credential=%5BREDACTED%5D"));
        assert!(redacted.contains("foo=bar"));
    }
}
