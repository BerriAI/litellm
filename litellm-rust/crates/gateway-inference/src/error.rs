use axum::http::StatusCode;
use axum::{
    Json,
    response::{IntoResponse, Response},
};
use litellm_core::RouteError;
use litellm_http::transport::Error as TransportError;
use litellm_llms::base_llm::ocr::error::Error as OcrError;
use serde_json::{Map, Value, json};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid request body: {0}")]
    InvalidBody(String),
    #[error(
        "/v1/messages: Invalid model name passed in model={0}. Call `/v1/models` to view available models for your key."
    )]
    UnknownModel(String),
    #[error(transparent)]
    Route(#[from] RouteError),
    #[error(transparent)]
    Ocr(#[from] OcrError),
    #[error("{0} is not implemented by the Rust gateway")]
    Unsupported(String),
    #[error("request body exceeds the size limit")]
    BodyTooLarge,
    #[error("{0}")]
    Internal(String),
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Unsupported(_)
            | Self::Route(RouteError::Unsupported(_))
            | Self::Ocr(OcrError::Unsupported(_)) => StatusCode::NOT_IMPLEMENTED,
            Self::BodyTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Ocr(
                OcrError::Auth(litellm_auth::Error::MissingApiKey { .. })
                | OcrError::MissingAzureAiCredentials
                | OcrError::MissingAzureDocumentIntelligenceCredentials
                | OcrError::MissingReductoApiKey,
            ) => StatusCode::UNAUTHORIZED,
            Self::Ocr(error) => error
                .http_status_code()
                .and_then(|status| StatusCode::from_u16(status).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Self::InvalidBody(_) | Self::UnknownModel(_) => StatusCode::BAD_REQUEST,
            Self::Route(RouteError::Transport(TransportError::Http { status, .. })) => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
            }
            Self::Route(RouteError::Auth(litellm_auth::Error::MissingApiKey { .. })) => {
                StatusCode::UNAUTHORIZED
            }
            Self::Route(error) if error.is_request() => StatusCode::BAD_REQUEST,
            Self::Route(_) | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn openai_response(self) -> Response {
        let status = self.status();
        let message = match &self {
            Self::UnknownModel(model) => format!("Invalid model name passed in model={model}"),
            _ => self.to_string(),
        };
        (
            status,
            Json(json!({"error": {
                "message": message,
                "type": error_type(status),
                "param": null,
                "code": status.as_u16(),
            }})),
        )
            .into_response()
    }

    /// The Anthropic error envelope Python's `AnthropicExceptionMapping` builds: an upstream
    /// body already in that shape passes through, any other has its message extracted.
    pub fn body(&self, request_id: Option<&str>) -> Value {
        let raw = match self {
            Self::Route(RouteError::Transport(TransportError::Http { body, .. })) => body.clone(),
            other => other.to_string(),
        };
        let parsed = serde_json::from_str::<Value>(&raw).ok();
        let envelope = match parsed {
            Some(Value::Object(object)) if is_anthropic_error(&object) => object,
            Some(Value::Object(object)) => {
                envelope(self.status(), provider_message(&object).unwrap_or(&raw))
            }
            _ => envelope(self.status(), &raw),
        };
        Value::Object(with_request_id(envelope, request_id))
    }

    /// An `event: error` frame, for a stream that fails after its headers went out.
    pub fn sse_frame(&self) -> String {
        format!("event: error\ndata: {}\n\n", self.body(None))
    }
}

fn error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

fn envelope(status: StatusCode, message: &str) -> Map<String, Value> {
    let Value::Object(envelope) = json!({
        "type": "error",
        "error": {"type": error_type(status), "message": message},
    }) else {
        unreachable!("a json object literal is an object")
    };
    envelope
}

fn is_anthropic_error(object: &Map<String, Value>) -> bool {
    object.get("type").and_then(Value::as_str) == Some("error")
        && object
            .get("error")
            .and_then(Value::as_object)
            .is_some_and(|error| error.contains_key("type") && error.contains_key("message"))
}

fn provider_message(object: &Map<String, Value>) -> Option<&str> {
    if let Some(detail) = object.get("detail").and_then(Value::as_object) {
        return detail.get("message").and_then(Value::as_str);
    }
    ["Message", "message"]
        .into_iter()
        .filter_map(|key| object.get(key).and_then(Value::as_str))
        .find(|message| !message.is_empty())
}

fn with_request_id(envelope: Map<String, Value>, request_id: Option<&str>) -> Map<String, Value> {
    match request_id {
        Some(id) if !id.is_empty() && !envelope.contains_key("request_id") => envelope
            .into_iter()
            .chain([("request_id".to_string(), Value::from(id))])
            .collect(),
        _ => envelope,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn upstream(status: u16, body: &str) -> Error {
        Error::Route(RouteError::Transport(TransportError::Http {
            status,
            body: body.into(),
        }))
    }

    #[rstest]
    #[case::anthropic_body_passes_through(
        upstream(529, r#"{"type":"error","error":{"type":"overloaded_error","message":"busy","extra":1}}"#),
        Some("req_1"),
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy", "extra": 1}, "request_id": "req_1"}),
    )]
    #[case::upstream_request_id_wins(
        upstream(400, r#"{"type":"error","error":{"type":"x","message":"m"},"request_id":"upstream"}"#),
        Some("caller"),
        json!({"type": "error", "error": {"type": "x", "message": "m"}, "request_id": "upstream"}),
    )]
    #[case::bedrock_detail(
        upstream(403, r#"{"detail":{"message":"denied"}}"#),
        None,
        json!({"type": "error", "error": {"type": "permission_error", "message": "denied"}}),
    )]
    #[case::aws_message(
        upstream(429, r#"{"Message":"slow down"}"#),
        None,
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}),
    )]
    #[case::plain_text_with_unmapped_status(
        upstream(502, "bad gateway"),
        None,
        json!({"type": "error", "error": {"type": "api_error", "message": "bad gateway"}}),
    )]
    #[case::unknown_model(
        Error::UnknownModel("nope".into()),
        None,
        json!({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "/v1/messages: Invalid model name passed in model=nope. Call `/v1/models` to view available models for your key.",
        }}),
    )]
    fn body_follows_the_anthropic_exception_mapping(
        #[case] error: Error,
        #[case] request_id: Option<&str>,
        #[case] expected: Value,
    ) {
        assert_eq!(error.body(request_id), expected);
    }

    #[rstest]
    #[case::upstream_status(upstream(429, ""), StatusCode::TOO_MANY_REQUESTS)]
    #[case::rejected_request(Error::Route(RouteError::InvalidRequest("top_k".into())), StatusCode::BAD_REQUEST)]
    #[case::missing_key(
        Error::Route(RouteError::Auth(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        })),
        StatusCode::UNAUTHORIZED,
    )]
    #[case::lost_connection(
        Error::Route(RouteError::Transport(TransportError::Network("reset".into()))),
        StatusCode::INTERNAL_SERVER_ERROR,
    )]
    fn status_follows_who_is_at_fault(#[case] error: Error, #[case] status: StatusCode) {
        assert_eq!(error.status(), status);
    }
}
