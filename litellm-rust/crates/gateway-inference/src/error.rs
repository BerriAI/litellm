use axum::http::StatusCode;
use axum::{
    Json,
    response::{IntoResponse, Response},
};
use litellm_inference::Error as InferenceError;
use litellm_llms::base_llm::ocr::error::Error as OcrError;
use serde_json::{Map, Value, json};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Auth(#[from] litellm_gateway_auth::Error),
    #[error("invalid request body")]
    InvalidBody(#[from] BodyError),
    #[error("invalid path parameter")]
    InvalidPath(#[source] axum::extract::rejection::PathRejection),
    #[error(
        "{route}: Invalid model name passed in model={model}. Call `/v1/models` to view available models for your key."
    )]
    UnknownModel { route: &'static str, model: String },
    #[error(transparent)]
    Route(#[from] InferenceError),
    #[error(transparent)]
    Ocr(#[from] OcrError),
    #[error("{0} is not implemented by the Rust gateway")]
    Unsupported(String),
    #[error("request body exceeds the size limit")]
    BodyTooLarge,
    #[error("unexpected HTTP host operation")]
    Internal,
}

#[derive(Debug, thiserror::Error)]
pub enum BodyError {
    #[error("could not read the body")]
    Read(#[source] axum::extract::rejection::BytesRejection),
    #[error("invalid JSON")]
    Json(#[source] serde_json::Error),
    #[error("expected a JSON object")]
    NotAnObject,
    #[error("invalid multipart body")]
    Multipart(#[source] axum::extract::multipart::MultipartRejection),
    #[error("invalid multipart field")]
    MultipartField(#[source] axum::extract::multipart::MultipartError),
    #[error("model is required")]
    MissingModel,
    #[error("invalid cache controls")]
    CacheControls(#[source] serde_json::Error),
    #[error("invalid caching flag")]
    Caching(#[source] serde_json::Error),
    #[error("cache durations must be finite and positive")]
    CacheDuration(#[source] Option<std::time::TryFromFloatSecsError>),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        self.openai_response()
    }
}

impl From<litellm_host_http::Error<InferenceError>> for Error {
    fn from(error: litellm_host_http::Error<InferenceError>) -> Self {
        match error {
            litellm_host_http::Error::Call(error) => Self::Route(error),
            litellm_host_http::Error::Protocol => Self::Internal,
        }
    }
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Auth(error) => error.status(),
            Self::InvalidPath(rejection) => rejection.status(),
            Self::Unsupported(_) => StatusCode::NOT_IMPLEMENTED,
            Self::BodyTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::InvalidBody(_) | Self::UnknownModel { .. } => StatusCode::BAD_REQUEST,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Ocr(error) => ocr_status(error),
            Self::Route(error) => route_status(error),
        }
    }

    pub fn openai_response(self) -> Response {
        (self.status(), Json(self.openai_body())).into_response()
    }

    /// The OpenAI envelope as an `event: error` frame, for a stream that fails after its
    /// headers went out.
    pub fn responses_sse_frame(&self) -> String {
        let event = json!({
            "type": "error",
            "code": self.status().as_u16().to_string(),
            "message": self.message(),
            "param": null,
        });
        format!("event: error\ndata: {event}\n\n")
    }

    fn openai_body(&self) -> Value {
        let status = self.status();
        json!({"error": {
            "message": self.message(),
            "type": error_type(status),
            "param": null,
            "code": status.as_u16(),
        }})
    }

    /// The Anthropic error envelope Python's `AnthropicExceptionMapping` builds: an upstream
    /// body already in that shape passes through, any other has its message extracted.
    pub fn body(&self, request_id: Option<&str>) -> Value {
        let raw = match self {
            Self::Route(error) if let Some((_, body)) = error.upstream_response() => {
                body.to_owned()
            }
            other => other.message(),
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

    fn message(&self) -> String {
        match self {
            Self::Auth(error) => error.to_string(),
            Self::InvalidBody(_)
            | Self::InvalidPath(_)
            | Self::UnknownModel { .. }
            | Self::Route(_)
            | Self::Ocr(_)
            | Self::Unsupported(_)
            | Self::BodyTooLarge
            | Self::Internal => litellm_error::Report::new(self).to_string(),
        }
    }
}

fn route_status(error: &InferenceError) -> StatusCode {
    if error.is_unsupported() {
        return StatusCode::NOT_IMPLEMENTED;
    }
    if let Some((status, _)) = error.upstream_response() {
        return StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    }
    if error.is_missing_credential() {
        return StatusCode::UNAUTHORIZED;
    }
    if error.is_request() {
        return StatusCode::BAD_REQUEST;
    }
    StatusCode::INTERNAL_SERVER_ERROR
}

fn ocr_status(error: &OcrError) -> StatusCode {
    if error.is_unsupported() {
        return StatusCode::NOT_IMPLEMENTED;
    }
    if let Some((status, ..)) = error.upstream_response() {
        return StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    }
    if error.is_missing_credential() {
        return StatusCode::UNAUTHORIZED;
    }
    error
        .http_status_code()
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
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
    Map::from_iter([
        ("type".to_owned(), Value::from("error")),
        (
            "error".to_owned(),
            json!({"type": error_type(status), "message": message}),
        ),
    ])
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
        Error::Route(litellm_http::Error::upstream(status, body.into()).into())
    }

    fn invalid_json() -> Error {
        BodyError::Json(serde_json::from_str::<Value>("{").unwrap_err()).into()
    }

    fn unknown_model() -> Error {
        Error::UnknownModel {
            route: "/v1/messages",
            model: "nope".into(),
        }
    }

    #[rstest]
    #[case::anthropic_body(
        upstream(529, r#"{"type":"error","error":{"type":"overloaded_error","message":"busy","extra":1}}"#),
        Some("req_1"),
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy", "extra": 1}, "request_id": "req_1"}),
    )]
    #[case::upstream_request_id_wins(
        upstream(400, r#"{"type":"error","error":{"type":"x","message":"m"},"request_id":"upstream"}"#),
        Some("caller"),
        json!({"type": "error", "error": {"type": "x", "message": "m"}, "request_id": "upstream"}),
    )]
    fn an_anthropic_provider_body_passes_through(
        #[case] error: Error,
        #[case] request_id: Option<&str>,
        #[case] expected: Value,
    ) {
        assert_eq!(error.body(request_id), expected);
    }

    #[rstest]
    #[case::ascii("busy ".repeat(80))]
    #[case::unicode("é".repeat(300))]
    fn long_provider_errors_preserve_the_messages_envelope(#[case] message: String) {
        let body = json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": message},
            "request_id": "provider-request",
        });
        let error = upstream(529, &body.to_string());
        let report = litellm_error::Report::new(&error).to_string();
        let diagnostic_body: String = body.to_string().chars().take(256).collect();
        assert_eq!(report.matches(&diagnostic_body).count(), 1, "{report}");
        assert!(report.ends_with("... (truncated)"), "{report}");
        assert!(!report.contains(&message), "{report}");
        assert!(!format!("{error:?}").contains(&message));
        assert_eq!(error.status(), StatusCode::from_u16(529).unwrap());
        assert_eq!(error.body(Some("gateway-request")), body);
    }

    #[rstest]
    #[case::bedrock_detail(
        upstream(403, r#"{"detail":{"message":"denied"}}"#),
        "permission_error",
        "denied"
    )]
    #[case::aws_message(
        upstream(429, r#"{"Message":"slow down"}"#),
        "rate_limit_error",
        "slow down"
    )]
    #[case::plain_text_with_unmapped_status(
        upstream(502, "bad gateway"),
        "api_error",
        "bad gateway"
    )]
    #[case::unknown_model(
        unknown_model(),
        "invalid_request_error",
        "/v1/messages: Invalid model name passed in model=nope. Call `/v1/models` to view available models for your key."
    )]
    fn other_failures_get_the_anthropic_envelope_with_their_message(
        #[case] error: Error,
        #[case] error_type: &str,
        #[case] message: &str,
    ) {
        let body = error.body(Some("req_1"));
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], error_type);
        assert_eq!(body["error"]["message"], message);
        assert_eq!(body["request_id"], "req_1");
    }

    #[rstest]
    #[case::upstream_status(upstream(429, ""), StatusCode::TOO_MANY_REQUESTS)]
    #[case::rejected_request(
        Error::Route(InferenceError::invalid_request(litellm_llms::ErrorDetail::Expected {
            subject: "top_k",
            expected: "an integer",
        })),
        StatusCode::BAD_REQUEST
    )]
    #[case::missing_key(
        Error::Route(InferenceError::Auth(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        })),
        StatusCode::UNAUTHORIZED,
    )]
    #[case::lost_connection(
        Error::Route(
            litellm_http::Error::from(tokio_tungstenite::tungstenite::Error::ConnectionClosed)
                .into()
        ),
        StatusCode::INTERNAL_SERVER_ERROR,
    )]
    #[case::invalid_body(invalid_json(), StatusCode::BAD_REQUEST)]
    #[case::missing_model(BodyError::MissingModel.into(), StatusCode::BAD_REQUEST)]
    #[case::unknown_model(unknown_model(), StatusCode::BAD_REQUEST)]
    #[case::oversized_body(Error::BodyTooLarge, StatusCode::PAYLOAD_TOO_LARGE)]
    #[case::unserved_path(Error::Unsupported("/v1/completions".into()), StatusCode::NOT_IMPLEMENTED)]
    #[case::host_protocol(
        litellm_host_http::Error::<InferenceError>::Protocol.into(),
        StatusCode::INTERNAL_SERVER_ERROR
    )]
    #[case::ocr_provider_answer(
        Error::Ocr(OcrError::Provider { status: 429, body: "slow".into(), headers: Default::default() }),
        StatusCode::TOO_MANY_REQUESTS
    )]
    #[case::ocr_missing_credential(
        Error::Ocr(OcrError::MissingCredential { credential: "Azure AI credentials", guidance: "set a key" }),
        StatusCode::UNAUTHORIZED
    )]
    #[case::ocr_request_fault(Error::Ocr(OcrError::MissingDocumentUrl), StatusCode::BAD_REQUEST)]
    #[case::ocr_response_fault(
        Error::Ocr(OcrError::EmptyContent),
        StatusCode::INTERNAL_SERVER_ERROR
    )]
    fn status_follows_who_is_at_fault(#[case] error: Error, #[case] status: StatusCode) {
        assert_eq!(error.status(), status);
    }

    #[test]
    fn a_body_fault_names_the_body_and_keeps_its_cause_once() {
        let cause = serde_json::from_str::<Value>("{").unwrap_err().to_string();
        let body = invalid_json().openai_body();
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            message.starts_with("invalid request body: invalid JSON: "),
            "{message}"
        );
        assert_eq!(message.matches(&cause).count(), 1, "{message}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], 400);
    }

    fn auth_outage() -> Error {
        Error::Auth(litellm_gateway_auth::Error::Unavailable(
            litellm_error::Source::new(std::io::Error::other("private-backend-detail")),
        ))
    }

    #[rstest]
    #[case::openai(auth_outage().openai_body()["error"]["message"].clone())]
    #[case::anthropic(auth_outage().body(None)["error"]["message"].clone())]
    #[case::responses_sse(Value::from(auth_outage().responses_sse_frame()))]
    #[case::anthropic_sse(Value::from(auth_outage().sse_frame()))]
    fn an_auth_failure_hides_its_backend_cause(#[case] rendered: Value) {
        let top = auth_outage().to_string();
        let rendered = rendered.to_string();
        assert!(rendered.contains(&top), "{rendered}");
        assert!(!rendered.contains("private-backend-detail"), "{rendered}");
    }

    #[rstest]
    #[case::provider_answer(upstream(429, "slow down"))]
    #[case::caller_fault(invalid_json())]
    #[case::host_bug(litellm_host_http::Error::<InferenceError>::Protocol.into())]
    fn the_responses_sse_frame_is_the_responses_error_event(#[case] error: Error) {
        let frame = error.responses_sse_frame();
        let data = frame
            .strip_prefix("event: error\ndata: ")
            .and_then(|data| data.strip_suffix("\n\n"))
            .unwrap_or_else(|| panic!("an event: error frame, got {frame:?}"));
        let event = serde_json::from_str::<Value>(data).unwrap();
        assert_eq!(event["type"], "error");
        assert_eq!(event["code"], error.status().as_u16().to_string());
        assert_eq!(event["message"], error.openai_body()["error"]["message"]);
    }
}
