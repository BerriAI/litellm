use axum::http::StatusCode;
use axum::{
    Json,
    response::{IntoResponse, Response},
};
use litellm_gateway_hooks::Refusal;
use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use litellm_llms::base_llm::ocr::error::Error as OcrError;
use litellm_router::Failure;
use serde_json::{Map, Value, json};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Auth(#[from] litellm_gateway_auth::Error),
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
    /// A hook turned the request away.
    #[error("{}", .0.message)]
    Refused(Refusal),
}

impl From<Refusal> for Error {
    fn from(refusal: Refusal) -> Self {
        Self::Refused(refusal)
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        self.openai_response()
    }
}

impl From<litellm_host_http::Error<RouteError>> for Error {
    fn from(error: litellm_host_http::Error<RouteError>) -> Self {
        match error {
            litellm_host_http::Error::Call(error) => Self::Route(error),
            litellm_host_http::Error::Protocol => Self::Internal(error.to_string()),
        }
    }
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Auth(error) => error.status(),
            Self::Refused(refusal) => {
                StatusCode::from_u16(refusal.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
            }
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
        match self {
            Self::Auth(error) => error.into_response(),
            Self::Refused(refusal) => refused(refusal),
            other => (other.status(), Json(other.openai_body())).into_response(),
        }
    }

    /// The response for a `/v1/messages` request: Anthropic's error envelope, or the proxy's
    /// own shape for an error raised before the provider call.
    pub fn messages_response(self, request_id: Option<&str>) -> Response {
        match self {
            Self::Refused(refusal) => refused(refusal),
            other => (other.status(), Json(other.body(request_id))).into_response(),
        }
    }

    /// An OpenAI error event, for a Chat Completions stream that fails after its headers
    /// went out.
    pub fn openai_sse_frame(&self) -> String {
        format!("data: {}\n\n", self.openai_body())
    }

    fn openai_body(&self) -> Value {
        let status = self.status();
        let message = match self {
            Self::UnknownModel(model) => format!("Invalid model name passed in model={model}"),
            Self::Route(RouteError::Transport(TransportError::Http { body, .. })) => {
                openai_error_message(body).unwrap_or_else(|| self.to_string())
            }
            _ => self.to_string(),
        };
        let mut error = json!({
            "message": message,
            "type": error_type(status),
            "param": null,
            "code": status.as_u16(),
        });
        if let Self::Route(RouteError::Transport(TransportError::Http { body, .. })) = self
            && let Some(fields) = azure_content_filter_fields(body)
        {
            error["provider_specific_fields"] = fields;
        }
        json!({ "error": error })
    }

    /// The Anthropic error envelope Python's `AnthropicExceptionMapping` builds: an upstream
    /// body already in that shape passes through, any other has its message extracted.
    pub fn body(&self, request_id: Option<&str>) -> Value {
        // Auth runs before the provider call, where the proxy answers with its own error shape.
        if let Self::Auth(error) = self {
            return error.body();
        }
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

    /// How the router counts this error against the deployment that produced it, if at all.
    pub fn failure(&self) -> Option<Failure> {
        match self {
            Self::Route(RouteError::Transport(TransportError::Http { status, .. })) => match status
            {
                429 => Some(Failure::RateLimited),
                401 | 402 | 404 => Some(Failure::Rejected),
                408 | 500.. => Some(Failure::Unavailable),
                _ => None,
            },
            Self::Route(RouteError::Transport(
                TransportError::Network(_) | TransportError::Connect(_),
            )) => Some(Failure::Unavailable),
            _ => None,
        }
    }

    /// An `event: error` frame, for a stream that fails after its headers went out.
    pub fn sse_frame(&self) -> String {
        format!("event: error\ndata: {}\n\n", self.body(None))
    }
}

/// A hook's refusal, in the proxy's error shape and with the hook's headers.
fn refused(refusal: Refusal) -> Response {
    let status = StatusCode::from_u16(refusal.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = match refusal.anthropic {
        true => json!({"type": "error", "error": {
            "type": refusal.kind,
            "message": refusal.message,
        }}),
        false => json!({"error": {
            "message": refusal.message,
            "type": refusal.kind,
            "param": refusal.param,
            "code": status.as_u16().to_string(),
        }}),
    };
    let mut response = (status, Json(body)).into_response();
    response.headers_mut().extend(refusal.headers);
    response
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

fn openai_error_message(body: &str) -> Option<String> {
    let body: Value = serde_json::from_str(body).ok()?;
    body.get("error")?
        .get("message")?
        .as_str()
        .map(str::to_owned)
}

/// Python's `AzureOpenAIExceptionMapping` keeps Azure's error and its content filter results
/// on a content policy violation. Only Azure nests an `inner_error`, which tells its errors
/// apart from another provider's.
fn azure_content_filter_fields(body: &str) -> Option<Value> {
    const POLICY_PHRASES: [&str; 7] = [
        "content_policy_violation",
        "responsibleaipolicyviolation",
        "the response was filtered due to the prompt triggering azure openai's content management",
        "your task failed as a result of our safety system",
        "the model produced invalid content",
        "content_filter_policy",
        "your request was rejected as a result of our safety system",
    ];
    let parsed: Value = serde_json::from_str(body).ok()?;
    let azure_error = parsed
        .get("error")
        .filter(|error| error.is_object())
        .unwrap_or(&parsed);
    let inner_error = ["inner_error", "innererror"].into_iter().find_map(|key| {
        azure_error
            .get(key)
            .or_else(|| parsed.get(key))
            .filter(|inner| !inner.is_null())
    })?;
    let lowered = body.to_lowercase();
    let violation = azure_error.get("code").and_then(Value::as_str)
        == Some("content_policy_violation")
        || inner_error.get("code").and_then(Value::as_str) == Some("ResponsibleAIPolicyViolation")
        || POLICY_PHRASES.iter().any(|phrase| lowered.contains(phrase));
    violation.then(|| {
        json!({
            "innererror": inner_error,
            "inner_error": inner_error,
            "azure_error": azure_error,
        })
    })
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
    #[case::openai_body(
        upstream(429, r#"{"error":{"message":"slow down","type":"requests","code":"rate_limit"}}"#),
        json!({"error": {"message": "slow down", "type": "rate_limit_error", "param": null, "code": 429}}),
    )]
    #[case::plain_text(
        upstream(502, "bad gateway"),
        json!({"error": {
            "message": "upstream request failed with status 502: bad gateway",
            "type": "api_error", "param": null, "code": 502,
        }}),
    )]
    fn openai_errors_carry_the_provider_message(#[case] error: Error, #[case] expected: Value) {
        let frame = error.openai_sse_frame();
        assert_eq!(error.openai_body(), expected);
        assert_eq!(frame, format!("data: {expected}\n\n"));
    }

    #[rstest]
    #[case::content_filter(
        r#"{"error":{"message":"The response was filtered due to the prompt triggering Azure OpenAI's content management policy.","type":null,"param":"prompt","code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"hate":{"filtered":true,"severity":"high"}}}}}"#,
        true,
    )]
    #[case::inner_error_spelling(
        r#"{"error":{"message":"blocked","code":"content_policy_violation","inner_error":{"code":"ResponsibleAIPolicyViolation"}}}"#,
        true,
    )]
    #[case::other_azure_error(
        r#"{"error":{"message":"bad value","code":"invalid_value","innererror":{"code":"Other"}}}"#,
        false
    )]
    #[case::openai_policy_error(
        r#"{"error":{"message":"rejected","code":"content_policy_violation","type":"invalid_request_error"}}"#,
        false,
    )]
    fn azure_content_filter_errors_keep_the_azure_error(#[case] body: &str, #[case] kept: bool) {
        let parsed: Value = serde_json::from_str(body).unwrap();
        let error = upstream(400, body).openai_body();
        let fields = &error["error"]["provider_specific_fields"];
        assert_eq!(!fields.is_null(), kept);
        if kept {
            assert_eq!(fields["azure_error"], parsed["error"]);
            assert_eq!(fields["innererror"], fields["inner_error"]);
            assert_eq!(fields["innererror"]["code"], "ResponsibleAIPolicyViolation");
        }
        assert_eq!(error["error"]["message"], parsed["error"]["message"]);
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

    #[rstest]
    #[case::rate_limited(upstream(429, ""), Some(Failure::RateLimited))]
    #[case::bad_key(upstream(401, ""), Some(Failure::Rejected))]
    #[case::unknown_upstream_model(upstream(404, ""), Some(Failure::Rejected))]
    #[case::request_timeout(upstream(408, ""), Some(Failure::Unavailable))]
    #[case::server_error(upstream(500, ""), Some(Failure::Unavailable))]
    #[case::overloaded(upstream(529, ""), Some(Failure::Unavailable))]
    #[case::bad_request(upstream(400, ""), None)]
    #[case::too_large(upstream(413, ""), None)]
    #[case::refused(
        Error::Route(RouteError::Transport(TransportError::Connect("refused".into()))),
        Some(Failure::Unavailable)
    )]
    #[case::reset(
        Error::Route(RouteError::Transport(TransportError::Network("reset".into()))),
        Some(Failure::Unavailable)
    )]
    #[case::caller_mistake(Error::InvalidBody("x".into()), None)]
    #[case::rejected_request(Error::Route(RouteError::InvalidRequest("top_k".into())), None)]
    fn failure_marks_what_another_deployment_could_serve(
        #[case] error: Error,
        #[case] failure: Option<Failure>,
    ) {
        assert_eq!(error.failure(), failure);
    }
}
