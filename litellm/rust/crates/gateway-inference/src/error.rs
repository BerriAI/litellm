use axum::http::{HeaderName, HeaderValue, StatusCode, header::RETRY_AFTER};
use axum::{
    Extension, Json,
    response::{AppendHeaders, IntoResponse, Response},
};
use litellm_gateway_auth_core::AuthzDenial;
use litellm_inference::Error as InferenceError;
use litellm_llms::base_llm::ocr::error::Error as OcrError;
use litellm_llms_types::formats::messages::MessagesErrorType;
use litellm_python_compat::{Value as PyValue, repr::repr};
use litellm_router_core::FailureKind;
use serde_json::{Map, Value, json};

use crate::{
    headers::Routing,
    hooks::{HookError, Refusal},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The auth core found a model list of the caller's that does not reach the model.
    #[error("operation is not permitted")]
    ModelAccess(#[from] AuthzDenial),
    #[error("invalid request body")]
    InvalidBody(#[from] BodyError),
    #[error(
        "{route}: Invalid model name passed in model={model}. Call `/v1/models` to view available models for your key."
    )]
    UnknownModel { route: &'static str, model: String },
    #[error(transparent)]
    Route(#[from] InferenceError),
    #[error(transparent)]
    Ocr(#[from] OcrError),
    #[error(transparent)]
    NoDeployments(#[from] NoDeployments),
    #[error(transparent)]
    Routing(litellm_router_core::Error),
    #[error("{}", .0.message)]
    Refused(Refusal),
    #[error(transparent)]
    Callback(litellm_error::Source),
    #[error("request body exceeds the size limit")]
    BodyTooLarge,
    /// The request body check refused the body, as Python's authentication refuses it.
    #[error("Authentication Error, {0}")]
    BodyRejected(litellm_gateway_auth_core::body::Rejection),
    /// The host could not read a stored credential the body names, so the body is not checked.
    #[error("could not read the stored credentials the request body names")]
    Credentials(#[source] litellm_gateway_auth_core::Error),
    #[error("unexpected HTTP host operation")]
    Internal,
}

/// Python's `RouterRateLimitError`, in its words: no deployment of `model` takes a call now.
#[derive(Debug, PartialEq)]
pub struct NoDeployments {
    pub model: String,
    /// Seconds until a deployment of the model is tried again.
    pub cooldown: f64,
    /// Every deployment of the model is cooling down.
    pub all_cooling: bool,
    pub pre_call_checks: bool,
    /// The ids of every deployment that is cooling down, of any model.
    pub cooling: Vec<String>,
}

impl std::error::Error for NoDeployments {}

impl std::fmt::Display for NoDeployments {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let whole = self.cooldown.fract() == 0.0 && self.cooldown.abs() < i64::MAX as f64;
        let seconds = match whole {
            true => PyValue::Int((self.cooldown as i64).into()),
            false => PyValue::Float(self.cooldown),
        };
        let reason = match self.all_cooling {
            true => " All deployments for selected model are in cooldown.",
            false => "",
        };
        let cooling = PyValue::List(self.cooling.iter().cloned().map(PyValue::Str).collect());
        write!(
            formatter,
            "No deployments available for selected model, Try again in {} seconds.{reason} Passed model={}. pre-call-checks={}, cooldown_list={}",
            repr(&seconds),
            self.model,
            repr(&PyValue::Bool(self.pre_call_checks)),
            repr(&cooling),
        )
    }
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

/// The error a request ended on, with what the router did for the call it ended: the deployment
/// the router says raised it, or everything it did for a stream the provider failed before any
/// content and no other attempt answered for.
#[derive(Debug)]
pub struct Ended {
    pub error: Box<Error>,
    pub routing: Option<Routing>,
}

impl Ended {
    /// What the routing headers layer reads off the response.
    pub(crate) fn routing(&self) -> Option<Extension<Routing>> {
        self.routing.clone().map(Extension)
    }
}

impl<E: Into<Error>> From<E> for Ended {
    fn from(error: E) -> Self {
        Self {
            error: Box::new(error.into()),
            routing: None,
        }
    }
}

/// The envelope a route answers an error in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorFormat {
    Anthropic,
    OpenAi,
}

impl Ended {
    pub(crate) fn response(self, format: ErrorFormat, request_id: Option<&str>) -> Response {
        match format {
            ErrorFormat::OpenAi => (self.routing(), *self.error).into_response(),
            ErrorFormat::Anthropic => (
                self.error.status(),
                self.routing(),
                AppendHeaders(self.error.retry_after()),
                Json(self.error.body(request_id)),
            )
                .into_response(),
        }
    }
}

impl From<HookError> for Error {
    fn from(ended: HookError) -> Self {
        match ended {
            HookError::Refused(refusal) => Self::Refused(refusal),
            HookError::Failed(failure) => Self::Callback(failure),
        }
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
            Self::ModelAccess(_) => StatusCode::FORBIDDEN,
            Self::BodyTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::InvalidBody(_) | Self::UnknownModel { .. } => StatusCode::BAD_REQUEST,
            Self::NoDeployments(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::BodyRejected(_) => StatusCode::UNAUTHORIZED,
            Self::Internal | Self::Callback(_) | Self::Credentials(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            Self::Ocr(error) => ocr_status(error),
            Self::Route(error) => route_status(error),
            Self::Routing(error) => routing_status(error),
            Self::Refused(refusal) => refusal.status,
        }
    }

    pub fn openai_response(self) -> Response {
        let body = Json(self.openai_body());
        match self {
            Self::Refused(refusal) => (
                refusal.status,
                AppendHeaders(crate::rate_limited::header_pairs(refusal.headers)),
                body,
            )
                .into_response(),
            other => (other.status(), AppendHeaders(other.retry_after()), body).into_response(),
        }
    }

    /// `retry-after` in whole seconds on a refusal the router made itself, as it refuses a
    /// deployment past its declared rate limits.
    pub(crate) fn retry_after(&self) -> Option<(HeaderName, HeaderValue)> {
        let Self::Routing(error) = self else {
            return None;
        };
        let after = error.refused()?.resets_in;
        let seconds = after.as_secs() + u64::from(after.subsec_nanos() > 0);
        Some((RETRY_AFTER, HeaderValue::from(seconds)))
    }

    /// A refusal names its own `type` and `code` when the proxy's exception does.
    fn openai_body(&self) -> Value {
        let status = self.status();
        let (kind, code) = match self {
            Self::Refused(refusal) => (refusal.kind.clone(), refusal.code.clone()),
            _ => (None, None),
        };
        json!({"error": {
            "message": self.message(),
            "type": kind.map_or_else(
                || json!(MessagesErrorType::for_status(status.as_u16())),
                Value::from,
            ),
            "param": null,
            "code": code.map_or_else(|| json!(status.as_u16()), Value::from),
        }})
    }

    /// The Anthropic error envelope Python's `AnthropicExceptionMapping` builds: an upstream
    /// body already in that shape passes through, any other has its message extracted.
    pub fn body(&self, request_id: Option<&str>) -> Value {
        if let Self::Refused(refusal) = self
            && refusal.at_authentication
        {
            return Value::Object(with_request_id(proxy_envelope(refusal), request_id));
        }
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
            Self::Refused(refusal) => refusal.message.clone(),
            Self::ModelAccess(_) | Self::BodyRejected(_) => self.to_string(),
            Self::Routing(error) if let Some(refused) = rate_limit_refusal(error) => refused,
            Self::InvalidBody(_)
            | Self::UnknownModel { .. }
            | Self::Route(_)
            | Self::Ocr(_)
            | Self::NoDeployments(_)
            | Self::Routing(_)
            | Self::Callback(_)
            | Self::BodyTooLarge
            | Self::Credentials(_)
            | Self::Internal => litellm_error::Report::new(self).to_string(),
        }
    }
}

/// The message Python's proxy answers a pre-call check's `RateLimitError` with.
fn rate_limit_refusal(error: &litellm_router_core::Error) -> Option<String> {
    error.refused()?;
    let failure = error.attempt()?;
    let detail = failure.detail.as_deref()?;
    (failure.kind == FailureKind::RateLimit).then(|| format!("litellm.{}: {detail}", failure.kind))
}

fn route_status(error: &InferenceError) -> StatusCode {
    if error.declined().is_some() {
        return StatusCode::NOT_IMPLEMENTED;
    }
    if let Some((status, _)) = error.upstream_response() {
        return StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    }
    if error.timeout().is_some() {
        return StatusCode::REQUEST_TIMEOUT;
    }
    if error.is_missing_credential() {
        return StatusCode::UNAUTHORIZED;
    }
    if error.is_request() {
        return StatusCode::BAD_REQUEST;
    }
    StatusCode::INTERNAL_SERVER_ERROR
}

/// The status of an error the router made itself: the status of the exception it stands for.
fn routing_status(error: &litellm_router_core::Error) -> StatusCode {
    let status = match error {
        litellm_router_core::Error::Attempt { failure, .. } => failure.status,
        other => other.pick_failure().and_then(|failure| failure.status),
    };
    status
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

fn ocr_status(error: &OcrError) -> StatusCode {
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

/// `anthropic_error_content` for an exception the proxy raises while it authenticates: its type
/// by the proxy's own status map, and its `litellm_error_type` when it names one neither map
/// gives the status.
fn proxy_envelope(refusal: &Refusal) -> Map<String, Value> {
    let status = refusal.status.as_u16();
    let kind = proxy_error_type(status);
    let litellm_error_type = refusal
        .kind
        .as_deref()
        .filter(|named| *named != kind && *named != openai_error_type(status))
        .map(|named| ("litellm_error_type".to_owned(), Value::from(named)));
    let error = [
        ("type".to_owned(), Value::from(kind)),
        ("message".to_owned(), Value::from(refusal.message.as_str())),
    ]
    .into_iter()
    .chain(litellm_error_type)
    .collect();
    Map::from_iter([
        ("type".to_owned(), Value::from("error")),
        ("error".to_owned(), Value::Object(error)),
    ])
}

/// `proxy_error_type`: `ANTHROPIC_ERROR_TYPE_MAP` and the proxy's own additions, then any other
/// client error as an invalid request.
fn proxy_error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        402 => "billing_error",
        403 => "permission_error",
        404 => "not_found_error",
        409 => "conflict_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        504 => "timeout_error",
        529 => "overloaded_error",
        _ if status != 408 && (400..500).contains(&status) => "invalid_request_error",
        _ => "api_error",
    }
}

/// `openai_error_type_for_status`.
fn openai_error_type(status: u16) -> &'static str {
    match status {
        401 => "authentication_error",
        403 => "permission_error",
        429 => "rate_limit_error",
        ..500 => "invalid_request_error",
        _ => "internal_server_error",
    }
}

fn envelope(status: StatusCode, message: &str) -> Map<String, Value> {
    Map::from_iter([
        ("type".to_owned(), Value::from("error")),
        (
            "error".to_owned(),
            json!({"type": MessagesErrorType::for_status(status.as_u16()), "message": message}),
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

    const BUDGET: &str = "Budget has been exceeded! Key=hash Current cost: 0.0, Max budget: 1e-06";

    fn budget_refusal(at_authentication: bool) -> Error {
        Error::Refused(Refusal {
            kind: Some("budget_exceeded".to_owned()),
            code: Some("422".to_owned()),
            at_authentication,
            ..Refusal::new(StatusCode::UNPROCESSABLE_ENTITY, BUDGET)
        })
    }

    #[rstest]
    #[case::at_authentication(
        true,
        json!({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": BUDGET,
            "litellm_error_type": "budget_exceeded",
        }})
    )]
    #[case::in_the_handler(
        false,
        json!({"type": "error", "error": {"type": "api_error", "message": BUDGET}})
    )]
    fn a_refusal_raised_while_authenticating_is_rendered_by_the_proxys_exception_handler(
        #[case] at_authentication: bool,
        #[case] expected: Value,
    ) {
        assert_eq!(budget_refusal(at_authentication).body(None), expected);
    }

    #[rstest]
    #[case::billing(402, None, "billing_error", None)]
    #[case::conflict(409, None, "conflict_error", None)]
    #[case::timeout(504, None, "timeout_error", None)]
    #[case::request_timeout(408, None, "api_error", None)]
    #[case::unprocessable(422, None, "invalid_request_error", None)]
    #[case::server(503, None, "api_error", None)]
    #[case::a_kind_the_status_gives(
        401,
        Some("authentication_error"),
        "authentication_error",
        None
    )]
    #[case::an_openai_kind_the_status_gives(
        422,
        Some("invalid_request_error"),
        "invalid_request_error",
        None
    )]
    #[case::its_own_kind(401, Some("auth_error"), "authentication_error", Some("auth_error"))]
    fn an_authentication_refusal_takes_the_proxys_type_and_names_its_own(
        #[case] status: u16,
        #[case] kind: Option<&str>,
        #[case] expected_type: &str,
        #[case] litellm_error_type: Option<&str>,
    ) {
        let refused = Error::Refused(Refusal {
            kind: kind.map(str::to_owned),
            at_authentication: true,
            ..Refusal::new(StatusCode::from_u16(status).unwrap(), "no")
        });

        let body = refused.body(None);

        assert_eq!(
            (
                body["error"]["type"].as_str(),
                body["error"]
                    .get("litellm_error_type")
                    .and_then(Value::as_str)
            ),
            (Some(expected_type), litellm_error_type)
        );
    }

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
    #[case::every_deployment_cooling(
        NoDeployments {
            model: "primary".into(),
            cooldown: 30.0,
            all_cooling: true,
            pre_call_checks: false,
            cooling: vec!["primary-1".into(), "primary-2".into()],
        },
        "No deployments available for selected model, Try again in 30 seconds. All deployments for selected model are in cooldown. Passed model=primary. pre-call-checks=False, cooldown_list=['primary-1', 'primary-2']"
    )]
    #[case::another_models_deployment_cooling(
        NoDeployments {
            model: "primary".into(),
            cooldown: 2.5,
            all_cooling: false,
            pre_call_checks: true,
            cooling: vec!["other-1".into()],
        },
        "No deployments available for selected model, Try again in 2.5 seconds. Passed model=primary. pre-call-checks=True, cooldown_list=['other-1']"
    )]
    #[case::nothing_cooling(
        NoDeployments {
            model: "primary".into(),
            cooldown: 5.0,
            all_cooling: false,
            pre_call_checks: false,
            cooling: vec![],
        },
        "No deployments available for selected model, Try again in 5 seconds. Passed model=primary. pre-call-checks=False, cooldown_list=[]"
    )]
    fn no_deployments_reads_as_pythons_router_rate_limit_error(
        #[case] refused: NoDeployments,
        #[case] message: &str,
    ) {
        let error = Error::from(refused);

        assert_eq!(error.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            error.body(None),
            json!({"type": "error", "error": {"type": "rate_limit_error", "message": message}})
        );
    }

    #[rstest]
    #[case::upstream_status(upstream(429, ""), StatusCode::TOO_MANY_REQUESTS)]
    #[case::declined(
        Error::Route(InferenceError::Declined(litellm_core_utils::decline::Decline::new(
            "unrecognized request parameter"
        ))),
        StatusCode::NOT_IMPLEMENTED
    )]
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
}
