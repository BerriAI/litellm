//! How a call failed, as Python's logging reads the exception the same call raises there
//! (`StandardLoggingPayloadSetup.get_error_information` and
//! `get_error_information_for_logging_payload` in `litellm/litellm_core_utils/litellm_logging.py`).
//! Each gateway [`Error`] is read as the exception Python raises for the same failure: a
//! provider's answer as `exception_type` maps it, a limiter's refusal as its
//! `ProxyRateLimitError`, a router refusal as its `RouterRateLimitError`, and the proxy's own
//! refusals as the `ProxyException` or `HTTPException` it raises.
//!
//! The message, the traceback and `error_str` are held as raised. Python redacts them only as it
//! builds the payload (`redact_string`), and so does [`CallFailure::error_information`].

use std::time::Duration;

use litellm_core_utils::exception_mapping_utils::{
    ExceptionContext, OriginalException, PublicError, TimeoutPhase, classify_status, exception_type,
};
use litellm_gateway_auth_core::{AuthzDenial, DeniedList};
use litellm_inference::Error as InferenceError;
use litellm_llms::base_llm::ocr::error::Error as OcrError;
use litellm_python_compat::{Value as PyValue, repr::repr};
use litellm_router_core::{AttemptError, FailureKind};
use litellm_standard_logging::{
    CLIENT_DISCONNECTED_MESSAGE, ErrorInformation, ExceptionClass, LoggedException,
    RateLimitCategory, RateLimitType, provider_request_id,
};

use crate::{Error, hooks::Refusal};

/// The `type` of Python's `ProxyRateLimitError`, which a limiter's refusal carries.
const THROTTLING: &str = "throttling_error";
/// The response header `parallel_request_limiter_v3` names the refusing limit's type in.
const RATE_LIMIT_TYPE_HEADER: &str = "rate_limit_type";
/// `PROXY_LLM_PROVIDER_FALLBACK`, the provider a limiter's refusal names when the call's model
/// resolves to none.
const PROXY_PROVIDER: &str = "litellm_proxy";

/// How a call that did not end as served failed, which the call's failure payload logs.
#[derive(Clone, Debug, PartialEq)]
pub enum CallFailure {
    /// The call ended on an error, read as the exception Python raises for it.
    Raised {
        exception: LoggedException,
        /// `str()` of the exception, which Python's failure handler passes as `error_str`.
        error_str: String,
    },
    /// The client left before the call ended, which Python logs in place of any exception
    /// (`_CLIENT_DISCONNECTED_ERROR_INFORMATION`, 499).
    ClientDisconnected,
}

impl CallFailure {
    /// The exception Python raises for a call that ended on `error`. `provider` is the provider
    /// of the deployment whose attempt raised it, else the one the call's model resolves to;
    /// `timeout` is that attempt's timeout.
    pub(crate) fn raised(error: &Error, provider: Option<&str>, timeout: Option<Duration>) -> Self {
        let traceback = litellm_error::Report::new(error).to_string();
        let raised = Raised::of(error, provider, timeout, &traceback);
        let error_str = raised
            .error_str
            .unwrap_or_else(|| raised.exception.message.clone());
        Self::Raised {
            exception: LoggedException {
                traceback,
                ..raised.exception
            },
            error_str,
        }
    }

    /// The payload's `error_information`, with `redact` (Python's `redact_string`) over what it
    /// logs of the message and the traceback.
    pub fn error_information(&self, redact: impl Fn(&str) -> String) -> ErrorInformation {
        match self {
            Self::Raised { exception, .. } => exception.error_information(redact),
            Self::ClientDisconnected => ErrorInformation::client_disconnected(),
        }
    }

    /// The payload's `error_str`, redacted by `redact` as the failure handler redacts it.
    pub fn error_str(&self, redact: impl Fn(&str) -> String) -> String {
        match self {
            Self::Raised { error_str, .. } => redact(error_str),
            Self::ClientDisconnected => CLIENT_DISCONNECTED_MESSAGE.to_owned(),
        }
    }
}

/// One error read as its Python exception, with its `str()` where it differs from its message.
struct Raised {
    exception: LoggedException,
    error_str: Option<String>,
}

impl Raised {
    fn of(error: &Error, provider: Option<&str>, timeout: Option<Duration>, report: &str) -> Self {
        let exception = match error {
            Error::Route(route) => route_failure(route, provider, timeout, report),
            Error::Ocr(ocr) => ocr_failure(ocr, provider, report),
            Error::ModelAccess(AuthzDenial::ModelAccess { list, .. }) => LoggedException {
                proxy_type: Some(denied_type(*list).to_owned()),
                ..raised_as(
                    ExceptionClass::ModelAccessDeniedProxyException,
                    "403",
                    &error.to_string(),
                )
            },
            Error::UnknownModel { .. } => {
                let detail = PyValue::Dict(vec![(
                    PyValue::Str("error".to_owned()),
                    PyValue::Str(error.to_string()),
                )]);
                return http_exception(ExceptionClass::ProxyModelNotFound, "400", &repr(&detail));
            }
            Error::NoDeployments(refused) => {
                raised_as(ExceptionClass::RouterRateLimit, "429", &refused.to_string())
            }
            Error::Routing(routing) => routing_failure(routing, provider, report),
            Error::Refused(refusal) => return refused(refusal, provider),
            Error::InvalidBody(_) | Error::BodyTooLarge => raised_as(
                ExceptionClass::ProxyException,
                error.status().as_str(),
                report,
            ),
            Error::Callback(_) | Error::Internal => {
                raised_as(ExceptionClass::Exception, "", report)
            }
        };
        Self {
            exception,
            error_str: None,
        }
    }
}

/// `ProxyErrorTypes.get_model_access_error_type_for_object` for the list that refused.
fn denied_type(list: DeniedList) -> &'static str {
    match list {
        DeniedList::Key => "key_model_access_denied",
    }
}

/// An exception of `class` with status `code`, raised with `message`, that names no provider.
fn raised_as(class: ExceptionClass, code: &str, message: &str) -> LoggedException {
    LoggedException {
        class,
        code: code.to_owned(),
        llm_provider: String::new(),
        message: class.message(message),
        provider_request_id: None,
        rate_limit_category: class
            .is_a(ExceptionClass::RateLimit)
            .then_some(RateLimitCategory::VendorRateLimit),
        rate_limit_type: None,
        proxy_type: None,
        traceback: String::new(),
    }
}

/// A Starlette `HTTPException` with `detail`: it has no `.message`, so its message is its
/// `str()`, `"{status_code}: {detail}"`.
fn http_exception(class: ExceptionClass, code: &str, detail: &str) -> Raised {
    let text = format!("{code}: {detail}");
    Raised {
        exception: raised_as(class, code, &text),
        error_str: Some(text),
    }
}

/// The exception `exception_type` maps `original` to for `provider`, or the class `class` the
/// route already decided.
fn provider_failure(
    original: &OriginalException,
    class: Option<PublicError>,
    provider: Option<&str>,
    headers: &[(String, String)],
) -> LoggedException {
    let provider = provider.unwrap_or_default();
    let context = ExceptionContext {
        custom_llm_provider: provider.to_owned(),
        ..ExceptionContext::default()
    };
    let mapped = exception_type(&context, None, original);
    let public = class.unwrap_or(mapped.error);
    LoggedException {
        llm_provider: provider.to_owned(),
        provider_request_id: provider_request_id(headers),
        ..raised_as(
            exception_class(public),
            &public.status_code().to_string(),
            &mapped.message,
        )
    }
}

/// The litellm class a provider failure maps to.
fn exception_class(error: PublicError) -> ExceptionClass {
    match error {
        PublicError::BadRequest => ExceptionClass::BadRequest,
        PublicError::ContextWindowExceeded => ExceptionClass::ContextWindowExceeded,
        PublicError::ContentPolicyViolation => ExceptionClass::ContentPolicyViolation,
        PublicError::Authentication => ExceptionClass::Authentication,
        PublicError::PermissionDenied => ExceptionClass::PermissionDenied,
        PublicError::NotFound => ExceptionClass::NotFound,
        PublicError::Timeout { .. } => ExceptionClass::Timeout,
        PublicError::RateLimit => ExceptionClass::RateLimit,
        PublicError::InternalServer => ExceptionClass::InternalServer,
        PublicError::BadGateway => ExceptionClass::BadGateway,
        PublicError::ServiceUnavailable => ExceptionClass::ServiceUnavailable,
        PublicError::ApiConnection => ExceptionClass::ApiConnection,
        PublicError::Api { .. } => ExceptionClass::Api,
    }
}

/// A route's failure: the provider's answer as `exception_type` maps it, a timeout as its
/// `Timeout`, a missing credential as the 401 the provider handler raises, a request the route
/// refused as a `BadRequestError`, and a call that got no answer as the `APIConnectionError`
/// Python maps an unanswered call to. A request the gateway declined takes the class of the
/// status it is answered with.
fn route_failure(
    route: &InferenceError,
    provider: Option<&str>,
    timeout: Option<Duration>,
    report: &str,
) -> LoggedException {
    let headers = route.upstream_headers();
    let original = match (route.upstream_response(), route.timeout()) {
        (Some((status, body)), _) => OriginalException::Http {
            status,
            body: body.to_owned(),
            headers: headers.to_vec(),
        },
        (None, Some(connect)) => OriginalException::Timeout {
            phase: match connect {
                true => TimeoutPhase::Connect,
                false => TimeoutPhase::Read,
            },
            timeout_seconds: timeout.map(|timeout| timeout.as_secs_f64()),
            elapsed_seconds: None,
        },
        (None, None) => OriginalException::Plain {
            message: report.to_owned(),
        },
    };
    let class = match &original {
        OriginalException::Plain { .. } if route.declined().is_some() => classify_status(501),
        OriginalException::Plain { .. } if route.is_missing_credential() => {
            Some(PublicError::Authentication)
        }
        OriginalException::Plain { .. } if route.is_request() => Some(PublicError::BadRequest),
        _ => None,
    };
    provider_failure(&original, class, provider, headers)
}

fn ocr_failure(ocr: &OcrError, provider: Option<&str>, report: &str) -> LoggedException {
    let (original, headers) = match ocr.upstream_response() {
        Some((status, body, headers)) => {
            let headers: Vec<(String, String)> = headers
                .iter()
                .filter_map(|(name, value)| {
                    Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
                })
                .collect();
            let original = OriginalException::Http {
                status,
                body: body.to_owned(),
                headers: headers.clone(),
            };
            (original, headers)
        }
        None => {
            let original = OriginalException::Plain {
                message: report.to_owned(),
            };
            (original, Vec::new())
        }
    };
    let class = match &original {
        OriginalException::Plain { .. } if ocr.is_missing_credential() => {
            Some(PublicError::Authentication)
        }
        OriginalException::Plain { .. } => ocr.http_status_code().and_then(classify_status),
        OriginalException::Http { .. }
        | OriginalException::Connection { .. }
        | OriginalException::Timeout { .. } => None,
    };
    provider_failure(&original, class, provider, &headers)
}

/// An error the router made itself: a pre-call check's `RateLimitError`, which names no
/// provider, a failed attempt's exception, or the exception a pick raises.
fn routing_failure(
    routing: &litellm_router_core::Error,
    provider: Option<&str>,
    report: &str,
) -> LoggedException {
    if let Some(failure) = routing.attempt() {
        let refused = routing.refused().is_some() && failure.kind == FailureKind::RateLimit;
        let names = (failure.names_provider && !refused)
            .then_some(provider)
            .flatten();
        return attempt_exception(failure, names, report);
    }
    match routing.pick_failure() {
        Some(failure) => attempt_exception(&failure, None, report),
        None => raised_as(ExceptionClass::Exception, "", report),
    }
}

fn attempt_exception(
    failure: &AttemptError,
    provider: Option<&str>,
    report: &str,
) -> LoggedException {
    let class = failure
        .class_names()
        .first()
        .map_or(ExceptionClass::Exception, |name| {
            ExceptionClass::named(name)
        });
    let code = match failure.status_absent {
        true => String::new(),
        false => failure
            .status
            .map(|status| status.to_string())
            .unwrap_or_default(),
    };
    LoggedException {
        llm_provider: provider.unwrap_or_default().to_owned(),
        ..raised_as(class, &code, failure.detail.as_deref().unwrap_or(report))
    }
}

/// A callback's refusal: a limiter's is its `ProxyRateLimitError`, which names the provider of
/// the call's model (`resolve_llm_provider_for_rate_limit`) and the limit that refused it, one
/// that names its own `type` is the `ProxyException` the callback raised, and any other is the
/// `HTTPException` it raised.
fn refused(refusal: &Refusal, provider: Option<&str>) -> Raised {
    let status = refusal.status.as_str();
    match refusal.kind.as_deref() {
        Some(THROTTLING) => {
            let rate_limit_type = refusal
                .headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(RATE_LIMIT_TYPE_HEADER))
                .and_then(|(_, value)| RateLimitType::from_v3(value));
            let raised = http_exception(ExceptionClass::ProxyRateLimit, status, &refusal.message);
            Raised {
                exception: LoggedException {
                    llm_provider: provider.unwrap_or(PROXY_PROVIDER).to_owned(),
                    message: ExceptionClass::ProxyRateLimit.message(&refusal.message),
                    rate_limit_category: Some(RateLimitCategory::LitellmRateLimit),
                    rate_limit_type,
                    ..raised.exception
                },
                ..raised
            }
        }
        Some(kind) => Raised {
            exception: LoggedException {
                proxy_type: Some(kind.to_owned()),
                ..raised_as(
                    ExceptionClass::ProxyException,
                    refusal.code.as_deref().unwrap_or(status),
                    &refusal.message,
                )
            },
            error_str: None,
        },
        None => http_exception(ExceptionClass::HttpException, status, &refusal.message),
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use rstest::rstest;

    use super::*;
    use crate::error::NoDeployments;

    fn upstream(status: u16, body: &str, headers: Vec<(String, String)>) -> Error {
        Error::Route(
            litellm_http::Error::upstream_with_headers(status, body.into(), headers).into(),
        )
    }

    fn logged(error: &Error, provider: Option<&str>) -> (LoggedException, String) {
        match CallFailure::raised(error, provider, None) {
            CallFailure::Raised {
                exception,
                error_str,
            } => (exception, error_str),
            CallFailure::ClientDisconnected => unreachable!(),
        }
    }

    fn information(error: &Error, provider: Option<&str>) -> ErrorInformation {
        CallFailure::raised(error, provider, None).error_information(str::to_owned)
    }

    #[rstest]
    #[case::rate_limited(
        429,
        r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
        "RateLimitError",
        "429",
        "429_RATE_LIMIT_EXCEEDED"
    )]
    #[case::overloaded(
        529,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        "InternalServerError",
        "500",
        "503_PROVIDER_OVERLOADED"
    )]
    #[case::prompt_too_long(
        400,
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}"#,
        "ContextWindowExceededError",
        "400",
        "400_CONTEXT_WINDOW_EXCEEDED"
    )]
    #[case::bad_key(
        401,
        "invalid x-api-key",
        "AuthenticationError",
        "401",
        "401_AUTHENTICATION_FAILED"
    )]
    #[case::not_found(404, "model: nope", "NotFoundError", "404", "404_RESOURCE_NOT_FOUND")]
    #[case::too_large(
        413,
        "request too large",
        "BadRequestError",
        "400",
        "400_INVALID_REQUEST"
    )]
    #[case::unavailable(
        503,
        "down",
        "ServiceUnavailableError",
        "503",
        "503_PROVIDER_OVERLOADED"
    )]
    fn a_provider_answer_is_the_class_exception_type_maps_it_to(
        #[case] status: u16,
        #[case] body: &str,
        #[case] class: &str,
        #[case] code: &str,
        #[case] normalized: &str,
    ) {
        let information = information(&upstream(status, body, Vec::new()), Some("anthropic"));
        assert_eq!(
            (
                information.error_class.as_deref(),
                information.error_code.as_deref(),
                information.llm_provider.as_deref(),
                information.normalized_error.as_deref(),
            ),
            (Some(class), Some(code), Some("anthropic"), Some(normalized))
        );
    }

    #[rstest]
    fn a_provider_failure_logs_its_message_request_id_and_cause() {
        let error = upstream(
            429,
            "slow down",
            vec![("request-id".to_owned(), "req_1".to_owned())],
        );
        let (exception, error_str) = logged(&error, Some("anthropic"));
        assert_eq!(
            (
                exception.message.as_str(),
                error_str.as_str(),
                exception.provider_request_id.as_deref(),
                exception.rate_limit_category,
                exception.is_expected_client_error(),
                exception.traceback.is_empty(),
            ),
            (
                "litellm.RateLimitError: AnthropicException - slow down",
                "litellm.RateLimitError: AnthropicException - slow down",
                Some("req_1"),
                Some(RateLimitCategory::VendorRateLimit),
                false,
                false,
            )
        );
    }

    #[rstest]
    fn a_lost_connection_is_an_api_connection_error() {
        let error = Error::Route(
            litellm_http::Error::from(tokio_tungstenite::tungstenite::Error::ConnectionClosed)
                .into(),
        );
        let information = information(&error, Some("anthropic"));
        assert_eq!(
            (
                information.error_class.as_deref(),
                information.error_code.as_deref()
            ),
            (Some("APIConnectionError"), Some("500"))
        );
    }

    #[rstest]
    fn a_missing_key_is_an_authentication_error() {
        let error = Error::Route(InferenceError::Auth(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        }));
        let information = information(&error, Some("anthropic"));
        assert_eq!(
            (
                information.error_class.as_deref(),
                information.error_code.as_deref()
            ),
            (Some("AuthenticationError"), Some("401"))
        );
    }

    fn limiter_refusal() -> Error {
        Error::Refused(Refusal {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded for api_key: hash. Limit type: requests.".to_owned(),
            kind: Some(THROTTLING.to_owned()),
            code: Some("429".to_owned()),
            headers: vec![
                ("retry-after".to_owned(), "60".to_owned()),
                ("rate_limit_type".to_owned(), "requests".to_owned()),
            ],
        })
    }

    #[rstest]
    #[case::resolved_provider(Some("anthropic"), "anthropic")]
    #[case::unresolved_provider(None, "litellm_proxy")]
    fn a_limiter_refusal_is_its_proxy_rate_limit_error(
        #[case] provider: Option<&str>,
        #[case] llm_provider: &str,
    ) {
        let failure = CallFailure::raised(&limiter_refusal(), provider, None);
        assert_eq!(
            failure.error_information(str::to_owned),
            ErrorInformation {
                error_code: Some("429".to_owned()),
                error_class: Some("ProxyRateLimitError".to_owned()),
                llm_provider: Some(llm_provider.to_owned()),
                traceback: Some(String::new()),
                error_message: Some(
                    "litellm.RateLimitError: Rate limit exceeded for api_key: hash. Limit type: requests."
                        .to_owned()
                ),
                error_rate_limit_category: Some("litellm_rate_limit".to_owned()),
                error_rate_limit_type: Some("requests".to_owned()),
                normalized_error: Some("429_RATE_LIMIT_EXCEEDED".to_owned()),
                ..ErrorInformation::default()
            }
        );
        assert_eq!(
            failure.error_str(str::to_owned),
            "429: Rate limit exceeded for api_key: hash. Limit type: requests."
        );
    }

    #[rstest]
    #[case::proxy_exception(
        Refusal { kind: Some("budget_exceeded".to_owned()), code: Some("400".to_owned()), ..Refusal::new(StatusCode::BAD_REQUEST, "over budget") },
        "ProxyException",
        "400",
        "over budget",
        "429_BUDGET_EXCEEDED"
    )]
    #[case::http_exception(
        Refusal::new(StatusCode::FORBIDDEN, "blocked"),
        "HTTPException",
        "403",
        "403: blocked",
        "403_PERMISSION_DENIED"
    )]
    fn another_refusal_is_the_exception_its_callback_raised(
        #[case] refusal: Refusal,
        #[case] class: &str,
        #[case] code: &str,
        #[case] message: &str,
        #[case] normalized: &str,
    ) {
        let information = information(&Error::Refused(refusal), None);
        assert_eq!(
            (
                information.error_class.as_deref(),
                information.error_code.as_deref(),
                information.error_message.as_deref(),
                information.normalized_error.as_deref(),
                information.llm_provider.as_deref(),
            ),
            (
                Some(class),
                Some(code),
                Some(message),
                Some(normalized),
                Some("")
            )
        );
    }

    #[rstest]
    fn no_deployments_is_a_router_rate_limit_error() {
        let error = Error::from(NoDeployments {
            model: "primary".into(),
            cooldown: 5.0,
            all_cooling: false,
            pre_call_checks: false,
            cooling: vec![],
        });
        let information = information(&error, Some("anthropic"));
        assert_eq!(
            (
                information.error_class.as_deref(),
                information.error_code.as_deref(),
                information.llm_provider.as_deref(),
                information.traceback.as_deref(),
                information.normalized_error.as_deref(),
                information.error_cooldown_deployments,
            ),
            (
                Some("RouterRateLimitError"),
                Some("429"),
                Some(""),
                Some(""),
                Some("429_NO_HEALTHY_DEPLOYMENTS"),
                None,
            )
        );
    }

    #[rstest]
    fn an_unknown_model_is_the_proxys_model_not_found_error() {
        let error = Error::UnknownModel {
            route: "/v1/messages",
            model: "nope".into(),
        };
        let (exception, error_str) = logged(&error, None);
        let message = "400: {'error': '/v1/messages: Invalid model name passed in model=nope. Call `/v1/models` to view available models for your key.'}";
        assert_eq!(
            (
                exception.class,
                exception.code.as_str(),
                exception.message.as_str(),
                error_str.as_str(),
                exception.is_expected_client_error(),
            ),
            (
                ExceptionClass::ProxyModelNotFound,
                "400",
                message,
                message,
                true
            )
        );
    }

    #[rstest]
    fn a_refused_model_is_the_key_model_access_denial() {
        let error = Error::ModelAccess(AuthzDenial::ModelAccess {
            list: DeniedList::Key,
            model: "claude".into(),
        });
        let information = information(&error, None);
        assert_eq!(
            (
                information.error_class.as_deref(),
                information.error_code.as_deref(),
                information.normalized_error.as_deref(),
                information.traceback.as_deref(),
            ),
            (
                Some("ModelAccessDeniedProxyException"),
                Some("403"),
                Some("403_MODEL_ACCESS_DENIED"),
                Some(""),
            )
        );
    }

    #[rstest]
    fn an_internal_failure_keeps_its_cause_as_the_traceback() {
        let information = information(&Error::Internal, None);
        assert_eq!(
            (
                information.error_class.as_deref(),
                information.error_code.as_deref(),
                information.traceback.as_deref(),
            ),
            (
                Some("Exception"),
                Some(""),
                Some("unexpected HTTP host operation")
            )
        );
    }

    #[rstest]
    fn a_client_disconnect_logs_python_s_499() {
        let failure = CallFailure::ClientDisconnected;
        assert_eq!(
            (
                failure.error_information(str::to_owned),
                failure.error_str(str::to_owned)
            ),
            (
                ErrorInformation::client_disconnected(),
                "Client disconnected the request".to_owned()
            )
        );
    }
}
