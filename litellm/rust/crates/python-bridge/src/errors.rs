use std::error::Error as StdError;

use litellm_auth::gcp::google::GoogleLoginFailure;
use litellm_error::{Report, Source};
use litellm_inference::Error;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use reqwest::header::HeaderMap;

pyo3::create_exception!(
    _native,
    RustBridgeDeclined,
    pyo3::exceptions::PyException,
    "The route declined before calling the provider, so the host may retry on its own path."
);

pyo3::create_exception!(
    _native,
    RustUpstreamError,
    pyo3::exceptions::PyException,
    "The provider call was already issued and the provider answered with a failure. Args are (status, body); the headers attribute holds the provider's (name, value) response headers."
);

const MESSAGES_REQUEST_ERROR: &str = "messages_request_error";
/// The rejection is a parameter the model does not support, which Python raises as
/// `UnsupportedParamsError`.
const MESSAGES_UNSUPPORTED_PARAM: &str = "messages_unsupported_param";
/// The model a provider config names on its `UnsupportedParamsError`, which Python then also
/// attributes to the request's provider.
const MESSAGES_UNSUPPORTED_PARAM_MODEL: &str = "messages_unsupported_param_model";
/// The phase label (`Connection` or `Read`) of a timed-out call, for Python's timeout text, or
/// `Body` for a streamed body read that timed out after the head arrived.
const MESSAGES_TIMEOUT: &str = "messages_timeout";
// `(received, expected, reset)` of a streamed body the connection ended early.
const MESSAGES_BODY_CUT: &str = "messages_body_cut";
/// No credential reached the provider, which Python raises as `AuthenticationError`.
const MESSAGES_MISSING_CREDENTIAL: &str = "messages_missing_credential";
/// What the provider config needed and did not get, when that was a named param rather than the key.
const MESSAGES_MISSING_PARAM: &str = "messages_missing_param";
/// Python refuses the request only once its logger knows the resolved model and provider (the
/// provider config's request transform), so the failure logs them.
pub(crate) const MESSAGES_LATE_REJECTION: &str = "messages_late_rejection";
/// Python's request transform raises a builtin exception (`TypeError`) that its handler does not
/// map, so the caller gets the native exception as it is.
const MESSAGES_PYTHON_RAISED: &str = "messages_python_raised";
/// The text of a refusal Python's provider config raises from its request transform, which its
/// HTTP handler maps as a provider error.
const MESSAGES_PROVIDER_REJECTION: &str = "messages_provider_rejection";
/// Python's provider config raises the refusal as litellm's `BadRequestError`.
const MESSAGES_PROVIDER_BAD_REQUEST: &str = "messages_provider_bad_request";
/// Why the STS call made for the credentials failed, as `(kind, operation, code, message)`.
const MESSAGES_STS_FAILURE: &str = "messages_sts_failure";
/// The `scheme://host:port` of a provider the call could not connect to.
const MESSAGES_CONNECT_ORIGIN: &str = "messages_connect_origin";
/// The provider replied with a body litellm refuses to read, which Python raises as `APIError` 500.
const MESSAGES_REFUSED_REPLY: &str = "messages_refused_reply";
/// Which failure Python's AWS signing raises for a credential Rust could not acquire, or `replay`
/// when the Bedrock invoke host runs Python's signing again for its exception. The Mantle routes
/// turn `replay` into `unavailable`, which maps from a fixed class list.
const MESSAGES_CREDENTIAL_FAILURE: &str = "messages_credential_failure";
/// `(class, text)` of the exception Python's Vertex AI login raises for a login Rust could not
/// complete, which the route raises in its place.
const MESSAGES_GOOGLE_LOGIN_FAILURE: &str = "messages_google_login_failure";
/// The AWS profile no config or credentials file defines, for the `profile_not_found` credential
/// failure that botocore raises as `ProfileNotFound`.
const MESSAGES_MISSING_PROFILE: &str = "messages_missing_profile";

struct ProviderRejection {
    text: String,
    bad_request: bool,
}

fn credential_failure(error: &Error) -> Option<&'static str> {
    let (Error::Auth(auth) | Error::Llm(litellm_llms::Error::Auth(auth))) = error else {
        return None;
    };
    match auth {
        litellm_auth::Error::Provider(_) => Some(aws_credential_failure(auth)),
        litellm_auth::Error::CredentialAcquisition(_)
        | litellm_auth::Error::CredentialChain { .. } => Some("replay"),
        litellm_auth::Error::InvalidConfiguration(_)
        | litellm_auth::Error::EmptyCallerCredential { .. }
        | litellm_auth::Error::MissingApiKey { .. }
        | litellm_auth::Error::MissingParam { .. }
        | litellm_auth::Error::MissingApiBase { .. }
        | litellm_auth::Error::InvalidHeader => None,
    }
}

fn google_login_failure(failure: &GoogleLoginFailure) -> (&'static str, &str) {
    (failure.exception.into(), failure.message.as_str())
}

fn innermost_aws_error<'a>(
    error: &'a (dyn StdError + 'static),
) -> Option<&'a litellm_auth_aws::Error> {
    std::iter::successors(Some(error), |&cause| cause.source())
        .filter_map(|cause| {
            cause
                .downcast_ref::<litellm_auth_aws::Error>()
                .or_else(|| cause.downcast_ref::<Source>()?.downcast_ref())
        })
        .last()
}

fn missing_param(error: &Error) -> Option<&litellm_auth::Error> {
    let (Error::Auth(auth) | Error::Llm(litellm_llms::Error::Auth(auth))) = error else {
        return None;
    };
    matches!(auth, litellm_auth::Error::MissingParam { .. }).then_some(auth)
}

fn missing_profile(error: &Error) -> Option<&str> {
    let (Error::Auth(auth) | Error::Llm(litellm_llms::Error::Auth(auth))) = error else {
        return None;
    };
    match innermost_aws_error(auth)? {
        litellm_auth_aws::Error::ProfileNotFound { profile } => Some(profile),
        _ => None,
    }
}

fn sts_failure(error: &Error) -> Option<(&'static str, &'static str, String, String)> {
    let (Error::Auth(auth) | Error::Llm(litellm_llms::Error::Auth(auth))) = error else {
        return None;
    };
    Some(match innermost_aws_error(auth)?.sts_failure()? {
        litellm_auth_aws::StsFailure::InvalidWebIdentityToken { message } => {
            ("invalid_web_identity_token", "", String::new(), message)
        }
        litellm_auth_aws::StsFailure::Rejected {
            operation,
            code,
            message,
        } => ("rejected", operation, code, message),
        litellm_auth_aws::StsFailure::Unreachable => {
            ("unreachable", "", String::new(), String::new())
        }
    })
}

fn aws_credential_failure(error: &(dyn StdError + 'static)) -> &'static str {
    match innermost_aws_error(error) {
        Some(litellm_auth_aws::Error::WebIdentityTokenReference) => "web_identity_reference",
        Some(litellm_auth_aws::Error::MissingWebIdentityToken) => "missing_web_identity_token",
        Some(litellm_auth_aws::Error::ProfileNotFound { .. }) => "profile_not_found",
        Some(failure) if failure.is_empty_default_chain() => "no_credentials",
        _ => "replay",
    }
}

fn provider_rejection(rejection: &(dyn StdError + 'static)) -> Option<ProviderRejection> {
    match rejection.downcast_ref::<litellm_llms::Error>() {
        Some(litellm_llms::Error::MissingField("max_tokens")) => {
            return Some(ProviderRejection {
                text: "max_tokens is required for Anthropic /v1/messages API".to_owned(),
                bad_request: false,
            });
        }
        Some(_) => return None,
        None => {}
    }
    match rejection.downcast_ref::<litellm_llms::ErrorDetail>() {
        Some(
            detail @ (litellm_llms::ErrorDetail::InvalidChoice {
                field: "reasoning effort",
                ..
            }
            | litellm_llms::ErrorDetail::UnsupportedBedrockWebSearchTool { .. }),
        ) => Some(ProviderRejection {
            text: detail.to_string(),
            bad_request: true,
        }),
        Some(
            detail @ (litellm_llms::ErrorDetail::InvalidReasoningEffort { .. }
            | litellm_llms::ErrorDetail::UnsupportedValue {
                field: "effort", ..
            }),
        ) => Some(ProviderRejection {
            text: detail.to_string(),
            bad_request: false,
        }),
        _ => None,
    }
}

/// Whether Python's provider config raises this refusal as litellm's own `BadRequestError`, as the
/// Bedrock invoke transform does, naming the provider it raises for.
pub(crate) fn raises_bad_request(error: &Error) -> bool {
    error
        .rejected_request()
        .and_then(provider_rejection)
        .is_some_and(|rejection| rejection.bad_request)
}

pub(crate) fn upstream_error(
    py: Python<'_>,
    status: u16,
    body: &str,
    headers: &HeaderMap,
) -> PyResult<PyErr> {
    let error = RustUpstreamError::new_err((status, body.to_owned()));
    let pairs: Vec<(&str, String)> = headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
            )
        })
        .collect();
    error.value(py).setattr("headers", pairs)?;
    Ok(error)
}

/// A request the caller got wrong is a `ValueError`; anything else is a `RuntimeError`.
pub(crate) fn by_fault(is_request: bool, error: &(dyn StdError + 'static)) -> PyErr {
    let message = Report::new(error).to_string();
    if is_request {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}

/// How a route flags a rejected request for its Python route module, which raises the public
/// exception for it instead of mapping the native one.
#[derive(Clone, Copy)]
pub(crate) enum RequestMarker {
    Unmarked,
    Messages,
}

/// Maps a route failure once. A core decline and a Python exception raised inside the call are
/// returned as they are, everything else is mapped and handed to `public`, the route's public
/// exception mapping.
pub(crate) fn route_failure(
    py: Python<'_>,
    error: Error,
    marker: RequestMarker,
    public: impl FnOnce(PyErr) -> PyResult<PyErr>,
) -> PyResult<PyErr> {
    if let Some(decline) = error.declined() {
        return Ok(RustBridgeDeclined::new_err(decline.reason));
    }
    if let Some(raised) = crate::secrets::python_error(py, &error) {
        return Ok(raised);
    }
    if let Some(raised) = crate::routes::messages::workload_identity::python_error(py, &error) {
        return Ok(raised);
    }
    if let Some((status, body)) = error.upstream_response() {
        let headers = error
            .upstream_headers()
            .iter()
            .filter_map(|(name, value)| {
                Some((
                    reqwest::header::HeaderName::from_bytes(name.as_bytes()).ok()?,
                    reqwest::header::HeaderValue::from_str(value).ok()?,
                ))
            })
            .collect::<HeaderMap>();
        return public(upstream_error(py, status, body, &headers)?);
    }
    let native = match (marker, error.rejected_request()) {
        (RequestMarker::Messages, Some(rejection))
            if matches!(
                rejection.downcast_ref::<litellm_llms::ErrorDetail>(),
                Some(litellm_llms::ErrorDetail::UnorderableTypes { .. })
            ) =>
        {
            let native = PyTypeError::new_err(rejection.to_string());
            let value = native.value(py);
            value.setattr(MESSAGES_LATE_REJECTION, true)?;
            value.setattr(MESSAGES_PYTHON_RAISED, true)?;
            native
        }
        (RequestMarker::Messages, Some(rejection))
            if matches!(
                rejection.downcast_ref::<litellm_llms::ErrorDetail>(),
                Some(
                    litellm_llms::ErrorDetail::MissingAzureApiBase
                        | litellm_llms::ErrorDetail::InvalidAwsRegion { .. }
                        | litellm_llms::ErrorDetail::InvalidMantleRegion {
                            check: litellm_llms::MantleRegionCheck::Request,
                            ..
                        }
                )
            ) =>
        {
            // Python raises these from building or signing the call, after the provider resolved.
            let native = PyValueError::new_err(rejection.to_string());
            let value = native.value(py);
            value.setattr(MESSAGES_LATE_REJECTION, true)?;
            value.setattr(MESSAGES_PYTHON_RAISED, true)?;
            native
        }
        (RequestMarker::Messages, Some(rejection)) => {
            let native = PyValueError::new_err(Report::new(rejection).to_string());
            let value = native.value(py);
            value.setattr(MESSAGES_REQUEST_ERROR, true)?;
            let detail = rejection.downcast_ref::<litellm_llms::ErrorDetail>();
            if matches!(
                detail,
                Some(
                    litellm_llms::ErrorDetail::UnsupportedParameter { .. }
                        | litellm_llms::ErrorDetail::NonNumericThinkingBudget { .. }
                        | litellm_llms::ErrorDetail::UnsupportedVertexAnthropicVersion { .. }
                        | litellm_llms::ErrorDetail::NonObjectVertexOutputFormat { .. }
                )
            ) {
                value.setattr(MESSAGES_UNSUPPORTED_PARAM, true)?;
            }
            if matches!(
                detail,
                Some(
                    litellm_llms::ErrorDetail::NonNumericThinkingBudget { .. }
                        | litellm_llms::ErrorDetail::UnsupportedVertexAnthropicVersion { .. }
                        | litellm_llms::ErrorDetail::NonObjectVertexOutputFormat { .. }
                )
            ) {
                value.setattr(MESSAGES_LATE_REJECTION, true)?;
            }
            if let Some(litellm_llms::ErrorDetail::UnsupportedVertexAnthropicVersion {
                model,
                ..
            }) = detail
            {
                value.setattr(MESSAGES_UNSUPPORTED_PARAM_MODEL, model.as_str())?;
            }
            if let Some(provider) = provider_rejection(rejection) {
                value.setattr(MESSAGES_PROVIDER_REJECTION, provider.text)?;
                if provider.bad_request {
                    value.setattr(MESSAGES_PROVIDER_BAD_REQUEST, true)?;
                }
            }
            native
        }
        (RequestMarker::Messages, None) => {
            let native = by_fault(error.is_request(), &error);
            let value = native.value(py);
            if let Some(connect) = error.timeout() {
                let phase = match (connect, error.body_read_timeout()) {
                    (true, _) => "Connection",
                    (false, true) => "Body",
                    (false, false) => "Read",
                };
                value.setattr(MESSAGES_TIMEOUT, phase)?;
            }
            if let Some(cut) = error.body_cut() {
                value.setattr(MESSAGES_BODY_CUT, (cut.received, cut.expected, cut.reset))?;
            }
            if let Some(origin) = error.connect_origin() {
                value.setattr(MESSAGES_CONNECT_ORIGIN, origin)?;
            }
            if error.is_missing_credential() {
                value.setattr(MESSAGES_MISSING_CREDENTIAL, true)?;
            }
            if let Some(missing) = missing_param(&error) {
                value.setattr(MESSAGES_MISSING_PARAM, missing.to_string())?;
            }
            if matches!(
                error.refused_response(),
                Some(litellm_llms::ErrorDetail::UnbillableUsage { .. })
            ) {
                value.setattr(MESSAGES_REFUSED_REPLY, true)?;
            }
            if let Some(failure) = credential_failure(&error) {
                value.setattr(MESSAGES_CREDENTIAL_FAILURE, failure)?;
            }
            if let Some(failure) = litellm_error::find::<GoogleLoginFailure>(&error) {
                value.setattr(MESSAGES_GOOGLE_LOGIN_FAILURE, google_login_failure(failure))?;
            }
            if let Some(failure) = sts_failure(&error) {
                value.setattr(MESSAGES_STS_FAILURE, failure)?;
            }
            if let Some(profile) = missing_profile(&error) {
                value.setattr(MESSAGES_MISSING_PROFILE, profile)?;
            }
            native
        }
        (RequestMarker::Unmarked, _) => by_fault(error.is_request(), &error),
    };
    public(native)
}

pub(crate) fn route_error_to_pyerr(error: Error) -> PyErr {
    Python::attach(|py| {
        route_failure(py, error, RequestMarker::Unmarked, Ok).unwrap_or_else(std::convert::identity)
    })
}

#[cfg(test)]
mod tests {
    use litellm_auth::gcp::google::PythonException;
    use litellm_core_utils::decline::Decline;
    use reqwest::header::{HeaderValue, RETRY_AFTER};

    use super::*;

    fn upstream_payload(py: Python<'_>, failure: &PyErr) -> ((u16, String), Vec<(String, String)>) {
        let value = failure.value(py);
        (
            value.getattr("args").unwrap().extract().unwrap(),
            value.getattr("headers").unwrap().extract().unwrap(),
        )
    }

    #[rstest::rstest]
    #[case::invalid_provider(Error::InvalidProvider("unknown".into()), true)]
    #[case::invalid_request(
        Error::invalid_request(litellm_llms::ErrorDetail::Expected {
            subject: "messages",
            expected: "non-empty",
        }),
        true
    )]
    #[case::lost_connection(
        litellm_http::Error::from(tokio_tungstenite::tungstenite::Error::ConnectionClosed).into(),
        false
    )]
    #[case::invalid_response(
        Error::invalid_response(litellm_llms::ErrorDetail::MissingField {
            subject: "response",
            field: "usage",
        }),
        false
    )]
    #[case::post_call_decline(
        Error::post_call(Error::Declined(Decline::new("result rejected"))),
        false
    )]
    fn route_failures_are_terminal(#[case] error: Error, #[case] is_request: bool) {
        Python::initialize();
        Python::attach(|py| {
            let failure = route_error_to_pyerr(error);
            assert!(!failure.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(failure.is_instance_of::<PyValueError>(py), is_request);
            assert_eq!(failure.is_instance_of::<PyRuntimeError>(py), !is_request);
        });
    }

    #[rstest::rstest]
    #[case::unmarked(RequestMarker::Unmarked)]
    #[case::messages(RequestMarker::Messages)]
    fn a_core_decline_hands_the_call_back_to_python_before_the_public_mapping(
        #[case] marker: RequestMarker,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = Error::from(litellm_llms::Error::Declined(Decline::new(
                "unrecognized request parameter",
            )));
            let declined = route_failure(py, error, marker, |_| {
                panic!("a decline must not reach the public mapping")
            })
            .unwrap();
            assert!(declined.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(
                declined
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(String,)>()
                    .unwrap(),
                ("unrecognized request parameter".into(),)
            );
        });
    }

    #[rstest::rstest]
    #[case::unmarked(RequestMarker::Unmarked)]
    #[case::messages(RequestMarker::Messages)]
    fn the_provider_answer_passes_through_with_a_headers_attribute(#[case] marker: RequestMarker) {
        Python::initialize();
        Python::attach(|py| {
            let error = litellm_http::Error::upstream(429, "slow down".into()).into();
            let upstream = route_failure(py, error, marker, Ok).unwrap();
            assert!(upstream.is_instance_of::<RustUpstreamError>(py));
            assert_eq!(
                upstream_payload(py, &upstream),
                ((429, "slow down".into()), Vec::new())
            );
            assert!(
                upstream
                    .value(py)
                    .getattr_opt(MESSAGES_REQUEST_ERROR)
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn upstream_error_keeps_the_provider_headers() {
        Python::initialize();
        Python::attach(|py| {
            let headers = HeaderMap::from_iter([(RETRY_AFTER, HeaderValue::from_static("17"))]);
            let upstream = upstream_error(py, 503, "busy", &headers).unwrap();
            assert_eq!(
                upstream_payload(py, &upstream),
                (
                    (503, "busy".into()),
                    vec![("retry-after".into(), "17".into())]
                )
            );
        });
    }

    #[rstest::rstest]
    #[case::ascii("busy ".repeat(80))]
    #[case::unicode("é".repeat(300))]
    fn long_provider_bodies_reach_python_intact(
        #[case] message: String,
        #[values(RequestMarker::Unmarked, RequestMarker::Messages)] marker: RequestMarker,
    ) {
        let envelope = serde_json::json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": message},
            "request_id": "provider-request",
        });
        let body = envelope.to_string();
        let error = litellm_http::Error::upstream(529, body.clone()).into();
        Python::initialize();
        Python::attach(|py| {
            let upstream = route_failure(py, error, marker, Ok).unwrap();
            assert!(upstream.is_instance_of::<RustUpstreamError>(py));
            let ((status, returned_body), headers) = upstream_payload(py, &upstream);
            assert_eq!(status, 529);
            assert_eq!(returned_body, body);
            assert!(headers.is_empty());
        });
    }

    #[test]
    fn missing_api_key_stays_a_runtime_error_while_other_auth_failures_are_value_errors() {
        Python::initialize();
        Python::attach(|py| {
            let missing = route_error_to_pyerr(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            }));
            assert!(missing.is_instance_of::<PyRuntimeError>(py));
            let invalid = route_error_to_pyerr(Error::Auth(litellm_auth::Error::InvalidHeader));
            assert!(invalid.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn a_wrapped_failure_renders_every_layer() {
        Python::initialize();
        Python::attach(|py| {
            let error = Error::post_call(
                litellm_http::Error::upstream(500, "hook backend down".into()).into(),
            );
            let report = Report::new(&error).to_string();
            let failure = route_error_to_pyerr(error);
            assert_eq!(failure.value(py).to_string(), report);
            assert!(report.contains("hook backend down"), "{report}");
        });
    }

    #[rstest::rstest]
    #[case::missing_max_tokens(
        litellm_llms::Error::MissingField("max_tokens").into(),
        Some(("max_tokens is required for Anthropic /v1/messages API", false))
    )]
    #[case::unmapped_effort(
        Error::invalid_request(litellm_llms::ErrorDetail::InvalidChoice {
            field: "reasoning effort",
            actual: "'ultra'".into(),
            choices: vec!["low", "none"],
        }),
        Some(("Unmapped reasoning effort: 'ultra'. Must be one of: 'low', 'none'.", true))
    )]
    #[case::bedrock_web_search(
        Error::invalid_request(litellm_llms::ErrorDetail::UnsupportedBedrockWebSearchTool {
            tool_type: "web_search_20250305".into(),
            model: "claude".into(),
        }),
        Some((
            "Bedrock does not support Anthropic's server-side web search tool (tool type 'web_search_20250305', model 'claude'). To use web search with this model, enable LiteLLM's web search interception so the proxy executes the search instead: https://docs.litellm.ai/docs/integrations/websearch_interception. Alternatively, remove the web_search tool from the request.",
            true
        ))
    )]
    #[case::other_missing_field(litellm_llms::Error::MissingField("model").into(), None)]
    #[case::unsupported_param(
        Error::invalid_request(litellm_llms::ErrorDetail::UnsupportedParameter {
            model: "claude".into(),
            param: "top_k".into(),
            value: "5".into(),
            hint: String::new(),
        }),
        None
    )]
    fn provider_config_rejections_carry_python_text(
        #[case] error: Error,
        #[case] expected: Option<(&str, bool)>,
    ) {
        assert_eq!(
            raises_bad_request(&error),
            expected.is_some_and(|(_, bad_request)| bad_request)
        );
        Python::initialize();
        Python::attach(|py| {
            let native = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            let value = native.value(py);
            let text = value
                .getattr_opt(MESSAGES_PROVIDER_REJECTION)
                .unwrap()
                .map(|value| value.extract::<String>().unwrap());
            let bad_request = value
                .getattr_opt(MESSAGES_PROVIDER_BAD_REQUEST)
                .unwrap()
                .is_some_and(|value| value.extract::<bool>().unwrap());
            assert_eq!(text.as_deref().map(|text| (text, bad_request)), expected);
            assert!(text.is_some() || !bad_request);
        });
    }

    fn web_identity_failure(failure: litellm_auth_aws::Error) -> litellm_auth_aws::Error {
        litellm_auth_aws::Error::Credentials {
            flow: litellm_auth_aws::CredentialFlow::WebIdentity,
            source: Source::new(
                aws_credential_types::provider::error::CredentialsError::provider_error(failure),
            ),
        }
    }

    #[rstest::rstest]
    #[case::environment_reference(
        litellm_auth_aws::Error::WebIdentityTokenReference,
        "web_identity_reference"
    )]
    #[case::unresolved_token(
        web_identity_failure(litellm_auth_aws::Error::MissingWebIdentityToken),
        "missing_web_identity_token"
    )]
    #[case::empty_default_chain(
        litellm_auth_aws::Error::Credentials {
            flow: litellm_auth_aws::CredentialFlow::DefaultChain,
            source: Source::new(aws_credential_types::provider::error::CredentialsError::not_loaded(
                "no providers in chain provided credentials",
            )),
        },
        "no_credentials"
    )]
    #[case::unreadable_token(
        web_identity_failure(litellm_auth_aws::Error::WebIdentityToken(Source::new(
            std::io::Error::other("secret source unavailable"),
        ))),
        "replay"
    )]
    #[case::sts_rejection(
        web_identity_failure(litellm_auth_aws::Error::MissingWebIdentityCredentials),
        "replay"
    )]
    #[case::assume_role(
        litellm_auth_aws::Error::Credentials {
            flow: litellm_auth_aws::CredentialFlow::AssumeRole,
            source: Source::new(aws_credential_types::provider::error::CredentialsError::provider_error(
                std::io::Error::other("AccessDenied"),
            )),
        },
        "replay"
    )]
    #[case::empty_profile(
        litellm_auth_aws::Error::Credentials {
            flow: litellm_auth_aws::CredentialFlow::Profile,
            source: Source::new(aws_credential_types::provider::error::CredentialsError::not_loaded(
                "profile not found",
            )),
        },
        "replay"
    )]
    #[case::missing_profile(
        litellm_auth_aws::Error::ProfileNotFound { profile: "absent".into() },
        "profile_not_found"
    )]
    #[case::expired(litellm_auth_aws::Error::ExpiredCredentials, "replay")]
    fn an_aws_credential_failure_names_what_python_signing_raises(
        #[case] failure: litellm_auth_aws::Error,
        #[case] expected: &str,
        #[values(false, true)] through_llm: bool,
    ) {
        let auth = litellm_auth::Error::from(failure);
        let error = match through_llm {
            true => Error::Llm(litellm_llms::Error::Auth(auth)),
            false => Error::Auth(auth),
        };
        Python::initialize();
        Python::attach(|py| {
            let native = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            assert!(native.is_instance_of::<PyRuntimeError>(py));
            let marked = native
                .value(py)
                .getattr(MESSAGES_CREDENTIAL_FAILURE)
                .unwrap()
                .extract::<String>()
                .unwrap();
            assert_eq!(marked, expected);
            let profile = native
                .value(py)
                .getattr_opt(MESSAGES_MISSING_PROFILE)
                .unwrap()
                .map(|value| value.extract::<String>().unwrap());
            assert_eq!(
                profile.as_deref(),
                (expected == "profile_not_found").then_some("absent")
            );
        });
    }

    #[rstest::rstest]
    #[case::acquisition(
        litellm_auth::Error::CredentialAcquisition(litellm_auth::ErrorDetail::Empty {
            subject: "token"
        }),
        Some("replay")
    )]
    #[case::chain(
        litellm_auth::Error::CredentialChain {
            last: Box::new(litellm_auth::Error::CredentialAcquisition(
                litellm_auth::ErrorDetail::Empty { subject: "token" },
            )),
            earlier: Vec::new(),
        },
        Some("replay")
    )]
    #[case::missing_key(
        litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        },
        None
    )]
    #[case::invalid_header(litellm_auth::Error::InvalidHeader, None)]
    fn only_credential_acquisition_failures_carry_the_credential_marker(
        #[case] auth: litellm_auth::Error,
        #[case] expected: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let unmarked =
                route_failure(py, Error::Auth(auth.clone()), RequestMarker::Unmarked, Ok).unwrap();
            assert!(
                unmarked
                    .value(py)
                    .getattr_opt(MESSAGES_CREDENTIAL_FAILURE)
                    .unwrap()
                    .is_none()
            );
            let native = route_failure(py, Error::Auth(auth), RequestMarker::Messages, Ok).unwrap();
            let marked = native
                .value(py)
                .getattr_opt(MESSAGES_CREDENTIAL_FAILURE)
                .unwrap()
                .map(|value| value.extract::<String>().unwrap());
            assert_eq!(marked.as_deref(), expected);
        });
    }

    type Marker = (String, String);

    #[rstest::rstest]
    #[case::plain(
        GoogleLoginFailure::new(PythonException::Refresh, "invalid_grant"),
        ("google.auth.exceptions.RefreshError".into(), "invalid_grant".into())
    )]
    #[case::file_failure(
        GoogleLoginFailure::new(PythonException::Value, "file failure"),
        ("ValueError".into(), "file failure".into())
    )]
    fn a_google_login_failure_names_python_s_exception(
        #[case] failure: GoogleLoginFailure,
        #[case] expected: Marker,
        #[values(false, true)] through_llm: bool,
    ) {
        let error = match through_llm {
            true => Error::Llm(litellm_llms::Error::Auth(failure.into())),
            false => Error::Auth(failure.into()),
        };
        Python::initialize();
        Python::attach(|py| {
            let raised = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            let named: Marker = raised
                .value(py)
                .getattr(MESSAGES_GOOGLE_LOGIN_FAILURE)
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(named, expected);
        });
    }

    #[test]
    fn a_wrapped_credential_failure_keeps_the_native_mapping() {
        let error = Error::post_call(Error::Auth(litellm_auth::Error::from(
            litellm_auth_aws::Error::WebIdentityTokenReference,
        )));
        Python::initialize();
        Python::attach(|py| {
            let native = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            assert!(
                native
                    .value(py)
                    .getattr_opt(MESSAGES_CREDENTIAL_FAILURE)
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn an_unorderable_comparison_raises_python_type_error_as_it_is() {
        Python::initialize();
        Python::attach(|py| {
            let error = Error::invalid_request(litellm_llms::ErrorDetail::UnorderableTypes {
                op: "<=",
                left: "str",
                right: "int",
            });
            let native = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            assert!(native.is_instance_of::<PyTypeError>(py));
            let value = native.value(py);
            assert_eq!(
                value.str().unwrap().to_string(),
                "'<=' not supported between instances of 'str' and 'int'"
            );
            assert!(value.getattr_opt(MESSAGES_PYTHON_RAISED).unwrap().is_some());
            assert!(
                value
                    .getattr_opt(MESSAGES_LATE_REJECTION)
                    .unwrap()
                    .is_some()
            );
            assert!(value.getattr_opt(MESSAGES_REQUEST_ERROR).unwrap().is_none());
        });
    }

    #[rstest::rstest]
    #[case::unbillable_usage(
        Error::invalid_response(litellm_llms::ErrorDetail::UnbillableUsage {
            reason: "got an array".into(),
        }),
        true
    )]
    #[case::other_invalid_response(
        Error::invalid_response(litellm_llms::ErrorDetail::MissingAzureApiBase),
        false
    )]
    #[case::rejected_request(
        Error::invalid_request(litellm_llms::ErrorDetail::UnbillableUsage {
            reason: "got an array".into(),
        }),
        false
    )]
    fn only_an_unbillable_reply_is_marked_refused(#[case] error: Error, #[case] refused: bool) {
        Python::initialize();
        Python::attach(|py| {
            let native = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            let marked = native
                .value(py)
                .getattr_opt(MESSAGES_REFUSED_REPLY)
                .unwrap()
                .is_some_and(|value| value.extract::<bool>().unwrap());
            assert_eq!(marked, refused);
        });
    }

    #[rstest::rstest]
    #[case::rejected_request(
        Error::invalid_request(litellm_llms::ErrorDetail::UnsupportedParameter {
            model: "claude".into(),
            param: "top_k".into(),
            value: "5".into(),
            hint: String::new(),
        }),
        Some("claude does not support top_k=5. To drop unsupported params, set `litellm.drop_params = True`.")
    )]
    #[case::missing_field(
        litellm_llms::Error::MissingField("max_tokens").into(),
        Some("missing required field: max_tokens")
    )]
    #[case::unresolvable_provider(Error::InvalidProvider("openai".into()), None)]
    #[case::invalid_response(
        Error::invalid_response(litellm_llms::ErrorDetail::MissingField {
            subject: "response",
            field: "usage",
        }),
        None
    )]
    fn only_messages_request_rejections_carry_the_marker_and_python_text(
        #[case] error: Error,
        #[case] rejection: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let unmarked = route_failure(py, error.clone(), RequestMarker::Unmarked, Ok).unwrap();
            assert!(
                unmarked
                    .value(py)
                    .getattr_opt(MESSAGES_REQUEST_ERROR)
                    .unwrap()
                    .is_none()
            );
            let native = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            let marked = native
                .value(py)
                .getattr_opt(MESSAGES_REQUEST_ERROR)
                .unwrap()
                .map(|value| value.extract::<bool>().unwrap());
            assert_eq!(marked.unwrap_or(false), rejection.is_some());
            if let Some(rejection) = rejection {
                assert!(native.is_instance_of::<PyValueError>(py));
                assert_eq!(native.value(py).to_string(), rejection);
            }
        });
    }

    #[rstest::rstest]
    #[case::vertex_version(
        litellm_llms::ErrorDetail::UnsupportedVertexAnthropicVersion {
            model: "claude-sonnet-4-5".into(),
            value: "latest".into(),
        },
        Some("claude-sonnet-4-5")
    )]
    #[case::vertex_output_format(
        litellm_llms::ErrorDetail::NonObjectVertexOutputFormat {
            model: "claude-sonnet-4-5".into(),
            actual: "str",
        },
        None
    )]
    #[case::shared_gate(
        litellm_llms::ErrorDetail::UnsupportedParameter {
            model: "claude-sonnet-4-5".into(),
            param: "top_k".into(),
            value: "5".into(),
            hint: String::new(),
        },
        None
    )]
    fn vertex_transform_refusals_are_late_unsupported_params(
        #[case] detail: litellm_llms::ErrorDetail,
        #[case] attributed_model: Option<&str>,
    ) {
        let late = !matches!(
            detail,
            litellm_llms::ErrorDetail::UnsupportedParameter { .. }
        );
        Python::initialize();
        Python::attach(|py| {
            let raised = route_failure(
                py,
                Error::invalid_request(detail),
                RequestMarker::Messages,
                Ok,
            )
            .unwrap();
            let value = raised.value(py);
            let flag = |name: &str| {
                value
                    .getattr_opt(name)
                    .unwrap()
                    .is_some_and(|flag| flag.extract::<bool>().unwrap())
            };
            assert!(flag(MESSAGES_UNSUPPORTED_PARAM));
            assert_eq!(flag(MESSAGES_LATE_REJECTION), late);
            assert_eq!(
                value
                    .getattr_opt(MESSAGES_UNSUPPORTED_PARAM_MODEL)
                    .unwrap()
                    .map(|model| model.extract::<String>().unwrap())
                    .as_deref(),
                attributed_model
            );
        });
    }

    #[rstest::rstest]
    #[case::provider_resolution(litellm_llms::MantleRegionCheck::ProviderResolution, true)]
    #[case::request(litellm_llms::MantleRegionCheck::Request, false)]
    fn an_invalid_mantle_region_is_flagged_where_python_raises_it(
        #[case] check: litellm_llms::MantleRegionCheck,
        #[case] resolving_provider: bool,
    ) {
        let error = Error::invalid_request(litellm_llms::ErrorDetail::InvalidMantleRegion {
            region: "US-EAST-1".into(),
            check,
        });
        Python::initialize();
        Python::attach(|py| {
            let raised = route_failure(py, error, RequestMarker::Messages, Ok).unwrap();
            assert!(raised.is_instance_of::<PyValueError>(py));
            let value = raised.value(py);
            assert_eq!(
                value.str().unwrap().to_string(),
                "Invalid AWS region format: 'US-EAST-1'. Region names must contain only lowercase letters, digits, and hyphens."
            );
            let flagged = |name: &str| value.getattr_opt(name).unwrap().is_some();
            assert_eq!(flagged(MESSAGES_REQUEST_ERROR), resolving_provider);
            assert_eq!(flagged(MESSAGES_PYTHON_RAISED), !resolving_provider);
            assert_eq!(flagged(MESSAGES_LATE_REJECTION), !resolving_provider);
        });
    }

    #[test]
    fn a_python_secret_manager_exception_skips_the_public_mapping() {
        Python::initialize();
        Python::attach(|py| {
            let raised = pyo3::exceptions::PyKeyError::new_err("vault down");
            let identity = raised.value(py).clone().unbind();
            let error = Error::from(crate::secrets::read_error(py, raised));
            let surfaced = route_failure(py, error, RequestMarker::Unmarked, |_| {
                panic!("the raised exception must not be remapped")
            })
            .unwrap();
            assert!(surfaced.value(py).is(identity.bind(py)));
        });
    }
}
