use std::error::Error as StdError;

use litellm_error::Report;
use litellm_inference::Error;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
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

/// Maps a route failure once. A Python exception raised inside the call is returned as it is,
/// everything else is mapped and handed to `public`, the route's public exception mapping.
pub(crate) fn route_failure(
    py: Python<'_>,
    error: Error,
    marker: RequestMarker,
    public: impl FnOnce(PyErr) -> PyResult<PyErr>,
) -> PyResult<PyErr> {
    if let Some(raised) = crate::secrets::python_error(py, &error) {
        return Ok(raised);
    }
    if let Some((status, body)) = error.upstream_response() {
        return public(upstream_error(py, status, body, &HeaderMap::new())?);
    }
    let native = match (marker, error.rejected_request()) {
        (RequestMarker::Messages, Some(rejection)) => {
            let native = PyValueError::new_err(Report::new(rejection).to_string());
            native.value(py).setattr(MESSAGES_REQUEST_ERROR, true)?;
            native
        }
        (RequestMarker::Messages, None) | (RequestMarker::Unmarked, _) => {
            by_fault(error.is_request(), &error)
        }
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
    #[case::unsupported(Error::Unsupported("test capability"), true)]
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
