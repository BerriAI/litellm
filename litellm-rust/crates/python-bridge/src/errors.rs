use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

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
    "The provider call was already issued and failed. Args are (status, message) and `headers` holds the provider's (name, value) pairs; a timeout is reported as 408 with no headers."
);

pub(crate) fn route_error_to_pyerr(error: RouteError) -> PyErr {
    match &error {
        RouteError::Rejected(rejected) => {
            Python::attach(|py| rejected_to_pyerr(py, rejected).unwrap_or_else(|error| error))
        }
        RouteError::Transport(TransportError::Timeout(_)) => {
            Python::attach(|py| timeout_to_pyerr(py, &error).unwrap_or_else(|error| error))
        }
        other => by_fault(other.is_request(), other.to_string()),
    }
}

/// `RustUpstreamError(status, body)` with the provider's headers as `(name, value)` pairs.
pub(crate) fn upstream_error(
    py: Python<'_>,
    status: u16,
    body: String,
    headers: Vec<(String, String)>,
) -> PyResult<PyErr> {
    let error = RustUpstreamError::new_err((status, body));
    error.value(py).setattr("headers", headers)?;
    Ok(error)
}

pub(crate) fn rejected_to_pyerr(
    py: Python<'_>,
    rejected: &litellm_http::response::Rejected,
) -> PyResult<PyErr> {
    upstream_error(
        py,
        rejected.status().as_u16(),
        rejected.text().into_owned(),
        litellm_http::response::header_pairs(rejected.headers()),
    )
}

/// A timeout reaches Python as the 408 its exception mapping already understands.
pub(crate) fn timeout_to_pyerr(py: Python<'_>, error: &impl ToString) -> PyResult<PyErr> {
    upstream_error(py, 408, error.to_string(), Vec::new())
}

/// A test reply the provider rejected.
#[cfg(test)]
pub(crate) fn rejected(
    status: u16,
    body: &str,
    headers: &[(&str, &str)],
) -> litellm_http::response::Rejected {
    let response = headers
        .iter()
        .fold(
            ::http::Response::builder().status(status),
            |builder, (name, value)| builder.header(*name, *value),
        )
        .body(bytes::Bytes::from(body.to_owned()))
        .unwrap();
    litellm_http::response::Rejected::new(response)
}

/// A request the caller got wrong is a `ValueError`; anything else is a `RuntimeError`.
pub(crate) fn by_fault(is_request: bool, message: String) -> PyErr {
    if is_request {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::unsupported(RouteError::Unsupported("test capability"), true)]
    #[case::invalid_provider(RouteError::InvalidProvider("unknown".into()), true)]
    #[case::invalid_request(RouteError::InvalidRequest("empty messages".into()), true)]
    #[case::connection(TransportError::Connect("unreachable".into()).into(), false)]
    #[case::network(TransportError::Network("reset".into()).into(), false)]
    #[case::invalid_response(RouteError::InvalidResponse("missing usage".into()), false)]
    fn route_failures_are_terminal(#[case] error: RouteError, #[case] is_request: bool) {
        Python::initialize();
        Python::attach(|py| {
            let failure = route_error_to_pyerr(error);
            assert!(!failure.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(failure.is_instance_of::<PyValueError>(py), is_request);
            assert_eq!(failure.is_instance_of::<PyRuntimeError>(py), !is_request);
        });
    }

    #[rstest::rstest]
    fn transport_status_body_and_headers_survive_python_mapping() {
        Python::initialize();
        Python::attach(|py| {
            let upstream = route_error_to_pyerr(
                rejected(
                    429,
                    "slow down",
                    &[
                        ("Retry-After", "17"),
                        ("x-provider-trace", "first"),
                        ("x-provider-trace", "second"),
                    ],
                )
                .into(),
            );
            assert!(upstream.is_instance_of::<RustUpstreamError>(py));
            assert_eq!(
                upstream
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(u16, String)>()
                    .unwrap(),
                (429, "slow down".into())
            );
            assert_eq!(
                upstream
                    .value(py)
                    .getattr("headers")
                    .unwrap()
                    .extract::<Vec<(String, String)>>()
                    .unwrap(),
                vec![
                    ("retry-after".into(), "17".into()),
                    ("x-provider-trace".into(), "first".into()),
                    ("x-provider-trace".into(), "second".into())
                ]
            );
        });
    }

    #[rstest::rstest]
    fn a_timeout_is_a_408_without_headers() {
        Python::initialize();
        Python::attach(|py| {
            let timeout = route_error_to_pyerr(TransportError::Timeout("deadline".into()).into());
            assert!(timeout.is_instance_of::<RustUpstreamError>(py));
            let (status, message): (u16, String) = timeout
                .value(py)
                .getattr("args")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(status, 408);
            assert!(message.contains("deadline"));
            assert!(
                timeout
                    .value(py)
                    .getattr("headers")
                    .unwrap()
                    .extract::<Vec<(String, String)>>()
                    .unwrap()
                    .is_empty()
            );
        });
    }

    #[test]
    fn missing_api_key_stays_a_runtime_error_while_other_auth_failures_are_value_errors() {
        Python::initialize();
        Python::attach(|py| {
            let missing =
                route_error_to_pyerr(RouteError::Auth(litellm_auth::Error::MissingApiKey {
                    provider: "Anthropic",
                    environment_variable: "ANTHROPIC_API_KEY",
                }));
            assert!(missing.is_instance_of::<PyRuntimeError>(py));
            let invalid =
                route_error_to_pyerr(RouteError::Auth(litellm_auth::Error::InvalidHeader));
            assert!(invalid.is_instance_of::<PyValueError>(py));
        });
    }
}
