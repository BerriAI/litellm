use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

pyo3::create_exception!(
    _native,
    RustUpstreamError,
    pyo3::exceptions::PyException,
    "The provider call was already issued and failed. Args are (status, message); status is 0 when there was no HTTP response."
);

pub(crate) fn route_error_to_pyerr(error: RouteError) -> PyErr {
    match error {
        RouteError::Transport(TransportError::Http { status, body }) => {
            RustUpstreamError::new_err((status, body))
        }
        other => by_fault(other.is_request(), other.to_string()),
    }
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
    #[case::network(TransportError::Network("timed out".into()).into(), false)]
    #[case::invalid_response(RouteError::InvalidResponse("missing usage".into()), false)]
    fn route_failures_are_terminal(#[case] error: RouteError, #[case] is_request: bool) {
        Python::initialize();
        Python::attach(|py| {
            let failure = route_error_to_pyerr(error);
            assert_eq!(failure.is_instance_of::<PyValueError>(py), is_request);
            assert_eq!(failure.is_instance_of::<PyRuntimeError>(py), !is_request);
        });
    }

    #[rstest::rstest]
    fn transport_status_survives_python_mapping() {
        Python::initialize();
        Python::attach(|py| {
            let upstream = route_error_to_pyerr(
                TransportError::Http {
                    status: 429,
                    body: "slow down".into(),
                }
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
