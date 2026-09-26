use litellm_core::{Phase, RouteError};
use litellm_http::transport::Error as TransportError;
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
    "The provider call was already issued and failed. Args are (status, message); status is 0 when there was no HTTP response."
);

pub(crate) fn route_error_to_pyerr(error: RouteError) -> PyErr {
    by_fault(error.is_request(), error.to_string())
}

/// A request the caller got wrong is a `ValueError`; anything else is a `RuntimeError`.
pub(crate) fn by_fault(is_request: bool, message: String) -> PyErr {
    if is_request {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}

/// Map a route error for a route whose host keeps a Python implementation.
///
/// The distinction the host needs is whether the provider was already called.
/// Everything raised before the request goes out is safe for the host to retry
/// on its own path; anything after it is not, because the provider has already
/// done the work and billed for it.
pub(crate) fn chat_completions_error_to_pyerr(error: RouteError) -> PyErr {
    match error.phase() {
        Phase::BeforeSend => RustBridgeDeclined::new_err(error.to_string()),
        Phase::AfterSend => RustUpstreamError::new_err(match error {
            RouteError::Transport(TransportError::Http { status, body }) => (status, body),
            RouteError::Transport(TransportError::Network(message))
            | RouteError::InvalidResponse(message) => (0u16, message),
            other => (0u16, other.to_string()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_status_and_dispatch_certainty_survive_python_mapping() {
        Python::initialize();
        Python::attach(|py| {
            let connect = chat_completions_error_to_pyerr(
                TransportError::Connect("unreachable".into()).into(),
            );
            assert!(connect.is_instance_of::<RustBridgeDeclined>(py));
            let network =
                chat_completions_error_to_pyerr(TransportError::Network("timed out".into()).into());
            assert!(network.is_instance_of::<RustUpstreamError>(py));
            let upstream = chat_completions_error_to_pyerr(
                TransportError::Http {
                    status: 429,
                    body: "slow down".into(),
                }
                .into(),
            );
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
