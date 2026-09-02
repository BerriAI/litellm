use litellm_core::error::CoreError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

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

pub(crate) fn core_error_to_pyerr(err: CoreError) -> PyErr {
    match err {
        CoreError::Auth(message) => PyValueError::new_err(message),
        CoreError::InvalidProvider(_)
        | CoreError::InvalidRequest(_)
        | CoreError::InvalidType { .. }
        | CoreError::MissingField(_) => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Map a core error for a route whose host keeps a Python implementation.
///
/// Only an explicit capability decline permits the host to try Python. Every
/// other error may have happened after provider dispatch and must be terminal.
pub(crate) fn fallback_route_error_to_pyerr(err: CoreError) -> PyErr {
    match err {
        CoreError::Unsupported(_) => RustBridgeDeclined::new_err(err.to_string()),
        CoreError::Http { status, body } => {
            RustUpstreamError::new_err((status, format!("{status}: {body}")))
        }
        other => RustUpstreamError::new_err((0u16, other.to_string())),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_routes_distinguish_declines_from_upstream_failures() {
        Python::initialize();
        Python::attach(|py| {
            let declines = [CoreError::Unsupported("unsupported")];
            for error in declines {
                let mapped = fallback_route_error_to_pyerr(error);
                assert!(mapped.is_instance_of::<RustBridgeDeclined>(py));
            }

            let upstream_failures = [
                (
                    CoreError::Http {
                        status: 429,
                        body: "rate limited".to_string(),
                    },
                    (429, "429: rate limited"),
                ),
                (
                    CoreError::Network("request timed out".to_string()),
                    (0, "upstream network error: request timed out"),
                ),
                (
                    CoreError::InvalidResponse("bad JSON".to_string()),
                    (0, "invalid response: bad JSON"),
                ),
                (
                    CoreError::Auth("missing key".to_string()),
                    (0, "missing key"),
                ),
                (
                    CoreError::InvalidRequest("invalid".to_string()),
                    (0, "invalid request: invalid"),
                ),
            ];
            for (error, expected) in upstream_failures {
                let mapped = fallback_route_error_to_pyerr(error);
                assert!(mapped.is_instance_of::<RustUpstreamError>(py));
                let args: (u16, String) = mapped
                    .value(py)
                    .getattr("args")
                    .and_then(|args| args.extract())
                    .expect("upstream error should carry status and message");
                assert_eq!(args, (expected.0, expected.1.to_string()));
            }
        });
    }
}
