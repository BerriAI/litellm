use litellm_core::error::Error;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    _native,
    RustBridgeDeclined,
    pyo3::exceptions::PyException,
    "The route declined before calling the provider, so the host may retry on its own path."
);

pub(crate) fn declined(error: impl std::fmt::Display) -> PyErr {
    RustBridgeDeclined::new_err(error.to_string())
}

pyo3::create_exception!(
    _native,
    RustUpstreamError,
    pyo3::exceptions::PyException,
    "The provider call was already issued and failed. Args are (status, message); status is 0 when there was no HTTP response."
);

pub(crate) fn core_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Auth(message) => PyValueError::new_err(message),
        Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_) => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Map a core error for a route whose host keeps a Python implementation.
///
/// The distinction the host needs is whether the provider was already called.
/// Everything raised before the request goes out is safe for the host to retry
/// on its own path; anything after it is not, because the provider has already
/// done the work and billed for it.
pub(crate) fn fallback_route_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Unsupported(_)
        | Error::Auth(_)
        | Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_)
        | Error::Routing(_)
        // Nothing reached the provider, so serving it on Python cannot double
        // bill and is the only way the caller gets an answer at all.
        | Error::Connect(_) => RustBridgeDeclined::new_err(err.to_string()),
        Error::Http { status, body } => {
            RustUpstreamError::new_err((status, format!("{status}: {body}")))
        }
        Error::Network(message) | Error::InvalidResponse(message) => {
            RustUpstreamError::new_err((0u16, message))
        }
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
            let declines = [
                Error::Unsupported("unsupported"),
                Error::Auth("missing key".to_string()),
                Error::InvalidProvider("unsupported".to_string()),
                Error::InvalidRequest("invalid".to_string()),
                Error::InvalidType {
                    expected: "string",
                    actual: "number",
                },
                Error::MissingField("model"),
                Error::Routing("no route".to_string()),
                Error::Connect("connection refused".to_string()),
            ];
            for error in declines {
                let mapped = fallback_route_error_to_pyerr(error);
                assert!(mapped.is_instance_of::<RustBridgeDeclined>(py));
            }

            let upstream_failures = [
                (
                    Error::Http {
                        status: 429,
                        body: "rate limited".to_string(),
                    },
                    (429, "429: rate limited"),
                ),
                (
                    Error::Network("request timed out".to_string()),
                    (0, "request timed out"),
                ),
                (
                    Error::InvalidResponse("bad JSON".to_string()),
                    (0, "bad JSON"),
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
