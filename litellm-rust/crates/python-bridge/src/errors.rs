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
/// The distinction the host needs is whether the provider was already called.
/// Everything raised before the request goes out is safe for the host to retry
/// on its own path; anything after it is not, because the provider has already
/// done the work and billed for it.
pub(crate) fn fallback_route_error_to_pyerr(err: CoreError) -> PyErr {
    match err {
        CoreError::Unsupported(_)
        | CoreError::Auth(_)
        | CoreError::InvalidProvider(_)
        | CoreError::InvalidRequest(_)
        | CoreError::InvalidType { .. }
        | CoreError::MissingField(_)
        | CoreError::Routing(_)
        // Nothing reached the provider, so serving it on Python cannot double
        // bill and is the only way the caller gets an answer at all.
        | CoreError::Connect(_) => RustBridgeDeclined::new_err(err.to_string()),
        CoreError::Http { status, body } => {
            RustUpstreamError::new_err((status, format!("{status}: {body}")))
        }
        CoreError::Network(message) | CoreError::InvalidResponse(message) => {
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
                CoreError::Unsupported("unsupported"),
                CoreError::Auth("missing key".to_string()),
                CoreError::InvalidProvider("unsupported".to_string()),
                CoreError::InvalidRequest("invalid".to_string()),
                CoreError::InvalidType {
                    expected: "string",
                    actual: "number",
                },
                CoreError::MissingField("model"),
                CoreError::Routing("no route".to_string()),
                CoreError::Connect("connection refused".to_string()),
            ];
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
                    (0, "request timed out"),
                ),
                (
                    CoreError::InvalidResponse("bad JSON".to_string()),
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
