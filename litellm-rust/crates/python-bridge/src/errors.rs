use litellm_core::error::Error;
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

pub(crate) fn fallback_route_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Unsupported(reason) => RustBridgeDeclined::new_err(reason),
        validation @ (Error::Auth(_)
        | Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_)) => core_error_to_pyerr(validation),
        Error::Routing(message) => PyRuntimeError::new_err(message),
        Error::Http { status, body } => {
            RustUpstreamError::new_err((status, format!("{status}: {body}")))
        }
        Error::Connect(message) | Error::Network(message) | Error::InvalidResponse(message) => {
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
    fn fallback_routes_only_decline_unsupported_requests() {
        Python::initialize();
        Python::attach(|py| {
            let declined = fallback_route_error_to_pyerr(Error::Unsupported("unsupported"));
            assert!(declined.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(
                declined
                    .value(py)
                    .getattr("args")
                    .and_then(|args| args.extract::<(String,)>())
                    .expect("decline should retain its bounded reason"),
                ("unsupported".to_string(),)
            );

            let validation_failures = [
                Error::Auth("missing key".to_string()),
                Error::InvalidProvider("unsupported".to_string()),
                Error::InvalidRequest("invalid".to_string()),
                Error::InvalidType {
                    expected: "string",
                    actual: "number",
                },
                Error::MissingField("model"),
            ];
            for error in validation_failures {
                let mapped = fallback_route_error_to_pyerr(error);
                assert!(mapped.is_instance_of::<PyValueError>(py));
            }

            let routing = fallback_route_error_to_pyerr(Error::Routing("no route".to_string()));
            assert!(routing.is_instance_of::<PyRuntimeError>(py));

            let upstream_failures = [
                (
                    Error::Http {
                        status: 429,
                        body: "rate limited".to_string(),
                    },
                    (429, "429: rate limited"),
                ),
                (
                    Error::Connect("connection refused".to_string()),
                    (0, "connection refused"),
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
