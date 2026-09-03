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
        // Declined before the provider was called: the host may fall back to
        // its own path without double billing.
        Error::Overloaded(_) => RustBridgeDeclined::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Map a core error for a route whose host keeps a Python implementation.
///
/// The distinction the host needs is whether the provider was already called.
/// Everything raised before the request goes out is safe for the host to retry
/// on its own path; anything after it is not, because the provider has already
/// done the work and billed for it.
pub(crate) fn chat_completions_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Unsupported(_)
        | Error::Auth(_)
        | Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_)
        | Error::Routing(_)
        | Error::Overloaded(_)
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
    fn overloaded_maps_to_declined_for_both_routes() {
        Python::initialize();
        Python::attach(|py| {
            let mapped = [
                core_error_to_pyerr(Error::Overloaded(
                    "native in-flight limit reached".to_string(),
                )),
                chat_completions_error_to_pyerr(Error::Overloaded(
                    "native in-flight limit reached".to_string(),
                )),
            ];
            for mapped in mapped {
                assert!(mapped.is_instance_of::<RustBridgeDeclined>(py));
            }
        });
    }
}
