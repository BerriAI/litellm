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
        Error::InvalidProvider(_) => PyValueError::new_err("Invalid provider configuration"),
        Error::InvalidRequest(_) => PyValueError::new_err("Invalid provider request"),
        Error::InvalidType { .. } | Error::MissingField(_) => {
            PyValueError::new_err(err.to_string())
        }
        Error::Auth(_) => PyValueError::new_err("Provider authentication failed"),
        Error::Http { status, .. } => {
            PyRuntimeError::new_err(format!("Provider request failed (HTTP {status})"))
        }
        Error::Network(_) | Error::Connect(_) => {
            PyRuntimeError::new_err("Provider transport failed")
        }
        Error::InvalidResponse(_) => PyRuntimeError::new_err("Invalid provider response"),
        Error::Routing(_) => PyRuntimeError::new_err("Provider routing failed"),
        Error::Unsupported(_) => PyRuntimeError::new_err("Operation is not supported"),
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
        Error::Unsupported(_) => RustBridgeDeclined::new_err("Operation is not supported"),
        Error::Auth(_) => RustBridgeDeclined::new_err("Provider authentication failed"),
        Error::InvalidProvider(_) => RustBridgeDeclined::new_err("Invalid provider configuration"),
        Error::InvalidRequest(_) => RustBridgeDeclined::new_err("Invalid provider request"),
        Error::InvalidType { .. } | Error::MissingField(_) => {
            RustBridgeDeclined::new_err(err.to_string())
        }
        Error::Routing(_) => RustBridgeDeclined::new_err("Provider routing failed"),
        Error::Connect(_) => RustBridgeDeclined::new_err("Could not reach provider"),
        Error::Http { status, .. } => {
            RustUpstreamError::new_err((status, format!("Provider request failed (HTTP {status})")))
        }
        Error::Network(_) => RustUpstreamError::new_err((0u16, "Provider transport failed")),
        Error::InvalidResponse(_) => {
            RustUpstreamError::new_err((0u16, "Invalid provider response"))
        }
    }
}

pub(crate) fn messages_provider_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Http { status, .. } => {
            RustUpstreamError::new_err((status, format!("Provider request failed (HTTP {status})")))
        }
        Error::Network(_) | Error::Connect(_) => {
            RustUpstreamError::new_err((0u16, "Provider transport failed"))
        }
        Error::InvalidResponse(_) => {
            RustUpstreamError::new_err((0u16, "Invalid provider response"))
        }
        error => core_error_to_pyerr(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_error_mappers_do_not_expose_private_details() {
        Python::initialize();
        Python::attach(|_| {
            for error in [
                Error::Auth("credential secret".into()),
                Error::Http {
                    status: 500,
                    body: "response secret".into(),
                },
                Error::Network("network secret".into()),
                Error::Connect("connection secret".into()),
                Error::InvalidResponse("parse secret".into()),
                Error::Routing("routing secret".into()),
            ] {
                assert!(!core_error_to_pyerr(error).to_string().contains("secret"));
            }

            for error in [
                Error::Auth("credential secret".into()),
                Error::Http {
                    status: 500,
                    body: "response secret".into(),
                },
                Error::Network("network secret".into()),
                Error::Connect("connection secret".into()),
                Error::InvalidResponse("parse secret".into()),
                Error::Routing("routing secret".into()),
            ] {
                assert!(
                    !chat_completions_error_to_pyerr(error)
                        .to_string()
                        .contains("secret")
                );
            }
        });
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}
