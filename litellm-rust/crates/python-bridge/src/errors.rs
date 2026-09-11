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

#[derive(Debug)]
pub(crate) enum BridgeError {
    InvalidArgument(String),
    Declined(String),
    Upstream {
        status: Option<u16>,
        message: String,
    },
    Internal(String),
    Host(PyErr),
}

impl From<BridgeError> for PyErr {
    fn from(error: BridgeError) -> Self {
        match error {
            BridgeError::InvalidArgument(message) => PyValueError::new_err(message),
            BridgeError::Declined(reason) => RustBridgeDeclined::new_err(reason),
            BridgeError::Upstream { status, message } => {
                RustUpstreamError::new_err((status.unwrap_or(0), message))
            }
            BridgeError::Internal(message) => PyRuntimeError::new_err(message),
            BridgeError::Host(error) => error,
        }
    }
}

pub(crate) fn required_route_error(err: Error) -> BridgeError {
    match err {
        Error::Auth(message) => BridgeError::InvalidArgument(message),
        Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_) => BridgeError::InvalidArgument(err.to_string()),
        other => BridgeError::Internal(other.to_string()),
    }
}

pub(crate) fn fallback_route_error(err: Error) -> BridgeError {
    match err {
        Error::Unsupported(_)
        | Error::Auth(_)
        | Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_)
        | Error::MissingApiKey { .. }
        | Error::MissingAzureAiCredentials
        | Error::MissingAzureDocumentIntelligenceCredentials
        | Error::MissingReductoApiKey
        | Error::Routing(_)
        | Error::Connect(_) => BridgeError::Declined(err.to_string()),
        Error::Http { status, body } => BridgeError::Upstream {
            status: Some(status),
            message: format!("{status}: {body}"),
        },
        Error::Network(message) | Error::InvalidResponse(message) => BridgeError::Upstream {
            status: None,
            message,
        },
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}

pub(crate) fn ocr_route_error(err: Error) -> BridgeError {
    match err {
        Error::MissingField("document_url" | "image_url") => {
            BridgeError::InvalidArgument("Document URL is required".into())
        }
        Error::Http { status, body } => BridgeError::Upstream {
            status: Some(status),
            message: body,
        },
        other => required_route_error(other),
    }
}

#[cfg(test)]
mod ocr_error_tests {
    use super::*;

    #[test]
    fn fallback_policy_never_declines_possible_dispatch() {
        for error in [
            Error::Network("timeout".into()),
            Error::InvalidResponse("malformed".into()),
            Error::Http {
                status: 429,
                body: "limited".into(),
            },
        ] {
            assert!(matches!(
                fallback_route_error(error),
                BridgeError::Upstream { .. }
            ));
        }
        for error in [
            Error::Connect("offline".into()),
            Error::Unsupported("shape"),
            Error::MissingApiKey { provider: "test" },
        ] {
            assert!(matches!(
                fallback_route_error(error),
                BridgeError::Declined(_)
            ));
        }
    }

    #[test]
    fn conversion_preserves_absent_status_and_host_exception_identity() {
        Python::initialize();
        Python::attach(|py| {
            let error: PyErr = BridgeError::Upstream {
                status: None,
                message: "timeout".into(),
            }
            .into();
            let args: (u16, String) = error.value(py).getattr("args").unwrap().extract().unwrap();
            assert_eq!(args, (0, "timeout".into()));
            let original = PyValueError::new_err("host exception");
            let retained = original.value(py).clone();
            let mapped: PyErr = BridgeError::Host(original).into();
            assert!(mapped.value(py).is(&retained));
        });
    }

    #[test]
    fn ocr_errors_preserve_provider_body() {
        Python::initialize();
        Python::attach(|py| {
            for field in ["document_url", "image_url"] {
                let mapped: PyErr = ocr_route_error(Error::MissingField(field)).into();
                assert!(mapped.is_instance_of::<PyValueError>(py));
                assert_eq!(mapped.value(py).to_string(), "Document URL is required");
            }
            let mapped: PyErr = ocr_route_error(Error::Http {
                status: 429,
                body: r#"{"message":"rate limited"}"#.to_string(),
            })
            .into();
            assert!(mapped.is_instance_of::<RustUpstreamError>(py));
            let args: (u16, String) = mapped
                .value(py)
                .getattr("args")
                .and_then(|args| args.extract())
                .expect("OCR failures retain status and unprefixed provider message");
            assert_eq!(args, (429, r#"{"message":"rate limited"}"#.to_string()));
        });
    }
}
