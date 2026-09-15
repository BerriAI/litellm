use litellm_core::ocr::Error;
use litellm_core::transport::Error as TransportError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::errors::{RustUpstreamError, core_error_to_pyerr};

pub(super) fn to_pyerr(error: Error) -> PyErr {
    let (mapped, status) = match error {
        Error::MissingDocumentUrl => (
            PyValueError::new_err(Error::MissingDocumentUrl.to_string()),
            Some(500),
        ),
        error @ Error::MissingField(_) => (PyValueError::new_err(error.to_string()), None),
        error if error.is_request() => (
            PyValueError::new_err(format!("invalid request: {error}")),
            Some(400),
        ),
        error if error.is_response() => (
            PyRuntimeError::new_err(format!("invalid response: {error}")),
            None,
        ),
        error @ (Error::InvalidRequest(_) | Error::Params(_) | Error::Headers(_)) => {
            (PyValueError::new_err(error.to_string()), Some(400))
        }
        Error::Transport(TransportError::Http { status, body }) => {
            (RustUpstreamError::new_err((status, body)), Some(status))
        }
        other => (core_error_to_pyerr(other), None),
    };
    attach_status(mapped, status)
}

fn attach_status(error: PyErr, status: Option<u16>) -> PyErr {
    if let Some(status) = status {
        Python::attach(|py| {
            let value = error.value(py);
            value.setattr("status_code", status).ok();
            value.setattr("message", value.to_string()).ok();
        });
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;

    #[test]
    fn preserves_python_validation_and_provider_details() {
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(Error::MissingDocumentUrl);
            assert!(mapped.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert_eq!(mapped.value(py).to_string(), "Document URL is required");
            assert_eq!(
                mapped
                    .value(py)
                    .getattr("status_code")
                    .unwrap()
                    .extract::<u16>()
                    .unwrap(),
                500
            );
            let mapped = to_pyerr(Error::Transport(litellm_core::transport::Error::Http {
                status: 429,
                body: r#"{"message":"rate limited"}"#.to_string(),
            }));
            assert!(mapped.is_instance_of::<RustUpstreamError>(py));
            let args: (u16, String) = mapped
                .value(py)
                .getattr("args")
                .and_then(|args| args.extract())
                .expect("OCR failures retain status and unprefixed provider message");
            assert_eq!(args, (429, r#"{"message":"rate limited"}"#.to_string()));

            let mapped = to_pyerr(Error::InvalidRequest("invalid format".into()));
            assert!(mapped.is_instance_of::<PyValueError>(py));
            assert_eq!(
                mapped
                    .value(py)
                    .getattr("status_code")
                    .unwrap()
                    .extract::<u16>()
                    .unwrap(),
                400
            );
        });
    }

    #[test]
    fn typed_request_and_response_failures_keep_python_contracts() {
        Python::initialize();
        Python::attach(|py| {
            let request = to_pyerr(Error::RequestField {
                path: "document.type".into(),
            });
            assert!(request.is_instance_of::<PyValueError>(py));
            assert_eq!(
                request
                    .value(py)
                    .getattr("status_code")
                    .unwrap()
                    .extract::<u16>()
                    .unwrap(),
                400
            );
            assert_eq!(
                request.value(py).to_string(),
                "invalid request: invalid OCR request field: document.type"
            );
            let response = to_pyerr(Error::EmptyContent);
            assert!(response.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                response.value(py).to_string(),
                "invalid response: OCR response is missing non-empty content"
            );
            assert!(!response.value(py).hasattr("status_code").unwrap());
        });
    }
}
