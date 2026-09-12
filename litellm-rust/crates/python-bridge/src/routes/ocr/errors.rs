use litellm_core::error::Error;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::{RustUpstreamError, core_error_to_pyerr};

pub(super) fn to_pyerr(error: Error) -> PyErr {
    match error {
        Error::MissingField("document_url" | "image_url") => {
            PyValueError::new_err("Document URL is required")
        }
        Error::Http { status, body } => upstream_error(status, body),
        Error::Network(message) if message.contains("timed out") => upstream_error(408, message),
        other => {
            let status = other.http_status_code();
            let error = core_error_to_pyerr(other);
            if let Some(status) = status {
                Python::attach(|py| {
                    let value = error.value(py);
                    value.setattr("status_code", status).ok();
                    value.setattr("message", value.to_string()).ok();
                });
            }
            error
        }
    }
}

fn upstream_error(status: u16, message: String) -> PyErr {
    let error = RustUpstreamError::new_err((status, message.clone()));
    Python::attach(|py| {
        let value = error.value(py);
        value.setattr("status_code", status).ok();
        value.setattr("message", message).ok();
    });
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_python_validation_and_provider_details() {
        Python::initialize();
        Python::attach(|py| {
            for field in ["document_url", "image_url"] {
                let mapped = to_pyerr(Error::MissingField(field));
                assert!(mapped.is_instance_of::<PyValueError>(py));
                assert_eq!(mapped.value(py).to_string(), "Document URL is required");
            }
            let mapped = to_pyerr(Error::Http {
                status: 429,
                body: r#"{"message":"rate limited"}"#.to_string(),
            });
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
}
