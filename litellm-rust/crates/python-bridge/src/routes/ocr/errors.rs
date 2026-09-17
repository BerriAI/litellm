use litellm_core::ocr::Error;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{RustUpstreamError, core_error_to_pyerr};

pub(super) fn to_pyerr(error: Error) -> PyErr {
    let status = error.http_status_code();
    let mapped = Python::attach(|py| -> PyResult<PyErr> {
        Ok(match error {
            Error::Provider {
                status,
                body,
                headers,
            } => upstream_error(py, status, body, headers)?,
            Error::Transport(litellm_core::transport::Error::Http { status, body }) => {
                upstream_error(py, status, body, Vec::new())?
            }
            Error::RequestFormat => {
                let error = core_error_to_pyerr(Error::RequestFormat.into());
                error
                    .value(py)
                    .setattr("ocr_request_format_error", true)
                    .ok();
                error
            }
            Error::FileRead { path, source } if source.kind() == std::io::ErrorKind::NotFound => {
                PyFileNotFoundError::new_err(format!("File not found: {}", path.display()))
            }
            Error::FileRead { source, .. } => PyOSError::new_err(source.to_string()),
            other => core_error_to_pyerr(other.into()),
        })
    })
    .unwrap_or_else(|error| error);
    attach_status(mapped, status)
}

fn upstream_error(
    py: Python<'_>,
    status: u16,
    body: String,
    headers: Vec<(String, String)>,
) -> PyResult<PyErr> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("content", &body)?;
    kwargs.set_item("headers", headers)?;
    let response = py
        .import("httpx")?
        .getattr("Response")?
        .call((status,), Some(&kwargs))?;
    let error = RustUpstreamError::new_err((status, body));
    error.value(py).setattr("response", response)?;
    Ok(error)
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
                400
            );
            let mapped = to_pyerr(Error::Provider {
                status: 429,
                body: r#"{"message":"rate limited"}"#.to_string(),
                headers: Vec::new(),
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
