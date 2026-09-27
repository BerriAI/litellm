use litellm_llms::base_llm::ocr::error::Error;
use pyo3::{
    exceptions::{PyFileNotFoundError, PyOSError},
    prelude::*,
};

use crate::errors::{RustUpstreamError, by_fault};

pub(super) fn to_pyerr(error: Error) -> PyErr {
    let status = error.http_status_code();
    let mapped = Python::attach(|py| -> PyResult<PyErr> {
        Ok(match error {
            Error::Provider {
                status,
                body,
                headers,
            } => upstream_error(py, status, body, headers)?,
            Error::Transport(litellm_http::transport::Error::Http { status, body }) => {
                upstream_error(py, status, body, Vec::new())?
            }
            Error::RequestFormat => {
                let error = by_fault(true, Error::RequestFormat.to_string());
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
            other => by_fault(is_request(&other), other.to_string()),
        })
    })
    .unwrap_or_else(|error| error);
    attach_status(mapped, status)
}

fn is_request(error: &Error) -> bool {
    error.is_request()
        || matches!(
            error,
            Error::Auth(_)
                | Error::InvalidProvider(_)
                | Error::InvalidRequest(_)
                | Error::MissingField(_)
                | Error::MissingDocumentUrl
        )
}

fn upstream_error(
    py: Python<'_>,
    status: u16,
    body: String,
    headers: Vec<(String, String)>,
) -> PyResult<PyErr> {
    let error = RustUpstreamError::new_err((status, body));
    error.value(py).setattr("headers", headers)?;
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
    use pyo3::exceptions::PyValueError;

    use super::*;

    #[rstest::rstest]
    #[case::abandoned(litellm_host::MachineFault::Abandoned)]
    #[case::unsupported(litellm_host::MachineFault::Unsupported("streaming"))]
    fn host_faults_are_internal_errors(#[case] fault: litellm_host::MachineFault) {
        Python::initialize();
        Python::attach(|py| {
            let error = Error::from(fault);
            assert!(!error.is_request());
            assert_eq!(error.http_status_code(), None);
            let mapped = to_pyerr(error);
            assert!(mapped.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            assert!(!mapped.value(py).hasattr("status_code").unwrap());
        });
    }

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
                headers: vec![("Retry-After".to_string(), "17".to_string())],
            });
            assert!(mapped.is_instance_of::<RustUpstreamError>(py));
            let headers: Vec<(String, String)> = mapped
                .value(py)
                .getattr("headers")
                .and_then(|headers| headers.extract())
                .expect("OCR failures retain provider headers");
            assert_eq!(headers, vec![("Retry-After".to_string(), "17".to_string())]);
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
    fn invalid_request_format_is_a_flagged_bad_request() {
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(Error::RequestFormat);
            let value = mapped.value(py);
            assert!(mapped.is_instance_of::<PyValueError>(py));
            assert!(
                value
                    .getattr("ocr_request_format_error")
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            assert_eq!(
                value
                    .getattr("status_code")
                    .unwrap()
                    .extract::<u16>()
                    .unwrap(),
                400
            );
            assert_eq!(
                value
                    .getattr("message")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                Error::RequestFormat.to_string()
            );
        });
    }

    fn file_read(kind: std::io::ErrorKind) -> Error {
        Error::FileRead {
            path: "/missing/scan.pdf".into(),
            source: std::sync::Arc::new(std::io::Error::new(kind, "disk said no")),
        }
    }

    #[test]
    fn missing_files_map_to_file_not_found_naming_the_path() {
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(file_read(std::io::ErrorKind::NotFound));
            assert!(mapped.is_instance_of::<PyFileNotFoundError>(py));
            assert_eq!(
                mapped.value(py).to_string(),
                "File not found: /missing/scan.pdf"
            );
        });
    }

    #[test]
    fn other_file_read_failures_map_to_os_error_with_the_io_message() {
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(file_read(std::io::ErrorKind::PermissionDenied));
            assert!(mapped.is_instance_of::<PyOSError>(py));
            assert!(!mapped.is_instance_of::<PyFileNotFoundError>(py));
            assert_eq!(mapped.value(py).to_string(), "disk said no");
        });
    }

    #[rstest::rstest]
    #[case::oversized(Error::TooLarge { limit: 7 })]
    #[case::malformed_field(Error::ResponseField { path: "pages[0].index".into() })]
    fn response_failures_are_statusless_runtime_errors(#[case] error: Error) {
        Python::initialize();
        Python::attach(|py| {
            let message = error.to_string();
            let mapped = to_pyerr(error);
            let value = mapped.value(py);
            assert!(mapped.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            assert!(!mapped.is_instance_of::<RustUpstreamError>(py));
            assert_eq!(value.to_string(), message);
            for attribute in ["status_code", "ocr_request_format_error", "headers"] {
                assert!(!value.hasattr(attribute).unwrap(), "{attribute}");
            }
        });
    }
}
