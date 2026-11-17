use std::io::ErrorKind;

use litellm_error::Report;
use litellm_llms::base_llm::ocr::error::Error;
use pyo3::{
    exceptions::{PyFileNotFoundError, PyOSError},
    prelude::*,
};

use crate::errors::{by_fault, upstream_error};

const REQUEST_FORMAT_ERROR: &str = "ocr_request_format_error";

pub(super) fn to_pyerr(error: Error) -> PyErr {
    Python::attach(|py| native(py, error).unwrap_or_else(std::convert::identity))
}

/// Maps an OCR route failure once. A Python exception raised inside the call is returned as it
/// is, everything else is mapped and handed to `public`, the route's public exception mapping.
pub(super) fn route_failure(
    py: Python<'_>,
    error: Error,
    public: impl FnOnce(PyErr) -> PyResult<PyErr>,
) -> PyResult<PyErr> {
    match crate::secrets::python_error(py, &error) {
        Some(raised) => Ok(raised),
        None => public(native(py, error)?),
    }
}

fn native(py: Python<'_>, error: Error) -> PyResult<PyErr> {
    let mapped = match (&error, error.upstream_response()) {
        (_, Some((status, body, headers))) => upstream_error(py, status, body, headers)?,
        (Error::RequestFormat, None) => {
            let mapped = by_fault(true, &error);
            mapped.value(py).setattr(REQUEST_FORMAT_ERROR, true)?;
            mapped
        }
        (Error::FileRead { path, cause }, None) if cause.kind() == ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(format!("File not found: {}", path.display()))
        }
        (Error::FileRead { cause, .. }, None) => {
            PyOSError::new_err(Report::new(cause.as_ref()).to_string())
        }
        (_, None) => by_fault(error.is_request(), &error),
    };
    if let Some(status) = error.http_status_code() {
        let value = mapped.value(py);
        value.setattr("status_code", status)?;
        value.setattr("message", value.str()?)?;
    }
    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use rstest::rstest;

    use super::*;
    use crate::errors::RustUpstreamError;

    fn status_code(py: Python<'_>, mapped: &PyErr) -> Option<u16> {
        mapped
            .value(py)
            .getattr_opt("status_code")
            .unwrap()
            .map(|status| status.extract().unwrap())
    }

    fn file_read(kind: ErrorKind) -> Error {
        Error::FileRead {
            path: "/missing/scan.pdf".into(),
            cause: std::sync::Arc::new(std::io::Error::new(kind, "disk said no")),
        }
    }

    #[rstest]
    #[case::provider(
        Error::Provider {
            status: 429,
            body: r#"{"message":"rate limited"}"#.into(),
            headers: Box::new(reqwest::header::HeaderMap::from_iter([(
                reqwest::header::RETRY_AFTER,
                reqwest::header::HeaderValue::from_static("17"),
            )])),
        },
        vec![("retry-after".into(), "17".into())]
    )]
    #[case::transport(litellm_http::Error::upstream(429, r#"{"message":"rate limited"}"#.into()).into(), vec![])]
    fn a_provider_answer_keeps_its_status_body_and_headers(
        #[case] error: Error,
        #[case] headers: Vec<(String, String)>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(error);
            assert!(mapped.is_instance_of::<RustUpstreamError>(py));
            let value = mapped.value(py);
            let args: (u16, String) = value.getattr("args").unwrap().extract().unwrap();
            assert_eq!(args, (429, r#"{"message":"rate limited"}"#.into()));
            let kept: Vec<(String, String)> = value.getattr("headers").unwrap().extract().unwrap();
            assert_eq!(kept, headers);
            assert_eq!(status_code(py, &mapped), Some(429));
        });
    }

    #[rstest]
    #[case::not_found(Some(404), 404, "OCR document download failed")]
    #[case::rate_limited(Some(429), 429, "OCR document download failed")]
    #[case::timeout(None, 408, "OCR document download timed out")]
    #[tokio::test]
    async fn document_download_failures_keep_the_upstream_exception(
        #[case] response_status: Option<u16>,
        #[case] expected_status: u16,
        #[case] expected_body: &str,
    ) {
        use std::time::Duration;

        use litellm_http::{Client, media::MediaFetcher};
        use litellm_llms::base_llm::ocr::{
            document::inline_remote_document, transformation::OcrConnection,
        };
        use litellm_llms_types::formats::ocr::OcrDocument;
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::TcpListener,
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1];
            socket.read_exact(&mut request).await.unwrap();
            if let Some(status) = response_status {
                socket.write_all(format!("HTTP/1.1 {status} Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
            } else {
                std::future::pending::<()>().await;
            }
        });
        let connection = OcrConnection {
            timeout: Duration::from_millis(100),
            ..OcrConnection::default()
        };
        let error = inline_remote_document(
            &MediaFetcher::for_test(Client::for_test()),
            OcrDocument::DocumentUrl {
                document_url: format!("http://{address}/document.pdf"),
                extra_fields: Default::default(),
            },
            &connection,
        )
        .await
        .unwrap_err();
        server.abort();
        if response_status.is_none() {
            assert!(litellm_error::find::<tokio::time::error::Elapsed>(&error).is_some());
        }
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(error);
            assert!(mapped.is_instance_of::<RustUpstreamError>(py), "{mapped}");
            let args: (u16, String) = mapped.value(py).getattr("args").unwrap().extract().unwrap();
            assert_eq!(args, (expected_status, expected_body.into()));
            assert_eq!(status_code(py, &mapped), Some(expected_status));
        });
    }

    #[rstest]
    #[case::missing_document_url(Error::MissingDocumentUrl)]
    #[case::invalid_features(Error::Features)]
    #[case::invalid_provider(Error::InvalidProvider("unknown".into()))]
    fn request_faults_are_value_errors_with_status_400(#[case] error: Error) {
        Python::initialize();
        Python::attach(|py| {
            let report = Report::new(&error).to_string();
            let mapped = to_pyerr(error);
            assert!(mapped.is_instance_of::<PyValueError>(py));
            assert_eq!(mapped.value(py).to_string(), report);
            assert_eq!(status_code(py, &mapped), Some(400));
        });
    }

    #[rstest]
    #[case::missing_api_key(litellm_auth::Error::MissingApiKey {
        provider: "Mistral",
        environment_variable: "MISTRAL_API_KEY",
    }
    .into())]
    #[case::missing_credential(Error::MissingCredential {
        credential: "Azure AI credentials",
        guidance: "set AZURE_AI_API_KEY",
    })]
    fn a_missing_credential_is_a_statusless_runtime_error(#[case] error: Error) {
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(error);
            assert!(mapped.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(status_code(py, &mapped), None);
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
                    .getattr(REQUEST_FORMAT_ERROR)
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            assert_eq!(status_code(py, &mapped), Some(400));
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

    #[test]
    fn missing_files_map_to_file_not_found_naming_the_path() {
        Python::initialize();
        Python::attach(|py| {
            let mapped = to_pyerr(file_read(ErrorKind::NotFound));
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
            let mapped = to_pyerr(file_read(ErrorKind::PermissionDenied));
            assert!(mapped.is_instance_of::<PyOSError>(py));
            assert!(!mapped.is_instance_of::<PyFileNotFoundError>(py));
            assert_eq!(mapped.value(py).to_string(), "disk said no");
        });
    }

    #[rstest]
    #[case::oversized(Error::TooLarge { limit: 7 })]
    #[case::malformed_field(Error::response_field(
        "pages[0].index",
        "-1".parse::<u32>().unwrap_err()
    ))]
    fn response_failures_are_statusless_runtime_errors(
        #[case] error: Error,
        #[values("status_code", "ocr_request_format_error", "headers")] attribute: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let report = Report::new(&error).to_string();
            let mapped = to_pyerr(error);
            let value = mapped.value(py);
            assert!(mapped.is_instance_of::<PyRuntimeError>(py));
            assert!(!mapped.is_instance_of::<RustUpstreamError>(py));
            assert_eq!(value.to_string(), report);
            assert!(!value.hasattr(attribute).unwrap());
        });
    }
}
