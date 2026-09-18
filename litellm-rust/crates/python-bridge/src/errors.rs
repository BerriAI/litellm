use std::fmt::Display;

use litellm_core::failure::{Phase, Rejection};
use litellm_core::{audio_transcription, ocr, responses};
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError};
use pyo3::prelude::*;

pyo3::create_exception!(
    _native,
    RustBridgeDeclined,
    PyException,
    "Rejected before any provider I/O, so the host may retry on its own path. Args are (message, rejection); rejection is one of the litellm_core::failure::Rejection names."
);

pyo3::create_exception!(
    _native,
    RustUpstreamError,
    PyException,
    "The provider was called and did not succeed. Args are (status, body, headers); status is 0 when no response arrived, and body then carries the transport or decoding failure."
);

pub(crate) fn to_pyerr(error: &impl Display, phase: Phase<'_>) -> PyErr {
    match phase {
        Phase::BeforeProvider(Rejection::FileNotFound(path)) => {
            PyFileNotFoundError::new_err(format!("File not found: {}", path.display()))
        }
        Phase::BeforeProvider(Rejection::FileRead(source)) => {
            PyOSError::new_err(source.to_string())
        }
        Phase::BeforeProvider(rejection) => {
            RustBridgeDeclined::new_err((error.to_string(), rejection.name()))
        }
        Phase::Upstream {
            status,
            body,
            headers,
        } => RustUpstreamError::new_err((status, body.to_owned(), headers.to_vec())),
        Phase::AfterProvider => PyErr::new::<RustUpstreamError, _>((
            0u16,
            error.to_string(),
            Vec::<(String, String)>::new(),
        )),
    }
}

pub(crate) fn audio_transcription_error_to_pyerr(error: audio_transcription::Error) -> PyErr {
    to_pyerr(&error, error.phase())
}

pub(crate) fn responses_error_to_pyerr(error: responses::Error) -> PyErr {
    to_pyerr(&error, error.phase())
}

pub(crate) fn ocr_error_to_pyerr(error: ocr::Error) -> PyErr {
    to_pyerr(&error, error.phase())
}

#[cfg(test)]
mod tests {
    use litellm_core::transport::Error as TransportError;

    use super::*;

    fn declined(py: Python<'_>, error: &PyErr) -> (String, String) {
        assert!(error.is_instance_of::<RustBridgeDeclined>(py), "{error}");
        error.value(py).getattr("args").unwrap().extract().unwrap()
    }

    fn upstream(py: Python<'_>, error: &PyErr) -> (u16, String, Vec<(String, String)>) {
        assert!(error.is_instance_of::<RustUpstreamError>(py), "{error}");
        error.value(py).getattr("args").unwrap().extract().unwrap()
    }

    #[test]
    fn transport_failures_split_by_phase_with_typed_fields() {
        Python::initialize();
        Python::attach(|py| {
            let connect = audio_transcription_error_to_pyerr(
                TransportError::Connect("unreachable".into()).into(),
            );
            let (message, rejection) = declined(py, &connect);
            assert_eq!(rejection, "unreachable");
            assert_eq!(message, "could not reach the provider: unreachable");

            let http = audio_transcription_error_to_pyerr(
                TransportError::Http {
                    status: 429,
                    body: "slow down".into(),
                }
                .into(),
            );
            assert_eq!(upstream(py, &http), (429, "slow down".into(), vec![]));

            let network = audio_transcription_error_to_pyerr(
                TransportError::Network("timed out".into()).into(),
            );
            let (status, body, _) = upstream(py, &network);
            assert_eq!(status, 0);
            assert_eq!(body, "upstream network error: timed out");
        });
    }

    #[test]
    fn credential_and_request_rejections_are_named() {
        Python::initialize();
        Python::attach(|py| {
            let missing = audio_transcription_error_to_pyerr(audio_transcription::Error::Auth(
                litellm_auth::Error::MissingApiKey {
                    provider: "Bedrock",
                    environment_variable: "AWS_BEARER_TOKEN_BEDROCK",
                },
            ));
            assert_eq!(declined(py, &missing).1, "credential");

            let invalid = audio_transcription_error_to_pyerr(
                audio_transcription::Error::InvalidRequest("bad audio".into()),
            );
            assert_eq!(
                declined(py, &invalid),
                (
                    "invalid request: bad audio".into(),
                    "invalid_request".into()
                )
            );
        });
    }

    #[test]
    fn ocr_failures_keep_provider_headers_file_errors_and_the_request_format_rejection() {
        Python::initialize();
        Python::attach(|py| {
            let provider = ocr_error_to_pyerr(ocr::Error::Provider {
                status: 429,
                body: r#"{"message":"rate limited"}"#.into(),
                headers: vec![("Retry-After".into(), "17".into())],
            });
            assert_eq!(
                upstream(py, &provider),
                (
                    429,
                    r#"{"message":"rate limited"}"#.into(),
                    vec![("Retry-After".into(), "17".into())]
                )
            );

            let missing = ocr_error_to_pyerr(ocr::Error::FileRead {
                path: "/tmp/scan.pdf".into(),
                source: std::sync::Arc::new(std::io::Error::from(std::io::ErrorKind::NotFound)),
            });
            assert!(missing.is_instance_of::<PyFileNotFoundError>(py));
            assert_eq!(
                missing.value(py).to_string(),
                "File not found: /tmp/scan.pdf"
            );

            let unreadable = ocr_error_to_pyerr(ocr::Error::FileRead {
                path: "/tmp/scan.pdf".into(),
                source: std::sync::Arc::new(std::io::Error::from(
                    std::io::ErrorKind::PermissionDenied,
                )),
            });
            assert!(unreadable.is_instance_of::<PyOSError>(py));

            assert_eq!(
                declined(py, &ocr_error_to_pyerr(ocr::Error::RequestFormat)).1,
                "request_format"
            );
            assert_eq!(
                declined(py, &ocr_error_to_pyerr(ocr::Error::MissingDocumentUrl)),
                ("Document URL is required".into(), "invalid_request".into())
            );
        });
    }
}
