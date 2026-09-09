use super::*;
use pyo3::exceptions::{PyTypeError, PyValueError};

enum ExpectedPythonError {
    Value(&'static str),
    Runtime(&'static str),
    Declined(&'static str),
    Upstream(u16, &'static str),
}

fn assert_python_error(py: Python<'_>, error: PyErr, expected: ExpectedPythonError) {
    match expected {
        ExpectedPythonError::Value(message) => {
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(error.to_string(), format!("ValueError: {message}"));
        }
        ExpectedPythonError::Runtime(message) => {
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(error.to_string(), format!("RuntimeError: {message}"));
        }
        ExpectedPythonError::Declined(message) => {
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(
                error
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(String,)>()
                    .unwrap(),
                (message.to_owned(),)
            );
        }
        ExpectedPythonError::Upstream(status, message) => {
            assert!(error.is_instance_of::<RustUpstreamError>(py));
            assert_eq!(
                error
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(u16, String)>()
                    .unwrap(),
                (status, message.to_owned())
            );
        }
    }
}

#[test]
fn generic_core_errors_keep_their_public_contract() {
    Python::initialize();
    Python::attach(|py| {
        for (error, expected) in [
            (
                CoreError::InvalidProvider("secret".into()),
                ExpectedPythonError::Value("Invalid provider configuration"),
            ),
            (
                CoreError::InvalidRequest("secret".into()),
                ExpectedPythonError::Value("Invalid provider request"),
            ),
            (
                CoreError::InvalidType {
                    expected: "string",
                    actual: "integer",
                },
                ExpectedPythonError::Value("expected string, got integer"),
            ),
            (
                CoreError::MissingField("model"),
                ExpectedPythonError::Value("missing required field: model"),
            ),
            (
                CoreError::Auth("secret".into()),
                ExpectedPythonError::Value("Provider authentication failed"),
            ),
            (
                CoreError::Http {
                    status: 418,
                    body: "secret".into(),
                },
                ExpectedPythonError::Runtime("Provider request failed (HTTP 418)"),
            ),
            (
                CoreError::Network("secret".into()),
                ExpectedPythonError::Runtime("Provider transport failed"),
            ),
            (
                CoreError::Connect("secret".into()),
                ExpectedPythonError::Runtime("Provider transport failed"),
            ),
            (
                CoreError::InvalidResponse("secret".into()),
                ExpectedPythonError::Runtime("Invalid provider response"),
            ),
            (
                CoreError::Routing("secret".into()),
                ExpectedPythonError::Runtime("Provider routing failed"),
            ),
            (
                CoreError::Unsupported("secret"),
                ExpectedPythonError::Runtime("Operation is not supported"),
            ),
        ] {
            let bridge_error = Error::Core(error);
            assert!(!bridge_error.to_string().contains("secret"));
            assert!(std::error::Error::source(&bridge_error).is_none());
            let python_error: PyErr = bridge_error.into();
            assert!(python_error.cause(py).is_none());
            assert_python_error(py, python_error, expected);
        }
    });
}

#[test]
fn chat_completions_core_errors_keep_retry_semantics() {
    Python::initialize();
    Python::attach(|py| {
        for (error, expected) in [
            (
                CoreError::InvalidProvider("secret".into()),
                ExpectedPythonError::Declined("Invalid provider configuration"),
            ),
            (
                CoreError::InvalidRequest("secret".into()),
                ExpectedPythonError::Declined("Invalid provider request"),
            ),
            (
                CoreError::InvalidType {
                    expected: "string",
                    actual: "integer",
                },
                ExpectedPythonError::Declined("expected string, got integer"),
            ),
            (
                CoreError::MissingField("model"),
                ExpectedPythonError::Declined("missing required field: model"),
            ),
            (
                CoreError::Auth("secret".into()),
                ExpectedPythonError::Declined("Provider authentication failed"),
            ),
            (
                CoreError::Http {
                    status: 503,
                    body: "secret".into(),
                },
                ExpectedPythonError::Upstream(503, "Provider request failed (HTTP 503)"),
            ),
            (
                CoreError::Network("secret".into()),
                ExpectedPythonError::Upstream(0, "Provider transport failed"),
            ),
            (
                CoreError::Connect("secret".into()),
                ExpectedPythonError::Declined("Could not reach provider"),
            ),
            (
                CoreError::InvalidResponse("secret".into()),
                ExpectedPythonError::Upstream(0, "Invalid provider response"),
            ),
            (
                CoreError::Routing("secret".into()),
                ExpectedPythonError::Declined("Provider routing failed"),
            ),
            (
                CoreError::Unsupported("secret"),
                ExpectedPythonError::Declined("Operation is not supported"),
            ),
        ] {
            let python_error = chat_completions_error_to_pyerr(error);
            assert!(!python_error.to_string().contains("secret"));
            assert!(python_error.cause(py).is_none());
            assert_python_error(py, python_error, expected);
        }
    });
}

#[test]
fn messages_provider_errors_keep_commit_semantics() {
    Python::initialize();
    Python::attach(|py| {
        for (error, expected) in [
            (
                CoreError::InvalidProvider("secret".into()),
                ExpectedPythonError::Value("Invalid provider configuration"),
            ),
            (
                CoreError::InvalidRequest("secret".into()),
                ExpectedPythonError::Value("Invalid provider request"),
            ),
            (
                CoreError::InvalidType {
                    expected: "string",
                    actual: "integer",
                },
                ExpectedPythonError::Value("expected string, got integer"),
            ),
            (
                CoreError::MissingField("model"),
                ExpectedPythonError::Value("missing required field: model"),
            ),
            (
                CoreError::Http {
                    status: 429,
                    body: "secret".into(),
                },
                ExpectedPythonError::Upstream(429, "Provider request failed (HTTP 429)"),
            ),
            (
                CoreError::Network("secret".into()),
                ExpectedPythonError::Upstream(0, "Provider transport failed"),
            ),
            (
                CoreError::Connect("secret".into()),
                ExpectedPythonError::Upstream(0, "Provider transport failed"),
            ),
            (
                CoreError::InvalidResponse("secret".into()),
                ExpectedPythonError::Upstream(0, "Invalid provider response"),
            ),
            (
                CoreError::Auth("secret".into()),
                ExpectedPythonError::Value("Provider authentication failed"),
            ),
            (
                CoreError::Routing("secret".into()),
                ExpectedPythonError::Runtime("Provider routing failed"),
            ),
            (
                CoreError::Unsupported("secret"),
                ExpectedPythonError::Runtime("Operation is not supported"),
            ),
        ] {
            let python_error = messages_provider_error_to_pyerr(error);
            assert!(!python_error.to_string().contains("secret"));
            assert!(python_error.cause(py).is_none());
            assert_python_error(py, python_error, expected);
        }
    });
}

#[test]
fn lifecycle_and_state_errors_keep_runtime_messages() {
    Python::initialize();
    Python::attach(|py| {
        for (error, message) in [
            (
                Error::LifecycleComplete(Route::Messages),
                "messages lifecycle is complete",
            ),
            (
                Error::RequestConsumed(Route::ChatCompletions),
                "chat completions request was already sent or cleared",
            ),
            (
                Error::StateCleared(Route::Ocr, StateSlot::PreCall),
                "OCR pre-call state was cleared",
            ),
            (
                Error::TerminalUnavailable(Route::Ocr),
                "OCR terminal record is unavailable",
            ),
        ] {
            assert_python_error(py, error.into(), ExpectedPythonError::Runtime(message));
        }
    });
}

#[test]
fn python_error_passthrough_preserves_the_exception_object() {
    Python::initialize();
    Python::attach(|py| {
        let error = py
            .run(c"raise RuntimeError('outer')", None, None)
            .expect_err("Python should raise");
        error.set_cause(py, Some(PyValueError::new_err("cause")));
        error.set_context(py, Some(PyTypeError::new_err("context")));
        let original_value = error.value(py).clone();
        let original_traceback = error
            .traceback(py)
            .expect("raised error should have a traceback");

        let converted: PyErr = Error::from(error).into();

        assert!(converted.value(py).is(&original_value));
        assert!(
            converted
                .traceback(py)
                .is_some_and(|traceback| traceback.is(&original_traceback))
        );
        assert_eq!(
            converted.cause(py).unwrap().to_string(),
            "ValueError: cause"
        );
        assert_eq!(
            converted.context(py).unwrap().to_string(),
            "TypeError: context"
        );
    });
}

#[test]
fn ocr_statuses_keep_sdk_exception_names() {
    for (status, class) in [
        (400, "BadRequestError"),
        (401, "AuthenticationError"),
        (403, "PermissionDeniedError"),
        (404, "NotFoundError"),
        (422, "UnprocessableEntityError"),
        (429, "RateLimitError"),
        (500, "InternalServerError"),
        (502, "BadGatewayError"),
        (503, "ServiceUnavailableError"),
        (504, "APIError"),
    ] {
        assert_eq!(OcrExceptionClass::from(status).as_ref(), class);
    }
}
