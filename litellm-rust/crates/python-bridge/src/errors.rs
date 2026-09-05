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
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

pub(crate) fn chat_completions_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Unsupported(_)
        | Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_)
        | Error::Routing(_) => RustBridgeDeclined::new_err(err.to_string()),
        Error::Auth(message) => RustUpstreamError::new_err((401u16, message)),
        Error::Http { status, body } => {
            RustUpstreamError::new_err((status, format!("{status}: {body}")))
        }
        Error::Connect(message) | Error::Network(message) | Error::InvalidResponse(message) => {
            RustUpstreamError::new_err((0u16, message))
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}

pub(crate) fn ocr_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::MissingField("document_url" | "image_url") => {
            PyValueError::new_err("Document URL is required")
        }
        Error::Http { status, body } => RustUpstreamError::new_err((status, body)),
        other => core_error_to_pyerr(other),
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    #[once]
    fn initialized_python() {
        Python::initialize();
    }

    #[rstest]
    #[case::connect(Error::Connect, "connection failed")]
    #[case::network(Error::Network, "connection reset")]
    #[case::invalid_response(Error::InvalidResponse, "invalid response")]
    fn chat_transport_errors_do_not_authorize_python_fallback(
        #[from(initialized_python)] (): (),
        #[case] error: fn(String) -> Error,
        #[case] message: &str,
    ) {
        Python::attach(|py| {
            let mapped = chat_completions_error_to_pyerr(error(message.to_string()));
            assert!(mapped.is_instance_of::<RustUpstreamError>(py));
            let args: (u16, String) = mapped
                .value(py)
                .getattr("args")
                .and_then(|args| args.extract())
                .expect("transport errors retain their status and message");
            assert_eq!(args, (0, message.to_string()));
        });
    }

    #[rstest]
    fn credential_failures_do_not_authorize_python_replay(#[from(initialized_python)] (): ()) {
        Python::attach(|py| {
            let error =
                chat_completions_error_to_pyerr(Error::Auth("credential exchange failed".into()));
            assert!(error.is_instance_of::<RustUpstreamError>(py));
            let args: (u16, String) = error.value(py).getattr("args").unwrap().extract().unwrap();
            assert_eq!(args, (401, "credential exchange failed".into()));
        });
    }

    #[rstest]
    #[case::document_url("document_url")]
    #[case::image_url("image_url")]
    fn ocr_errors_preserve_python_validation(
        #[from(initialized_python)] (): (),
        #[case] field: &'static str,
    ) {
        Python::attach(|py| {
            let mapped = ocr_error_to_pyerr(Error::MissingField(field));
            assert!(mapped.is_instance_of::<PyValueError>(py));
            assert_eq!(mapped.value(py).to_string(), "Document URL is required");
        });
    }

    #[rstest]
    fn ocr_errors_preserve_provider_details(#[from(initialized_python)] (): ()) {
        Python::attach(|py| {
            let mapped = ocr_error_to_pyerr(Error::Http {
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
        });
    }
}
