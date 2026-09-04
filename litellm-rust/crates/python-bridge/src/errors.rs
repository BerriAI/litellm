use litellm_core::error::{Error, ErrorCode, ProviderState};
use pyo3::prelude::*;
use pyo3::types::PyNone;

pyo3::create_exception!(
    _native,
    RustPreparationError,
    pyo3::exceptions::PyException,
    "Rust stopped before provider execution and the host may use another implementation."
);

pyo3::create_exception!(
    _native,
    RustExecutionError,
    pyo3::exceptions::PyException,
    "Rust owns provider execution and the host must not retry through another implementation."
);

pub(crate) fn core_error_to_pyerr(error: Error) -> PyErr {
    match error {
        Error::InvalidType { .. }
        | Error::MissingField(_)
        | Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::Auth(_)
        | Error::Connect(_)
        | Error::Routing(_)
        | Error::Unsupported(_) => structured_error::<RustPreparationError>(
            error_code(&error),
            error.to_string(),
            None,
            ProviderState::NotStarted,
        ),
        Error::Http { status, .. } => structured_error::<RustExecutionError>(
            ErrorCode::Upstream,
            format!("upstream request failed with status {status}"),
            Some(status),
            ProviderState::ResponseReceived,
        ),
        Error::Network(_) => structured_error::<RustExecutionError>(
            ErrorCode::Transport,
            "upstream network error".to_string(),
            None,
            ProviderState::MayHaveStarted,
        ),
        Error::InvalidResponse(message) => structured_error::<RustExecutionError>(
            ErrorCode::InvalidResponse,
            format!("invalid response: {message}"),
            None,
            ProviderState::ResponseReceived,
        ),
    }
}

fn error_code(error: &Error) -> ErrorCode {
    match error {
        Error::InvalidType { .. } | Error::MissingField(_) | Error::InvalidRequest(_) => {
            ErrorCode::InvalidRequest
        }
        Error::InvalidProvider(_) | Error::Unsupported(_) => ErrorCode::Unsupported,
        Error::Auth(_) => ErrorCode::Authentication,
        Error::Connect(_) | Error::Network(_) => ErrorCode::Transport,
        Error::Routing(_) => ErrorCode::Routing,
        Error::Http { .. } => ErrorCode::Upstream,
        Error::InvalidResponse(_) => ErrorCode::InvalidResponse,
    }
}

fn structured_error<E>(
    code: ErrorCode,
    message: String,
    status_code: Option<u16>,
    provider_state: ProviderState,
) -> PyErr
where
    E: pyo3::type_object::PyTypeInfo,
{
    let error = Python::attach(|py| PyErr::from_type(py.get_type::<E>(), (message.clone(),)));
    Python::attach(|py| {
        let value = error.value(py);
        value
            .setattr("code", code.as_str())
            .expect("exception attributes are writable");
        value
            .setattr("message", message)
            .expect("exception attributes are writable");
        match status_code {
            Some(status) => value
                .setattr("status_code", status)
                .expect("exception attributes are writable"),
            None => value
                .setattr("status_code", PyNone::get(py))
                .expect("exception attributes are writable"),
        }
        value
            .setattr("provider_state", provider_state.as_str())
            .expect("exception attributes are writable");
    });
    error
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let preparation = py.get_type::<RustPreparationError>();
    let execution = py.get_type::<RustExecutionError>();
    module.add("RustPreparationError", &preparation)?;
    module.add("RustExecutionError", &execution)?;
    module.add("RustBridgeDeclined", preparation)?;
    module.add("RustUpstreamError", execution)
}

pub(crate) fn ocr_error_to_pyerr(error: Error) -> PyErr {
    core_error_to_pyerr(error)
}

pub(crate) fn chat_completions_error_to_pyerr(error: Error) -> PyErr {
    core_error_to_pyerr(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_expose_structured_ownership_fields() {
        Python::initialize();
        Python::attach(|py| {
            let preparation =
                core_error_to_pyerr(Error::InvalidRequest("invalid document".to_string()));
            assert!(preparation.is_instance_of::<RustPreparationError>(py));
            assert_eq!(
                preparation
                    .value(py)
                    .getattr("code")
                    .and_then(|item| item.extract::<String>())
                    .expect("code is a string"),
                "invalid_request"
            );

            let execution = core_error_to_pyerr(Error::Http {
                status: 429,
                body: "secret body".to_string(),
            });
            assert!(execution.is_instance_of::<RustExecutionError>(py));
            let value = execution.value(py);
            assert_eq!(
                value
                    .getattr("status_code")
                    .and_then(|item| item.extract::<u16>())
                    .expect("status is an integer"),
                429
            );
            assert_eq!(
                value
                    .getattr("provider_state")
                    .and_then(|item| item.extract::<String>())
                    .expect("provider state is a string"),
                "response_received"
            );
            assert!(!value.to_string().contains("secret body"));
        });
    }
}
