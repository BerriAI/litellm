use litellm_core::transport::Error as TransportError;
use litellm_core::{Error, audio_transcription, chat_completions, messages, ocr, responses};
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

fn auth_is_value_error(error: &litellm_auth::Error) -> bool {
    !matches!(error, litellm_auth::Error::MissingApiKey { .. })
}

pub(crate) fn messages_error_to_pyerr(error: messages::Error) -> PyErr {
    core_error_to_pyerr(error.into())
}

pub(crate) fn audio_transcription_error_to_pyerr(error: audio_transcription::Error) -> PyErr {
    core_error_to_pyerr(error.into())
}

pub(crate) fn responses_error_to_pyerr(error: responses::Error) -> PyErr {
    core_error_to_pyerr(error.into())
}

pub(crate) fn core_error_to_pyerr(error: Error) -> PyErr {
    let value_error = match &error {
        Error::Ocr(error) => {
            error.is_request()
                || matches!(
                    error,
                    ocr::Error::Auth(_)
                        | ocr::Error::InvalidProvider(_)
                        | ocr::Error::InvalidRequest(_)
                        | ocr::Error::MissingField(_)
                        | ocr::Error::MissingDocumentUrl
                )
        }
        Error::Messages(error) => match error {
            messages::Error::Auth(source) => auth_is_value_error(source),
            messages::Error::InvalidProvider(_)
            | messages::Error::InvalidRequest(_)
            | messages::Error::Headers(_) => true,
            _ => false,
        },
        Error::AudioTranscription(error) => match error {
            audio_transcription::Error::Auth(source) => auth_is_value_error(source),
            audio_transcription::Error::InvalidProvider(_)
            | audio_transcription::Error::InvalidRequest(_)
            | audio_transcription::Error::Headers(_)
            | audio_transcription::Error::InvalidType { .. }
            | audio_transcription::Error::MissingField(_)
            | audio_transcription::Error::Aws(_) => true,
            _ => false,
        },
        Error::ChatCompletions(error) => match error {
            chat_completions::Error::Auth(source) => auth_is_value_error(source),
            chat_completions::Error::InvalidProvider(_)
            | chat_completions::Error::InvalidRequest(_)
            | chat_completions::Error::Headers(_)
            | chat_completions::Error::InvalidType { .. }
            | chat_completions::Error::MissingField(_)
            | chat_completions::Error::Aws(_) => true,
            _ => false,
        },
        Error::Responses(error) => match error {
            responses::Error::Auth(source) => auth_is_value_error(source),
            responses::Error::InvalidProvider(_)
            | responses::Error::InvalidRequest(_)
            | responses::Error::Headers(_) => true,
            _ => false,
        },
    };
    if value_error {
        PyValueError::new_err(error.to_string())
    } else {
        PyRuntimeError::new_err(error.to_string())
    }
}

/// Map a route error for a route whose host keeps a Python implementation.
///
/// The distinction the host needs is whether the provider was already called.
/// Everything raised before the request goes out is safe for the host to retry
/// on its own path; anything after it is not, because the provider has already
/// done the work and billed for it.
pub(crate) fn chat_completions_error_to_pyerr(error: chat_completions::Error) -> PyErr {
    use chat_completions::Error;
    match error {
        Error::Unsupported(_)
        | Error::Auth(_)
        | Error::Aws(_)
        | Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_)
        | Error::Headers(_)
        | Error::Transport(TransportError::Connect(_)) => {
            RustBridgeDeclined::new_err(error.to_string())
        }
        Error::Transport(TransportError::Http { status, body }) => {
            RustUpstreamError::new_err((status, body))
        }
        Error::Transport(TransportError::Network(message)) | Error::InvalidResponse(message) => {
            RustUpstreamError::new_err((0u16, message))
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_status_and_dispatch_certainty_survive_python_mapping() {
        Python::initialize();
        Python::attach(|py| {
            let connect = chat_completions_error_to_pyerr(
                TransportError::Connect("unreachable".into()).into(),
            );
            assert!(connect.is_instance_of::<RustBridgeDeclined>(py));
            let network =
                chat_completions_error_to_pyerr(TransportError::Network("timed out".into()).into());
            assert!(network.is_instance_of::<RustUpstreamError>(py));
            let upstream = chat_completions_error_to_pyerr(
                TransportError::Http {
                    status: 429,
                    body: "slow down".into(),
                }
                .into(),
            );
            assert_eq!(
                upstream
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(u16, String)>()
                    .unwrap(),
                (429, "slow down".into())
            );
        });
    }

    #[test]
    fn missing_api_key_stays_a_runtime_error_while_other_auth_failures_are_value_errors() {
        Python::initialize();
        Python::attach(|py| {
            let missing = messages_error_to_pyerr(messages::Error::Auth(
                litellm_auth::Error::MissingApiKey {
                    provider: "Anthropic",
                    environment_variable: "ANTHROPIC_API_KEY",
                },
            ));
            assert!(missing.is_instance_of::<PyRuntimeError>(py));
            let invalid =
                messages_error_to_pyerr(messages::Error::Auth(litellm_auth::Error::InvalidHeader));
            assert!(invalid.is_instance_of::<PyValueError>(py));
        });
    }
}
