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
            _ => error.is_request(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_api_key_stays_a_runtime_error_while_other_auth_failures_are_value_errors() {
        Python::initialize();
        Python::attach(|py| {
            let missing = audio_transcription_error_to_pyerr(audio_transcription::Error::Auth(
                litellm_auth::Error::MissingApiKey {
                    provider: "Bedrock",
                    environment_variable: "AWS_BEARER_TOKEN_BEDROCK",
                },
            ));
            assert!(missing.is_instance_of::<PyRuntimeError>(py));
            let invalid = audio_transcription_error_to_pyerr(audio_transcription::Error::Auth(
                litellm_auth::Error::InvalidHeader,
            ));
            assert!(invalid.is_instance_of::<PyValueError>(py));
        });
    }
}
