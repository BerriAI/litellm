use litellm_core::transport::Error as TransportError;
use litellm_core::{Error, audio_transcription, chat_completions, messages, responses};
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

pub(crate) fn core_error_to_pyerr(error: impl Into<Error>) -> PyErr {
    let error = error.into();
    let value_error = match &error {
        Error::Ocr(error) => {
            error.is_request()
                || matches!(error, litellm_core::ocr::Error::InvalidProvider(_))
                || matches!(error, litellm_core::ocr::Error::Auth(source) if !matches!(source, litellm_auth::Error::MissingApiKey { .. }))
        }
        Error::Messages(error) => match error {
            messages::Error::Auth(source) => {
                !matches!(source, litellm_auth::Error::MissingApiKey { .. })
            }
            messages::Error::InvalidProvider(_)
            | messages::Error::InvalidRequest(_)
            | messages::Error::Params(_)
            | messages::Error::Headers(_) => true,
            _ => false,
        },
        Error::AudioTranscription(error) => match error {
            audio_transcription::Error::Auth(source) => {
                !matches!(source, litellm_auth::Error::MissingApiKey { .. })
            }
            audio_transcription::Error::InvalidProvider(_)
            | audio_transcription::Error::InvalidRequest(_)
            | audio_transcription::Error::Params(_)
            | audio_transcription::Error::Headers(_)
            | audio_transcription::Error::InvalidType { .. }
            | audio_transcription::Error::MissingField(_)
            | audio_transcription::Error::Aws(_) => true,
            _ => false,
        },
        Error::ChatCompletions(error) => match error {
            chat_completions::Error::Auth(source) => {
                !matches!(source, litellm_auth::Error::MissingApiKey { .. })
            }
            chat_completions::Error::InvalidProvider(_)
            | chat_completions::Error::InvalidRequest(_)
            | chat_completions::Error::Params(_)
            | chat_completions::Error::Headers(_)
            | chat_completions::Error::InvalidType { .. }
            | chat_completions::Error::MissingField(_)
            | chat_completions::Error::Aws(_) => true,
            _ => false,
        },
        Error::Responses(error) => match error {
            responses::Error::Auth(source) => {
                !matches!(source, litellm_auth::Error::MissingApiKey { .. })
            }
            responses::Error::InvalidProvider(_)
            | responses::Error::InvalidRequest(_)
            | responses::Error::Params(_)
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
        | Error::Params(_)
        | Error::Headers(_)
        | Error::Transport(TransportError::Connect(_)) => {
            RustBridgeDeclined::new_err(error.to_string())
        }
        Error::ResponseTransform(source) => RustUpstreamError::new_err((0u16, source.to_string())),
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
    fn response_normalization_failure_never_declines_or_loses_the_cause() {
        use std::error::Error as _;
        Python::initialize();
        let error = chat_completions::Error::ResponseTransform(Box::new(
            chat_completions::Error::MissingField("usage"),
        ));
        assert!(matches!(
            error
                .source()
                .unwrap()
                .downcast_ref::<Box<chat_completions::Error>>(),
            Some(source) if **source == chat_completions::Error::MissingField("usage")
        ));
        let mapped = chat_completions_error_to_pyerr(error);
        Python::attach(|py| {
            assert!(mapped.is_instance_of::<RustUpstreamError>(py));
            assert!(!mapped.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(
                mapped
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(u16, String)>()
                    .unwrap(),
                (0, "missing required field: usage".into())
            );
        });
    }

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
}
