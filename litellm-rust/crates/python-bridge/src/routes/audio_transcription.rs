use crate::errors::core_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions};
use litellm_core::Error;
use litellm_core::audio_transcription::AudioTranscriptionRequest;
use litellm_core::audio_transcription::audio_transcription as run_route;
use litellm_core::request_context::LiteLlmRequestContext;
use pyo3::prelude::*;
use serde_json::{Map, Value};
use std::future::Future;

#[derive(FromPyObject)]
struct AudioTranscriptionInputs {
    model: String,
    audio: Py<PyAny>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    optional_params: Map<String, Value>,
}

fn prepare_transcription(
    input: AudioTranscriptionInputs,
    options: NativeRequestOptions,
    context: NativeRequestContext,
    _callback_adapter: Option<Py<PyAny>>,
    python_context: crate::execution::PythonCallContext<'_>,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let provider_admitted =
        litellm_core::audio_transcription::transcription_admitted(options.provider("bedrock"));
    let context: LiteLlmRequestContext = context.into();
    if let litellm_core::native_outcome::NativeOutcome::Declined(decline) =
        super::definition::admission(provider_admitted, &context)
    {
        return Err(crate::errors::RustBridgeDeclined::new_err(decline.reason()));
    }
    let py = python_context.py;
    let audio = py
        .import("litellm.rust_bridge.transcription")?
        .getattr("_consume_audio_for_native")?
        .call1((input.audio.bind(py),))?;
    let audio: Value = litellm_python_interop::from_py(&audio)?;
    Ok(async move {
        run_route(
            AudioTranscriptionRequest {
                model: &input.model,
                audio,
                optional_params: input.optional_params,
            },
            &options.into(),
            &context,
        )
        .await
    })
}

bridge_route! {
    sync = transcription,
    asynchronous = atranscription,
    request = AudioTranscriptionInputs,
    prepare = prepare_transcription,
    errors = core_error_to_pyerr,
}
