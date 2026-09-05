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
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    audio: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    optional_params: Map<String, Value>,
}

fn prepare_transcription(
    input: AudioTranscriptionInputs,
    options: NativeRequestOptions,
    context: NativeRequestContext,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let context: LiteLlmRequestContext = context.into();
    let audio = input.audio;
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
