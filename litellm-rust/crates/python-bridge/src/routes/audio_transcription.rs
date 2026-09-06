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
    _callback_adapter: Option<Py<PyAny>>,
    _python_context: crate::execution::PythonCallContext<'_>,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let provider_supported = litellm_core::audio_transcription::transcription_provider_supported(
        options.provider("bedrock"),
    );
    let context: LiteLlmRequestContext = context.into();
    if let Some(reason) = super::definition::request_decline(provider_supported, &context) {
        return Err(crate::errors::RustBridgeDeclined::new_err(reason));
    }
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

#[pyfunction]
#[pyo3(signature = (_model, custom_llm_provider, *, context))]
fn transcription_decline(
    _model: &str,
    custom_llm_provider: &str,
    context: NativeRequestContext,
) -> Option<String> {
    let context: LiteLlmRequestContext = context.into();
    super::definition::request_decline(
        litellm_core::audio_transcription::transcription_provider_supported(custom_llm_provider),
        &context,
    )
}

bridge_route! {
    sync = transcription,
    asynchronous = atranscription,
    request = AudioTranscriptionInputs,
    prepare = prepare_transcription,
    errors = core_error_to_pyerr,
    extra = [transcription_decline],
}
