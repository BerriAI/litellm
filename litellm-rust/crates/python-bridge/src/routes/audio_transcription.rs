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
    if let Some(reason) = transcription_decline(
        &input.model,
        input.options.provider("bedrock"),
        input
            .optional_params
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        false,
        false,
        input
            .optional_params
            .get("response_format")
            .and_then(Value::as_str),
    ) {
        return Err(crate::errors::RustBridgeDeclined::new_err(reason));
    }
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

#[pyfunction]
#[pyo3(signature = (_model, custom_llm_provider, *, stream=false, has_agentic_hook=false, has_custom_client=false, request_format=None))]
fn transcription_decline(
    _model: &str,
    custom_llm_provider: &str,
    stream: bool,
    has_agentic_hook: bool,
    has_custom_client: bool,
    request_format: Option<&str>,
) -> Option<String> {
    super::definition::request_decline(
        litellm_core::audio_transcription::transcription_provider_supported(custom_llm_provider),
        stream,
        has_agentic_hook,
        has_custom_client,
        request_format,
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
