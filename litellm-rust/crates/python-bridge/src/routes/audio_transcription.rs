use std::future::Future;

use litellm_ai_gateway::io::audio_transcription::{
    AudioTranscriptionRequest, audio_transcription as run_audio_transcription,
};
use litellm_core::error::CoreResult;
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

fn prepare_transcription(
    inputs: AudioTranscriptionInputs,
) -> PyResult<impl Future<Output = CoreResult<Value>> + Send + 'static> {
    let audio = inputs.audio;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;

    Ok(async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        run_audio_transcription(AudioTranscriptionRequest {
            model: &model,
            audio,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params,
            timeout,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        })
        .await
    })
}

bridge_route! {
    sync = transcription,
    asynchronous = atranscription,
    inputs = AudioTranscriptionInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        audio: Value,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<Value>,
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_transcription,
    errors = core_error_to_pyerr,
}
