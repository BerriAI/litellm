use litellm_ai_gateway::io::audio_transcription::{
    AudioTranscriptionRequest, audio_transcription as run_audio_transcription,
};
use litellm_core::error::CoreResult;
use litellm_python_interop::from_py;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::marshal::{RouteOptions, object_or_empty};
use crate::routes::BridgeRoute;

struct AudioTranscriptionCall {
    options: RouteOptions,
    audio: Value,
    optional_params: Map<String, Value>,
}

impl BridgeRoute<AudioTranscriptionInputs> for AudioTranscriptionCall {
    type Output = Value;

    fn from_python(py: Python<'_>, inputs: AudioTranscriptionInputs) -> PyResult<Self> {
        Ok(Self {
            options: RouteOptions::from_python(
                py,
                inputs.model,
                inputs.api_key,
                inputs.api_base,
                inputs.custom_llm_provider,
                inputs.extra_headers,
                inputs.timeout_seconds,
            )?,
            audio: from_py(inputs.audio.bind(py))?,
            optional_params: object_or_empty(py, "optional_params", inputs.optional_params)?,
        })
    }

    async fn run(self) -> CoreResult<Value> {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = self.options;
        run_audio_transcription(AudioTranscriptionRequest {
            model: &model,
            audio: self.audio,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params: self.optional_params,
            timeout,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        })
        .await
    }
}

bridge_route! {
    sync = transcription,
    asynchronous = atranscription,
    inputs = AudioTranscriptionInputs,
    required = {
        model: String,
        audio: Py<PyAny>,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        optional_params: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    },
    call = AudioTranscriptionCall,
    errors = core_error_to_pyerr,
}
