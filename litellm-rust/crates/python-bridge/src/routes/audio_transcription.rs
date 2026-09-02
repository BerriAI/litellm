use litellm_ai_gateway::io::audio_transcription::{
    AudioTranscriptionRequest, audio_transcription as run_audio_transcription,
};
use litellm_python_interop::from_py;
use pyo3::prelude::*;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{optional_object_to_map, optional_timeout};

use super::runtime::{run_async, run_sync};

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn transcription(
    py: Python<'_>,
    model: String,
    audio: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let audio = from_py(audio.bind(py))?;
    let extra_headers = match extra_headers {
        Some(headers) => Some(optional_object_to_map(py, "extra_headers", Some(headers))?),
        None => None,
    };
    let optional_params = optional_object_to_map(py, "optional_params", optional_params)?;
    let timeout = optional_timeout(timeout_seconds);
    run_sync(
        py,
        async move {
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
        },
        core_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn atranscription(
    py: Python<'_>,
    model: String,
    audio: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let audio = from_py(audio.bind(py))?;
    let extra_headers = match extra_headers {
        Some(headers) => Some(optional_object_to_map(py, "extra_headers", Some(headers))?),
        None => None,
    };
    let optional_params = optional_object_to_map(py, "optional_params", optional_params)?;
    let timeout = optional_timeout(timeout_seconds);
    run_async(
        py,
        async move {
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
        },
        core_error_to_pyerr,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(transcription, module)?)?;
    module.add_function(wrap_pyfunction!(atranscription, module)?)
}
