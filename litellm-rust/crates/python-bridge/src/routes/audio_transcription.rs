use std::time::Duration;

use litellm_core::audio_transcription::{
    AudioTranscriptionRequest, audio_transcription as run_audio_transcription,
};
use litellm_core::error::Error;
use litellm_python_interop::from_py;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::execution::{run_async, run_sync};
use crate::function_trace::trace_call;
use crate::marshal::{optional_object, optional_object_to_map, optional_timeout};

struct TranscriptionInputs {
    model: String,
    audio: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    optional_params: Map<String, Value>,
    timeout: Option<Duration>,
}

#[allow(clippy::too_many_arguments)]
fn marshal_inputs(
    py: Python<'_>,
    model: String,
    audio: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<TranscriptionInputs> {
    Ok(TranscriptionInputs {
        model,
        audio: from_py(audio.bind(py))?,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers: optional_object(py, "extra_headers", extra_headers)?,
        optional_params: optional_object_to_map(py, "optional_params", optional_params)?,
        timeout: optional_timeout(timeout_seconds),
    })
}

async fn call(inputs: TranscriptionInputs) -> Result<Value, Error> {
    run_audio_transcription(AudioTranscriptionRequest {
        model: &inputs.model,
        audio: inputs.audio,
        api_key: inputs.api_key.as_deref(),
        api_base: inputs.api_base.as_deref(),
        custom_llm_provider: inputs.custom_llm_provider.as_deref(),
        extra_headers: inputs.extra_headers,
        optional_params: inputs.optional_params,
        timeout: inputs.timeout,
    })
    .await
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None, trace=false))]
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
    trace: bool,
) -> PyResult<Py<PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        audio,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;
    run_sync(py, trace_call(call(inputs), trace), core_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None, trace=false))]
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
    trace: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        audio,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;
    run_async(py, trace_call(call(inputs), trace), core_error_to_pyerr)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(transcription, module)?)?;
    module.add_function(wrap_pyfunction!(atranscription, module)?)
}
