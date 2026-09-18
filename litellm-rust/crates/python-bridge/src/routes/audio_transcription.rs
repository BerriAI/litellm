use litellm_core::audio_transcription::{AudioTranscriptionRequest, audio_transcription};
use litellm_core::timeout;
use litellm_host_python::{from_py_argument, run_async, run_sync};
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::audio_transcription_error_to_pyerr;
use crate::marshal::{extra_headers_argument, optional_params_argument};

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn transcription(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = from_py_argument)] audio: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let request = AudioTranscriptionRequest {
        model,
        audio,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params: optional_params.unwrap_or_default(),
        timeout: timeout::from_seconds(timeout_seconds),
    };
    run_sync(
        py,
        audio_transcription(request),
        audio_transcription_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn atranscription<'py>(
    py: Python<'py>,
    model: String,
    #[pyo3(from_py_with = from_py_argument)] audio: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let request = AudioTranscriptionRequest {
        model,
        audio,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params: optional_params.unwrap_or_default(),
        timeout: timeout::from_seconds(timeout_seconds),
    };
    run_async(
        py,
        audio_transcription(request),
        audio_transcription_error_to_pyerr,
    )
}
