use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_core::audio_transcription::lifecycle::{
    AudioTranscriptionRoute, OwnedAudioTranscriptionRequest,
};
use litellm_python_interop::from_py_preserving_errors as from_py;

use crate::lifecycle::completed::{self, PythonCompletedRoute};
use crate::lifecycle::contract::{PythonCallType, RequestField};
use crate::lifecycle::request::{object, optional_string, options, required};

impl PythonCompletedRoute for AudioTranscriptionRoute {
    const SYNC_CALL_TYPE: PythonCallType = PythonCallType::Transcription;
    const ASYNC_CALL_TYPE: PythonCallType = PythonCallType::AsyncTranscription;

    fn admit(request: &Bound<'_, PyDict>) -> PyResult<()> {
        let audio = from_py(&required(request, RequestField::Audio)?)?;
        crate::errors::admit(litellm_core::audio_transcription::admit(
            &required(request, RequestField::Model)?.extract::<String>()?,
            optional_string(request, RequestField::CustomLlmProvider)?.as_deref(),
            &audio,
        ))
    }

    fn project(request: &Bound<'_, PyDict>) -> PyResult<OwnedAudioTranscriptionRequest> {
        Ok(OwnedAudioTranscriptionRequest {
            options: options(request)?,
            audio: from_py(&required(request, RequestField::Audio)?)?,
            optional_params: object(request, RequestField::OptionalParams)?,
        })
    }
}

#[pyfunction]
fn _transcription_lifecycle(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<AudioTranscriptionRoute>(py, request, args, kwargs, asynchronous, host)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(
        module,
        wrap_pyfunction!(_transcription_lifecycle, module)?,
    )
}
