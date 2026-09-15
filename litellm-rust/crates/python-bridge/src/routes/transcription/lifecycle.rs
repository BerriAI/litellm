use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_core::audio_transcription::AudioTranscriptionAdmission;
use litellm_core::audio_transcription::lifecycle::{
    AudioTranscriptionRoute, OwnedAudioTranscriptionRequest,
};
use litellm_core::call_lifecycle::admission::Inspection;
use litellm_python_interop::from_py_preserving_errors as from_py;

use crate::lifecycle::completed::{self, PythonCompletedRoute};
use crate::lifecycle::contract::{PythonCallType, RequestField};
use crate::lifecycle::request::{
    exact_optional_object, exact_optional_string, object, optional_string, options, required,
};

impl PythonCompletedRoute for AudioTranscriptionRoute {
    const SYNC_CALL_TYPE: PythonCallType = PythonCallType::Transcription;
    const ASYNC_CALL_TYPE: PythonCallType = PythonCallType::AsyncTranscription;

    fn project_admission(request: &Bound<'_, PyDict>) -> PyResult<Self::Admission> {
        let model = required(request, RequestField::Model)?;
        let provider = request.get_item(RequestField::CustomLlmProvider.key(request.py()))?;
        let audio_value = required(request, RequestField::Audio)?;
        let optional_params = request.get_item(RequestField::OptionalParams.key(request.py()))?;
        if !exact_optional_string(Some(&model))
            || !exact_optional_string(provider.as_ref())
            || !exact_optional_object(Some(&audio_value))
            || !exact_optional_object(optional_params.as_ref())
        {
            return Ok(Inspection::Uninspectable);
        }
        let audio = from_py(&audio_value)?;
        Ok(Inspection::Inspectable(AudioTranscriptionAdmission {
            model: model.extract()?,
            provider: optional_string(request, RequestField::CustomLlmProvider)?,
            audio,
        }))
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
fn transcription(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<AudioTranscriptionRoute>(py, request, args, kwargs, false, host)
}

#[pyfunction]
fn atranscription(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<AudioTranscriptionRoute>(py, request, args, kwargs, true, host)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, wrap_pyfunction!(transcription, module)?)?;
    crate::routes::definition::add_function(module, wrap_pyfunction!(atranscription, module)?)
}
