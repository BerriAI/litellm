use litellm_core::call_lifecycle::admission::Inspection;
use litellm_core::chat_completions::ChatCompletionsAdmission;
use litellm_core::chat_completions::lifecycle::{
    ChatCompletionsRoute, OwnedChatCompletionsRequest,
};
use litellm_python_interop::from_py_preserving_errors as from_py;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::lifecycle::completed::{self, PythonCompletedRoute};
use crate::lifecycle::contract::{PythonCallType, RequestField};
use crate::lifecycle::request::{
    exact_list, exact_optional_object, exact_optional_string, object, options, required,
};

impl PythonCompletedRoute for ChatCompletionsRoute {
    const SYNC_CALL_TYPE: PythonCallType = PythonCallType::Completion;
    const ASYNC_CALL_TYPE: PythonCallType = PythonCallType::AsyncCompletion;

    fn project_admission(request: &Bound<'_, PyDict>) -> PyResult<Self::Admission> {
        let model = required(request, RequestField::Model)?;
        let provider = request.get_item(RequestField::CustomLlmProvider.key(request.py()))?;
        let messages = required(request, RequestField::Messages)?;
        let params = request.get_item(RequestField::OptionalParams.key(request.py()))?;
        let headers = request.get_item(RequestField::ExtraHeaders.key(request.py()))?;
        let facts = request.get_item(RequestField::HostFacts.key(request.py()))?;
        if !exact_optional_string(Some(&model))
            || !exact_optional_string(provider.as_ref())
            || !exact_list(&messages)
            || !exact_optional_object(params.as_ref())
            || !exact_optional_object(headers.as_ref())
            || !exact_optional_object(facts.as_ref())
        {
            return Ok(Inspection::Uninspectable);
        }
        let provider: Option<String> = provider
            .as_ref()
            .map(|value| value.extract::<Option<String>>())
            .transpose()?
            .flatten();
        Ok(Inspection::Inspectable(ChatCompletionsAdmission {
            model: model.extract()?,
            provider,
            messages: from_py(&messages)?,
            params: object(request, RequestField::OptionalParams)?,
            headers: Some(object(request, RequestField::ExtraHeaders)?),
            context: facts
                .map(|value| from_py(&value))
                .transpose()?
                .unwrap_or_default(),
        }))
    }

    fn project(request: &Bound<'_, PyDict>) -> PyResult<OwnedChatCompletionsRequest> {
        Ok(OwnedChatCompletionsRequest {
            options: options(request)?,
            messages: from_py(&required(request, RequestField::Messages)?)?,
            optional_params: object(request, RequestField::OptionalParams)?,
        })
    }
}

#[pyfunction]
fn chat_completions(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<ChatCompletionsRoute>(py, request, args, kwargs, false, host)
}

#[pyfunction]
fn achat_completions(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<ChatCompletionsRoute>(py, request, args, kwargs, true, host)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, wrap_pyfunction!(chat_completions, module)?)?;
    crate::routes::definition::add_function(module, wrap_pyfunction!(achat_completions, module)?)
}
