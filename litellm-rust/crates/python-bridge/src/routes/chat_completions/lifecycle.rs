use litellm_core::chat_completions::lifecycle::{
    ChatCompletionsRoute, OwnedChatCompletionsRequest,
};
use litellm_python_interop::from_py_preserving_errors as from_py;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::lifecycle::completed::{self, PythonCompletedRoute};
use crate::lifecycle::contract::{PythonCallType, RequestField};
use crate::lifecycle::request::{object, optional_string, options, required};

impl PythonCompletedRoute for ChatCompletionsRoute {
    const SYNC_CALL_TYPE: PythonCallType = PythonCallType::Completion;
    const ASYNC_CALL_TYPE: PythonCallType = PythonCallType::AsyncCompletion;

    fn admit(request: &Bound<'_, PyDict>) -> PyResult<()> {
        crate::errors::admit(litellm_core::chat_completions::admit(
            &required(request, RequestField::Model)?.extract::<String>()?,
            optional_string(request, RequestField::CustomLlmProvider)?.as_deref(),
            from_py(&required(request, RequestField::Messages)?)?,
            &object(request, RequestField::OptionalParams)?,
            Some(&object(request, RequestField::ExtraHeaders)?),
            request
                .get_item(RequestField::HostFacts.key(request.py()))?
                .map(|value| from_py(&value))
                .transpose()?
                .unwrap_or_default(),
        ))
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
fn _chat_completions_lifecycle(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<ChatCompletionsRoute>(py, request, args, kwargs, asynchronous, host)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(
        module,
        wrap_pyfunction!(_chat_completions_lifecycle, module)?,
    )
}
