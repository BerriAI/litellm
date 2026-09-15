use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_core::messages::lifecycle::{MessagesRoute, OwnedMessagesRequest};
use litellm_python_interop::from_py_preserving_errors as from_py;

use crate::lifecycle::completed::{self, PythonCompletedRoute};
use crate::lifecycle::contract::{PythonCallType, RequestField};
use crate::lifecycle::request::{optional_string, options, required};

impl PythonCompletedRoute for MessagesRoute {
    const SYNC_CALL_TYPE: PythonCallType = PythonCallType::AnthropicMessages;
    const ASYNC_CALL_TYPE: PythonCallType = PythonCallType::AnthropicMessages;

    fn admit(request: &Bound<'_, PyDict>) -> PyResult<()> {
        crate::errors::admit(litellm_core::messages::admit(
            &required(request, RequestField::Model)?.extract::<String>()?,
            optional_string(request, RequestField::CustomLlmProvider)?.as_deref(),
            request
                .get_item(RequestField::HasAgenticHook.key(request.py()))?
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or(false),
        ))
    }

    fn project(request: &Bound<'_, PyDict>) -> PyResult<OwnedMessagesRequest> {
        Ok(OwnedMessagesRequest {
            options: options(request)?,
            body: from_py(&required(request, RequestField::Body)?)?,
        })
    }
}

#[pyfunction]
fn _messages_lifecycle(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<MessagesRoute>(py, request, args, kwargs, asynchronous, host)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, wrap_pyfunction!(_messages_lifecycle, module)?)
}
