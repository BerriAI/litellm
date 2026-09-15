use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_core::messages::lifecycle::{MessagesRoute, OwnedMessagesRequest};
use litellm_python_interop::from_py_preserving_errors as from_py;

use crate::lifecycle::completed::{self, PythonCompletedRoute};
use crate::lifecycle::contract::{PythonCallType, RequestField};
use crate::lifecycle::request::{
    exact_optional_bool, exact_optional_object, exact_optional_string, options, required,
};

impl PythonCompletedRoute for MessagesRoute {
    const SYNC_CALL_TYPE: PythonCallType = PythonCallType::AnthropicMessages;
    const ASYNC_CALL_TYPE: PythonCallType = PythonCallType::AnthropicMessages;

    fn admit(request: &Bound<'_, PyDict>) -> PyResult<()> {
        let model = required(request, RequestField::Model)?;
        let provider = request.get_item(RequestField::CustomLlmProvider.key(request.py()))?;
        let body = request.get_item(RequestField::Body.key(request.py()))?;
        let host_hook = request.get_item(RequestField::HasAgenticHook.key(request.py()))?;
        if !exact_optional_string(Some(&model))
            || !exact_optional_string(provider.as_ref())
            || !exact_optional_object(body.as_ref())
            || !exact_optional_bool(host_hook.as_ref())
        {
            return crate::errors::admit(Err(
                litellm_core::call_lifecycle::admission::AdmissionDecline::Uninspectable,
            ));
        }
        let provider: Option<String> = provider
            .as_ref()
            .map(|value| value.extract::<Option<String>>())
            .transpose()?
            .flatten();
        crate::errors::admit(litellm_core::messages::admit(
            &model.extract::<String>()?,
            provider.as_deref(),
            host_hook
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
fn messages(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<MessagesRoute>(py, request, args, kwargs, false, host)
}

#[pyfunction]
fn amessages(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    completed::run::<MessagesRoute>(py, request, args, kwargs, true, host)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, wrap_pyfunction!(messages, module)?)?;
    crate::routes::definition::add_function(module, wrap_pyfunction!(amessages, module)?)
}
