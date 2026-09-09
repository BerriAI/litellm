use super::streaming;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use litellm_core::lifecycle::FailureStage;
use litellm_core::lifecycle::{ErrorDisposition, Lifecycle, Outcome};
use litellm_core::messages::lifecycle::{MessagesRoute, Observations, Operation, Options, machine};
use litellm_core::messages::types::{MessagesEndpoint, MessagesOptions, ProviderMessagesRequest};
use litellm_python_interop::{Pythonized, from_py, run_async_value, run_sync_value};

use crate::arguments::MessagesArguments;
use crate::callbacks::pre_call_args;
use crate::errors::{Error, Route, core_error_to_pyerr, messages_provider_error_to_pyerr};
use crate::marshal::optional_timeout;
use crate::retained::{RequestRoots, RetainedCallback};

#[pyclass]
struct MessagesState {
    streaming: bool,
    callback: RetainedCallback,
    pending: Option<(
        MessagesEndpoint,
        litellm_core::lifecycle::PreCallBody<Value>,
    )>,
}

#[pymethods]
impl MessagesState {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.callback.traverse(&visit)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let retained = {
            let mut state = slf.borrow_mut();
            (state.callback.clear(), state.pending.take())
        };
        drop(retained);
    }
}

fn decode_options(arguments: &MessagesArguments<'_>) -> PyResult<MessagesOptions> {
    Ok(MessagesOptions {
        model: arguments
            .model
            .clone()
            .ok_or_else(|| PyValueError::new_err("messages requires model"))?,
        api_key: arguments.api_key.clone(),
        api_base: arguments.api_base.clone(),
        custom_llm_provider: arguments.custom_llm_provider.clone(),
        extra_headers: arguments
            .extra_headers
            .as_ref()
            .map(from_py::<Map<String, Value>>)
            .transpose()?,
        timeout: optional_timeout(arguments.timeout_seconds)?,
    })
}

#[pyclass]
struct MessagesLifecycle {
    machine: Lifecycle<MessagesRoute>,
    asynchronous: bool,
    pending_operation: Option<litellm_core::lifecycle::program::OperationTicket>,
}

impl MessagesLifecycle {
    fn preparation_permit(
        &mut self,
    ) -> PyResult<litellm_core::lifecycle::program::PreparationPermit<MessagesRoute>> {
        let ticket = self.pending_operation.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no build operation is pending")
        })?;
        self.machine
            .preparation_permit(ticket)
            .map_err(core_error_to_pyerr)
    }

    fn provider_permit(
        &mut self,
    ) -> PyResult<litellm_core::lifecycle::program::ProviderPermit<MessagesRoute>> {
        let ticket = self.pending_operation.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no send operation is pending")
        })?;
        self.machine
            .provider_permit(ticket)
            .map_err(core_error_to_pyerr)
    }
}

#[pymethods]
impl MessagesLifecycle {
    #[new]
    fn new(asynchronous: bool, internal_call: bool) -> PyResult<Self> {
        Ok(Self {
            machine: machine(Options {
                asynchronous,
                internal_call,
                ..Options::default()
            })
            .map_err(core_error_to_pyerr)?,
            asynchronous,
            pending_operation: None,
        })
    }

    fn advance(
        &mut self,
        outcome: u8,
        logger_available: bool,
        has_fallbacks: bool,
    ) -> PyResult<bool> {
        let outcome = match outcome {
            0 => Outcome::Success,
            1 => Outcome::Failure,
            _ => Outcome::Abort,
        };
        let ticket = self.pending_operation.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no lifecycle operation is pending")
        })?;
        self.machine
            .complete_operation(
                ticket,
                outcome,
                Observations {
                    logger_available,
                    has_fallbacks,
                },
            )
            .map(|transition| transition.error == ErrorDisposition::Replace)
            .map_err(core_error_to_pyerr)
    }

    fn complete(&self) -> Option<bool> {
        match self.machine.operation() {
            Operation::Complete(outcome) => Some(outcome == Outcome::Success),
            _ => None,
        }
    }

    fn failed_after_provider_response(&self) -> bool {
        self.machine.failure_stage() == Some(FailureStage::AfterProviderResponse)
    }
}

#[pyfunction]
fn invoke(
    py: Python<'_>,
    machine: Py<MessagesLifecycle>,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let (operation, asynchronous) = {
        let mut machine = machine.borrow_mut(py);
        let ticket = machine.machine.issue().map_err(core_error_to_pyerr)?;
        let operation = ticket.operation();
        machine.pending_operation = Some(ticket);
        (operation, machine.asynchronous)
    };
    crate::driver::invoke(py, operation, asynchronous, Route::Messages, host)
}

#[pyfunction]
fn build_request(
    py: Python<'_>,
    machine: Py<MessagesLifecycle>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
) -> PyResult<Py<MessagesState>> {
    let permit = machine.borrow_mut(py).preparation_permit()?;
    let bag = arguments.bind(py);
    validate_arguments(bag)?;
    let call = bag.extract::<MessagesArguments<'_>>()?;
    let options = decode_options(&call)?;
    let endpoint = py
        .detach(|| permit.messages(options))
        .map_err(core_error_to_pyerr)?;
    let body = call.body.cast_into::<PyDict>()?;
    let callback_body: Value = from_py(body.as_any())?;
    let streaming = callback_body.get("stream").and_then(Value::as_bool) == Some(true);
    let snapshot = if streaming {
        litellm_core::lifecycle::PreCallBody::StructuredAtBuild {
            authorized: endpoint
                .capture_body(callback_body.clone())
                .map_err(core_error_to_pyerr)?,
            callback: callback_body,
        }
    } else {
        endpoint
            .capture_buffered_body(callback_body)
            .map_err(core_error_to_pyerr)?
    };
    let headers = PyDict::new(py);
    for (name, value) in endpoint.headers() {
        headers.set_item(name, value)?;
    }
    let serialized = py.import("json")?.call_method1("dumps", (&body,))?;
    let message = PyDict::new(py);
    message.set_item("role", "user")?;
    message.set_item("content", serialized)?;
    let kwargs =
        pre_call_args(vec![message], "", &body, endpoint.url(), &headers).into_pyobject(py)?;
    Py::new(
        py,
        MessagesState {
            streaming,
            callback: RetainedCallback::new(
                RequestRoots::new(
                    arguments,
                    body.unbind().into_any(),
                    headers.unbind().into_any(),
                ),
                logging,
                kwargs.unbind(),
            ),
            pending: Some((endpoint, snapshot)),
        },
    )
}

#[pyfunction]
fn pre_call(py: Python<'_>, state: Py<MessagesState>) -> PyResult<()> {
    let (logging, arguments) = {
        let state = state.borrow(py);
        (
            state.callback.logging(py, Route::Messages)?,
            state.callback.pre_call(py, Route::Messages)?,
        )
    };
    logging
        .bind(py)
        .call_method(pyo3::intern!(py, "pre_call"), (), Some(arguments.bind(py)))?;
    Ok(())
}

fn take_request(py: Python<'_>, state: &Py<MessagesState>) -> PyResult<ProviderMessagesRequest> {
    let ((endpoint, pre_call_body), headers) = {
        let mut state = state.borrow_mut(py);
        let pending = state
            .pending
            .take()
            .ok_or(Error::RequestConsumed(Route::Messages))?;
        let roots = state.callback.roots(Route::Messages)?;
        (pending, roots.headers(py))
    };
    let headers = headers
        .cast::<PyDict>()?
        .iter()
        .map(|(name, value)| Ok((name.extract()?, value.extract()?)))
        .collect::<PyResult<_>>()?;
    let litellm_core::lifecycle::PreCallBody::StructuredAtBuild { authorized, .. } = pre_call_body
    else {
        return Err(PyRuntimeError::new_err(
            "messages requires a structured-at-build body",
        ));
    };
    Ok(endpoint.settle(authorized, headers))
}

fn validate_arguments(arguments: &Bound<'_, PyDict>) -> PyResult<()> {
    let body = arguments
        .get_item("body")?
        .ok_or_else(|| PyValueError::new_err("messages requires body"))?;
    if !body.is_instance_of::<PyDict>() {
        return Err(PyTypeError::new_err("body must be a dict"));
    }
    if let Some(headers) = arguments
        .get_item("extra_headers")?
        .filter(|value| !value.is_none())
        && !headers.is_instance_of::<PyDict>()
    {
        return Err(PyTypeError::new_err("extra_headers must be a dict"));
    }
    Ok(())
}

#[pyfunction]
fn send(
    py: Python<'_>,
    machine: Py<MessagesLifecycle>,
    state: Py<MessagesState>,
    host: Py<PyAny>,
) -> PyResult<Bound<'_, PyAny>> {
    let permit = machine.borrow_mut(py).provider_permit()?;
    let streaming = state.borrow(py).streaming;
    let request = take_request(py, &state)?;
    if streaming {
        return streaming::send(py, permit, request, host);
    }
    litellm_python_interop::run_async_py(py, async move {
        let _state = state;
        let response =
            run_async_value(permit.messages(request), messages_provider_error_to_pyerr).await?;
        Ok(Pythonized(response))
    })
}

#[pyfunction]
fn send_sync(
    py: Python<'_>,
    machine: Py<MessagesLifecycle>,
    state: Py<MessagesState>,
) -> PyResult<Py<PyAny>> {
    let permit = machine.borrow_mut(py).provider_permit()?;
    let request = take_request(py, &state)?;
    let response = run_sync_value(
        py,
        permit.messages(request),
        messages_provider_error_to_pyerr,
    )?;
    Ok(Pythonized(response).into_pyobject(py)?.unbind().into_any())
}

#[pyfunction]
fn committed_failure() -> PyResult<()> {
    Err(Error::upstream(0, "Messages lifecycle failed after the provider returned").into())
}

#[pyfunction]
fn messages(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    let body = arguments
        .bind(py)
        .get_item("body")?
        .ok_or_else(|| PyValueError::new_err("messages requires body"))?;
    if body
        .cast::<PyDict>()?
        .get_item("stream")?
        .is_some_and(|value| value.extract::<bool>().unwrap_or(false))
    {
        return Err(Error::declined("synchronous Messages streaming is not supported").into());
    }
    crate::driver::drive_sync(py, runner(py)?, arguments, bindings(py)?)
}

#[pyfunction]
fn amessages(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    crate::driver::drive_async(py, runner(py)?, arguments, bindings(py)?)
}

fn runner(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static RUNNER: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = RUNNER.get(py) {
        return Ok(module.bind(py));
    }
    let module = py.import("litellm.rust_bridge.messages")?;
    Ok(RUNNER.get_or_init(py, || module.unbind()).bind(py))
}

fn bindings(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static BINDINGS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = BINDINGS.get(py) {
        return Ok(module.bind(py));
    }
    let module = PyModule::new(py, "_messages_bindings")?;
    module.add("Lifecycle", py.get_type::<MessagesLifecycle>())?;
    module.add("invoke", wrap_pyfunction!(invoke, &module)?)?;
    module.add("build_request", wrap_pyfunction!(build_request, &module)?)?;
    module.add("pre_call", wrap_pyfunction!(pre_call, &module)?)?;
    module.add("send", wrap_pyfunction!(send, &module)?)?;
    module.add("send_sync", wrap_pyfunction!(send_sync, &module)?)?;
    module.add(
        "committed_failure",
        wrap_pyfunction!(committed_failure, &module)?,
    )?;
    Ok(BINDINGS.get_or_init(py, || module.unbind()).bind(py))
}

pub(in crate::routes) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, wrap_pyfunction!(messages, module)?)?;
    crate::routes::definition::add_function(module, wrap_pyfunction!(amessages, module)?)?;
    Ok(())
}

#[cfg(feature = "trace-parity")]
pub(in crate::routes) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register(module)
}

#[cfg(test)]
#[path = "../../../tests/unit/routes/messages/bridge.rs"]
mod tests;
