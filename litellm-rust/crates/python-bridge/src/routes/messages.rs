use litellm_core::lifecycle::FailureStage;
use litellm_core::lifecycle::{ErrorDisposition, Lifecycle, Outcome};
use litellm_core::messages::lifecycle::{MessagesRoute, Observations, Operation, Options, machine};
use litellm_core::messages::types::{MessagesRequest, ProviderMessagesRequest};
use litellm_core::messages::{execute_prepared_messages_provider_call, prepare_provider_request};
use litellm_python_interop::{Pythonized, from_py, run_async_value, run_sync_value, to_py};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use crate::driver::{ADDITIONAL_ARGS, API_BASE, API_KEY, COMPLETE_INPUT_DICT, HEADERS, INPUT};
use crate::errors::{RustUpstreamError, core_error_to_pyerr, messages_provider_error_to_pyerr};
use crate::marshal::optional_timeout;

#[pyclass]
struct MessagesState {
    arguments: Option<Py<PyDict>>,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyDict>>,
    prepared: Option<ProviderMessagesRequest>,
}

#[pymethods]
impl MessagesState {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.arguments)?;
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let roots = {
            let mut state = slf.borrow_mut();
            (
                state.arguments.take(),
                state.body.take(),
                state.headers.take(),
                state.prepared.take(),
            )
        };
        drop(roots);
    }
}

fn scalar(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    arguments
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()
}

fn decode_request(py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<MessagesRequest> {
    let body = arguments
        .get_item(pyo3::intern!(py, "body"))?
        .ok_or_else(|| PyValueError::new_err("messages requires body"))?;
    let timeout = optional_timeout(
        arguments
            .get_item(pyo3::intern!(py, "timeout_seconds"))?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<f64>())
            .transpose()?,
    )?;
    Ok(MessagesRequest {
        body: from_py(&body)?,
        model: scalar(arguments, "model")?
            .ok_or_else(|| PyValueError::new_err("messages requires model"))?,
        api_key: scalar(arguments, "api_key")?,
        api_base: scalar(arguments, "api_base")?,
        custom_llm_provider: scalar(arguments, "custom_llm_provider")?,
        extra_headers: arguments
            .get_item("extra_headers")?
            .filter(|value| !value.is_none())
            .map(|value| from_py::<Map<String, Value>>(&value))
            .transpose()?,
        timeout,
    })
}

#[pyclass]
struct MessagesLifecycle {
    machine: Lifecycle<MessagesRoute>,
    asynchronous: bool,
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
        self.machine
            .advance(
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
        let machine = machine.borrow(py);
        (machine.machine.operation(), machine.asynchronous)
    };
    crate::driver::invoke(py, operation, asynchronous, false, "messages", host)
}

#[pyfunction]
fn prepare(
    py: Python<'_>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
) -> PyResult<Py<MessagesState>> {
    let bag = arguments.bind(py);
    let request = decode_request(py, bag)?;
    let prepared = py
        .detach(|| prepare_provider_request(request))
        .map_err(core_error_to_pyerr)?;
    let body = to_py(py, &prepared.body)?
        .into_bound(py)
        .cast_into::<PyDict>()?;
    let headers = PyDict::new(py);
    for (name, value) in &prepared.upstream_headers {
        headers.set_item(name, value)?;
    }
    let additional = PyDict::new(py);
    additional.set_item(COMPLETE_INPUT_DICT, &body)?;
    additional.set_item(API_BASE, &prepared.url)?;
    additional.set_item(HEADERS, &headers)?;
    let kwargs = PyDict::new(py);
    let serialized = py.import("json")?.call_method1("dumps", (&body,))?;
    let message = PyDict::new(py);
    message.set_item("role", "user")?;
    message.set_item("content", serialized)?;
    kwargs.set_item(INPUT, vec![message])?;
    kwargs.set_item(API_KEY, "")?;
    kwargs.set_item(ADDITIONAL_ARGS, additional)?;
    logging
        .bind(py)
        .call_method(pyo3::intern!(py, "pre_call"), (), Some(&kwargs))?;
    Py::new(
        py,
        MessagesState {
            arguments: Some(arguments),
            body: Some(body.unbind()),
            headers: Some(headers.unbind()),
            prepared: Some(prepared),
        },
    )
}

fn take_request(py: Python<'_>, state: &Py<MessagesState>) -> PyResult<ProviderMessagesRequest> {
    let (mut prepared, body, headers) = {
        let mut state = state.borrow_mut(py);
        let prepared = state.prepared.take().ok_or_else(|| {
            PyRuntimeError::new_err("messages request was already sent or cleared")
        })?;
        let body = state
            .body
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("messages body was cleared"))?
            .clone_ref(py);
        let headers = state
            .headers
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("messages headers were cleared"))?
            .clone_ref(py);
        (prepared, body, headers)
    };
    prepared.body = from_py(body.bind(py).as_any())?;
    prepared.upstream_headers = headers
        .bind(py)
        .iter()
        .map(|(name, value)| Ok((name.extract()?, value.extract()?)))
        .collect::<PyResult<_>>()?;
    Ok(prepared)
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
fn send(py: Python<'_>, state: Py<MessagesState>) -> PyResult<Bound<'_, PyAny>> {
    let request = take_request(py, &state)?;
    litellm_python_interop::run_async_py(py, async move {
        let _state = state;
        let response = run_async_value(
            execute_prepared_messages_provider_call(request),
            messages_provider_error_to_pyerr,
        )
        .await?;
        Ok(Pythonized(response))
    })
}

#[pyfunction]
fn send_sync(py: Python<'_>, state: Py<MessagesState>) -> PyResult<Py<PyAny>> {
    let request = take_request(py, &state)?;
    let response = run_sync_value(
        py,
        execute_prepared_messages_provider_call(request),
        messages_provider_error_to_pyerr,
    )?;
    Ok(Pythonized(response).into_pyobject(py)?.unbind().into_any())
}

#[pyfunction]
fn committed_failure() -> PyResult<()> {
    Err(RustUpstreamError::new_err((
        0u16,
        "Messages lifecycle failed after the provider returned",
    )))
}

#[pyfunction]
fn messages(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    runner(py)?
        .getattr("_drive_sync")?
        .call1((arguments, bindings(py)?))
}

#[pyfunction]
fn amessages(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    runner(py)?
        .getattr("_drive_async")?
        .call1((arguments, bindings(py)?))
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
    module.add("prepare", wrap_pyfunction!(prepare, &module)?)?;
    module.add("send", wrap_pyfunction!(send, &module)?)?;
    module.add("send_sync", wrap_pyfunction!(send_sync, &module)?)?;
    module.add(
        "committed_failure",
        wrap_pyfunction!(committed_failure, &module)?,
    )?;
    Ok(BINDINGS.get_or_init(py, || module.unbind()).bind(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, wrap_pyfunction!(messages, module)?)?;
    crate::routes::definition::add_function(module, wrap_pyfunction!(amessages, module)?)?;
    Ok(())
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_state_rejects_invalid_timeouts_without_panicking() {
        Python::initialize();
        Python::attach(|py| {
            for timeout in [-1.0, 0.0, f64::NAN, f64::INFINITY, f64::MAX] {
                let arguments = PyDict::new(py);
                arguments.set_item("model", "model").unwrap();
                arguments.set_item("body", PyDict::new(py)).unwrap();
                arguments.set_item("timeout_seconds", timeout).unwrap();
                let error = match decode_request(py, &arguments) {
                    Ok(_) => panic!("invalid timeout should fail normally"),
                    Err(error) => error,
                };
                assert!(error.is_instance_of::<PyValueError>(py));
            }
        });
    }
}
