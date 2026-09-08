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
    let (method, awaiting) = match operation {
        Operation::Setup => ("setup", false),
        Operation::DeploymentPre => ("deployment_pre", true),
        Operation::Prepare => ("prepare", false),
        Operation::PreCall => {
            return Err(PyRuntimeError::new_err(
                "messages lifecycle selected an unsupported pre-call operation",
            ));
        }
        Operation::Send if asynchronous => ("send", true),
        Operation::Send => ("send_sync", false),
        Operation::DeploymentSuccess => ("deployment_success", true),
        Operation::DeploymentFailure => ("deployment_failure", true),
        Operation::SyncSuccess => ("sync_success", false),
        Operation::AsyncSuccess => ("async_success", false),
        Operation::SyncSuccessIfNeeded => ("sync_success_if_needed", false),
        Operation::SyncFailure => ("sync_failure", false),
        Operation::AsyncFailure => ("async_failure", true),
        Operation::Restore => ("restore", false),
        Operation::Complete(_) => {
            return Err(PyRuntimeError::new_err("messages lifecycle is complete"));
        }
    };
    Ok((awaiting, host.getattr(py, method)?.call0(py)?))
}

#[pyfunction]
fn prepare(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Py<MessagesState>> {
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
    let logging = bag
        .get_item("litellm_logging_obj")?
        .filter(|value| !value.is_none())
        .map(|value| value.unbind())
        .ok_or_else(|| PyRuntimeError::new_err("messages logging was not initialized"))?;
    let additional = PyDict::new(py);
    additional.set_item(pyo3::intern!(py, "complete_input_dict"), &body)?;
    additional.set_item(pyo3::intern!(py, "api_base"), &prepared.url)?;
    additional.set_item(pyo3::intern!(py, "headers"), &headers)?;
    let kwargs = PyDict::new(py);
    let serialized = py.import("json")?.call_method1("dumps", (&body,))?;
    let message = PyDict::new(py);
    message.set_item("role", "user")?;
    message.set_item("content", serialized)?;
    kwargs.set_item("input", vec![message])?;
    kwargs.set_item("api_key", "")?;
    kwargs.set_item("additional_args", additional)?;
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
    driver(py)?.getattr("drive_sync")?.call1((arguments,))
}

#[pyfunction]
fn amessages(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    driver(py)?.getattr("drive_async")?.call1((arguments,))
}

fn driver(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static DRIVER: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = DRIVER.get(py) {
        return Ok(module.bind(py));
    }
    let module = crate::driver::compile(py, "messages", HOST)?;
    module.add("_Lifecycle", py.get_type::<MessagesLifecycle>())?;
    module.add("_invoke", wrap_pyfunction!(invoke, &module)?)?;
    module.add("_prepare", wrap_pyfunction!(prepare, &module)?)?;
    module.add("_send", wrap_pyfunction!(send, &module)?)?;
    module.add("_send_sync", wrap_pyfunction!(send_sync, &module)?)?;
    module.add(
        "_committed_failure",
        wrap_pyfunction!(committed_failure, &module)?,
    )?;
    Ok(DRIVER.get_or_init(py, || module.unbind()).bind(py))
}

const HOST: &str = r#"
from datetime import datetime
from litellm import utils
from litellm.types.utils import CallTypes
from litellm.rust_bridge.messages import initialize_logging, invoke_terminal, retain_stream_response

class Host:
    def __init__(self, arguments, asynchronous):
        self.machine = _Lifecycle(asynchronous, utils.is_internal_call.get())
        self.arguments = arguments
        self.current = arguments
        self.asynchronous = asynchronous
        self.logger = arguments.get('litellm_logging_obj')
        self.lifecycle_owned = self.logger is None
        self.state = None
        self.response = None
        self.error = None
        self.start = datetime.now()
        self.end = None
        self.streaming = False

    def setup(self):
        self.logger = initialize_logging(self.arguments, self.asynchronous)
        self.arguments['litellm_logging_obj'] = self.logger
        self.streaming = self.logger.stream is True

    async def deployment_pre(self):
        if not self.lifecycle_owned:
            return
        modified = await utils.async_pre_call_deployment_hook(self.current, 'anthropic_messages')
        if modified is not None:
            self.current = modified
        self.current['litellm_logging_obj'] = self.logger

    def prepare(self):
        self.state = _prepare(self.current)

    def send_sync(self):
        self.response = _send_sync(self.state)
        self.end = datetime.now()

    async def send(self):
        self.response = await _send(self.state)
        self.end = datetime.now()

    async def deployment_success(self):
        if self.lifecycle_owned:
            self.response = await utils.async_post_call_success_deployment_hook(self.current, self.response, CallTypes.aanthropic_messages)
        if self.streaming:
            self.response = retain_stream_response(
                self.response,
                (self.arguments, self.current, self.state),
                self.logger,
                self.start,
            )

    async def deployment_failure(self):
        if self.lifecycle_owned:
            await utils.async_post_call_failure_deployment_hook(self.current, self.error, 'anthropic_messages')

    def terminal(self, action, value):
        if self.streaming or not self.lifecycle_owned:
            return None
        return invoke_terminal(action, (self.arguments, self.current, self.state), self.logger, None, value, self.start, self.end)

    def sync_success(self): return self.terminal('sync_success', self.response)
    def async_success(self): return self.terminal('async_success', self.response)
    def sync_success_if_needed(self): return self.terminal('sync_success_if_needed', self.response)
    def sync_failure(self): return self.terminal('sync_failure', self.error)
    def async_failure(self): return self.terminal('async_failure', self.error)
    def restore(self):
        if not self.streaming and self.lifecycle_owned:
            utils._restore_correlation_context_if_supported(self.logger)

    def advance(self, outcome, error=None):
        if error is not None and self.end is None:
            self.end = datetime.now()
        if self.logger is None:
            self.logger = self.arguments.get('litellm_logging_obj')
        replace = self.machine.advance(outcome, self.logger is not None, self.current.get('fallbacks') is not None)
        if replace:
            self.error = error

    def result(self):
        if self.machine.complete():
            return self.response
        if self.machine.failed_after_provider_response():
            _committed_failure()
        raise self.error
"#;

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
