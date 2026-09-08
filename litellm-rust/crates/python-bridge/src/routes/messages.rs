use litellm_core::lifecycle::FailureStage;
use litellm_core::lifecycle::{ErrorDisposition, Lifecycle, Outcome};
use litellm_core::messages::execute_provider_messages_request;
use litellm_core::messages::lifecycle::{MessagesRoute, Observations, Operation, Options, machine};
use litellm_core::messages::request::build_endpoint;
use litellm_core::messages::types::{MessagesEndpoint, MessagesOptions, ProviderMessagesRequest};
use litellm_python_interop::{Pythonized, from_py, run_async_value, run_sync_value};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use crate::driver::{ADDITIONAL_ARGS, API_BASE, API_KEY, COMPLETE_INPUT_DICT, HEADERS, INPUT};
use crate::errors::{RustUpstreamError, core_error_to_pyerr, messages_provider_error_to_pyerr};
use crate::marshal::optional_timeout;
use crate::retained::RequestRoots;

#[pyclass]
struct MessagesState {
    roots: Option<RequestRoots>,
    logging: Option<Py<PyAny>>,
    pre_call: Option<Py<PyDict>>,
    pending: Option<(
        MessagesEndpoint,
        litellm_core::lifecycle::PreCallBody<Value>,
    )>,
}

#[pymethods]
impl MessagesState {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(roots) = &self.roots {
            roots.traverse(&visit)?;
        }
        visit.call(&self.logging)?;
        visit.call(&self.pre_call)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let roots = {
            let mut state = slf.borrow_mut();
            (
                state.roots.take(),
                state.logging.take(),
                state.pre_call.take(),
                state.pending.take(),
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

fn decode_options(py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<MessagesOptions> {
    let timeout = optional_timeout(
        arguments
            .get_item(pyo3::intern!(py, "timeout_seconds"))?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<f64>())
            .transpose()?,
    )?;
    Ok(MessagesOptions {
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
    crate::driver::invoke(py, operation, asynchronous, "messages", host)
}

#[pyfunction]
fn build_request(
    py: Python<'_>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
) -> PyResult<Py<MessagesState>> {
    let bag = arguments.bind(py);
    let options = decode_options(py, bag)?;
    let endpoint = py
        .detach(|| build_endpoint(options))
        .map_err(core_error_to_pyerr)?;
    let body = bag
        .get_item("body")?
        .ok_or_else(|| PyValueError::new_err("messages requires body"))?
        .cast_into::<PyDict>()?;
    let callback_body: Value = from_py(body.as_any())?;
    let snapshot = endpoint
        .capture_buffered_body(callback_body)
        .map_err(core_error_to_pyerr)?;
    let headers = PyDict::new(py);
    for (name, value) in endpoint.headers() {
        headers.set_item(name, value)?;
    }
    let additional = PyDict::new(py);
    additional.set_item(COMPLETE_INPUT_DICT, &body)?;
    additional.set_item(API_BASE, endpoint.url())?;
    additional.set_item(HEADERS, &headers)?;
    let kwargs = PyDict::new(py);
    let serialized = py.import("json")?.call_method1("dumps", (&body,))?;
    let message = PyDict::new(py);
    message.set_item("role", "user")?;
    message.set_item("content", serialized)?;
    kwargs.set_item(INPUT, vec![message])?;
    kwargs.set_item(API_KEY, "")?;
    kwargs.set_item(ADDITIONAL_ARGS, additional)?;
    Py::new(
        py,
        MessagesState {
            roots: Some(RequestRoots::new(
                arguments,
                body.unbind().into_any(),
                headers.unbind().into_any(),
            )),
            logging: Some(logging),
            pre_call: Some(kwargs.unbind()),
            pending: Some((endpoint, snapshot)),
        },
    )
}

#[pyfunction]
fn pre_call(py: Python<'_>, state: Py<MessagesState>) -> PyResult<()> {
    let (logging, arguments) = {
        let state = state.borrow(py);
        let logging = state
            .logging
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("messages logging state was cleared"))?
            .clone_ref(py);
        let arguments = state
            .pre_call
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("messages pre-call state was cleared"))?
            .clone_ref(py);
        (logging, arguments)
    };
    logging
        .bind(py)
        .call_method(pyo3::intern!(py, "pre_call"), (), Some(arguments.bind(py)))?;
    Ok(())
}

fn take_request(py: Python<'_>, state: &Py<MessagesState>) -> PyResult<ProviderMessagesRequest> {
    let ((endpoint, pre_call_body), headers) = {
        let mut state = state.borrow_mut(py);
        let pending = state.pending.take().ok_or_else(|| {
            PyRuntimeError::new_err("messages request was already sent or cleared")
        })?;
        let roots = state
            .roots
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("messages roots were cleared"))?;
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
fn send(py: Python<'_>, state: Py<MessagesState>) -> PyResult<Bound<'_, PyAny>> {
    let request = take_request(py, &state)?;
    litellm_python_interop::run_async_py(py, async move {
        let _state = state;
        let response = run_async_value(
            execute_provider_messages_request(request),
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
        execute_provider_messages_request(request),
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

    #[pyfunction]
    fn snapshot(py: Python<'_>, state: Py<MessagesState>) -> PyResult<Py<PyAny>> {
        let request = take_request(py, &state)?;
        let body: Value = serde_json::from_slice(request.body())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        litellm_python_interop::to_py(py, &(body, request.headers()))
    }

    #[test]
    fn callback_aliases_survive_snapshot_and_roots_are_collectible() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "messages_test").unwrap();
            module
                .add_function(wrap_pyfunction!(build_request, &module).unwrap())
                .unwrap();
            module
                .add_function(wrap_pyfunction!(pre_call, &module).unwrap())
                .unwrap();
            module
                .add_function(wrap_pyfunction!(snapshot, &module).unwrap())
                .unwrap();
            let globals = PyDict::new(py);
            globals.set_item("native", module).unwrap();
            py.run(
                c"
import gc
import weakref

class Opaque:
    pass

class Logger:
    def pre_call(self, **kwargs):
        view = kwargs['additional_args']
        self.body = view['complete_input_dict']
        self.headers = view['headers']
        assert self.body is body
        assert self.body['messages'][0] is message
        message['content'] = 'changed'
        self.headers['x-hook'] = 'changed'
        view['complete_input_dict'] = {'replacement': True}
        view['headers'] = {'replacement': 'true'}

message = {'role': 'user', 'content': 'original'}
body = {'model': 'model', 'messages': [message], 'max_tokens': 16}
opaque = Opaque()
logger = Logger()
arguments = dict(model='model', body=body, api_key='test',
                 custom_llm_provider='anthropic', opaque=opaque,
                 litellm_logging_obj=logger)
state = native.build_request(arguments, logger)
native.pre_call(state)
wire_body, wire_headers = native.snapshot(state)
assert wire_body['messages'][0]['content'] == 'original'
assert body['messages'][0]['content'] == 'changed'
assert dict(wire_headers)['x-hook'] == 'changed'
try:
    native.snapshot(state)
except RuntimeError:
    pass
else:
    raise AssertionError('request was sent twice')
arguments['cycle'] = state
logger.body['cycle'] = state
logger.headers['cycle'] = state
alive = weakref.ref(opaque)
del arguments, logger, opaque, body
gc.collect()
assert alive() is not None
del state
gc.collect()
assert alive() is None
",
                Some(&globals),
                Some(&globals),
            )
            .unwrap();
        });
    }

    #[test]
    fn decode_state_rejects_invalid_timeouts_without_panicking() {
        Python::initialize();
        Python::attach(|py| {
            for timeout in [-1.0, 0.0, f64::NAN, f64::INFINITY, f64::MAX] {
                let arguments = PyDict::new(py);
                arguments.set_item("model", "model").unwrap();
                arguments.set_item("body", PyDict::new(py)).unwrap();
                arguments.set_item("timeout_seconds", timeout).unwrap();
                let error = match decode_options(py, &arguments) {
                    Ok(_) => panic!("invalid timeout should fail normally"),
                    Err(error) => error,
                };
                assert!(error.is_instance_of::<PyValueError>(py));
            }
        });
    }
}
