use litellm_core::Error;
use litellm_core::chat_completions::lifecycle::{
    Admission, ChatCompletionsRoute, Observations, Operation, Options, machine,
};
use litellm_core::chat_completions::types::ChatCompletionsRequest;
use litellm_core::chat_completions::{
    chat_completions_decline_reason, chat_completions_with_terminal,
};
use litellm_core::lifecycle::{
    CallLifecycleContext, ErrorDisposition, ExecutedCall, Lifecycle, Outcome, TerminalRecord,
};
use litellm_python_interop::{Pythonized, from_py, run_async_value, run_sync_value, to_py};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use crate::driver::{ADDITIONAL_ARGS, API_BASE, API_KEY, COMPLETE_INPUT_DICT, HEADERS, INPUT};
use crate::errors::{RustBridgeDeclined, chat_completions_error_to_pyerr, core_error_to_pyerr};
use crate::marshal::optional_timeout;

#[pyclass]
struct ChatCompletionsState {
    arguments: Option<Py<PyDict>>,
    model: Option<String>,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyAny>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    timeout: Option<std::time::Duration>,
    terminal: Option<TerminalRecord>,
}

#[pymethods]
impl ChatCompletionsState {
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
                state.terminal.take(),
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

fn value(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Value> {
    arguments
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("chat completions requires {name}")))
        .and_then(|value| from_py(&value))
}

fn optional_map(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<Map<String, Value>>> {
    arguments
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| from_py(&value))
        .transpose()
}

fn admission(arguments: &Bound<'_, PyDict>) -> PyResult<Admission> {
    Ok(Admission {
        model: scalar(arguments, "model")?
            .ok_or_else(|| PyValueError::new_err("chat completions requires model"))?,
        messages: value(arguments, "messages")?,
        optional_params: optional_map(arguments, "optional_params")?.unwrap_or_default(),
        custom_llm_provider: scalar(arguments, "custom_llm_provider")?,
    })
}

#[pyclass]
struct ChatCompletionsLifecycle {
    machine: Lifecycle<ChatCompletionsRoute>,
    asynchronous: bool,
}

#[pymethods]
impl ChatCompletionsLifecycle {
    #[new]
    fn new(
        arguments: &Bound<'_, PyDict>,
        asynchronous: bool,
        internal_call: bool,
    ) -> PyResult<Self> {
        match machine(
            &admission(arguments)?,
            Options {
                asynchronous,
                internal_call,
            },
        )
        .map_err(core_error_to_pyerr)?
        {
            Ok(machine) => Ok(Self {
                machine,
                asynchronous,
            }),
            Err(decline) => Err(RustBridgeDeclined::new_err(decline.reason())),
        }
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
}

#[pyfunction]
fn invoke(
    py: Python<'_>,
    machine: Py<ChatCompletionsLifecycle>,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let (operation, asynchronous) = {
        let machine = machine.borrow(py);
        (machine.machine.operation(), machine.asynchronous)
    };
    crate::driver::invoke(py, operation, asynchronous, false, "chat completions", host)
}

#[pyfunction]
fn prepare(
    py: Python<'_>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
) -> PyResult<Py<ChatCompletionsState>> {
    let bag = arguments.bind(py);
    let admission = admission(bag)?;
    let api_key = scalar(bag, "api_key")?;
    let api_base = scalar(bag, "api_base")?;
    let headers = bag
        .get_item("extra_headers")?
        .filter(|value| !value.is_none());
    let timeout = optional_timeout(
        bag.get_item("timeout_seconds")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<f64>())
            .transpose()?,
    )?;
    let complete_input = PyDict::new(py);
    complete_input.set_item("model", &admission.model)?;
    complete_input.set_item("messages", bag.get_item("messages")?)?;
    if let Some(optional_params) = bag.get_item("optional_params")? {
        complete_input.call_method1("update", (optional_params,))?;
    }
    let additional = PyDict::new(py);
    additional.set_item(COMPLETE_INPUT_DICT, &complete_input)?;
    additional.set_item(API_BASE, bag.get_item("api_base")?)?;
    additional.set_item(HEADERS, &headers)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(INPUT, bag.get_item("messages")?)?;
    kwargs.set_item(API_KEY, bag.get_item("logging_api_key")?)?;
    kwargs.set_item(ADDITIONAL_ARGS, additional)?;
    logging
        .bind(py)
        .call_method("pre_call", (), Some(&kwargs))?;
    Py::new(
        py,
        ChatCompletionsState {
            arguments: Some(arguments),
            model: Some(admission.model),
            body: Some(complete_input.unbind()),
            headers: headers.map(Bound::unbind),
            api_key,
            api_base,
            custom_llm_provider: admission.custom_llm_provider,
            timeout,
            terminal: None,
        },
    )
}

struct OwnedRequest {
    model: String,
    messages: Value,
    optional_params: Map<String, Value>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    timeout: Option<std::time::Duration>,
    call_id: String,
}

fn take_request(py: Python<'_>, state: &Py<ChatCompletionsState>) -> PyResult<OwnedRequest> {
    let (arguments, body, headers) = {
        let state = state.borrow(py);
        let arguments = state
            .arguments
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions state was cleared"))?
            .clone_ref(py);
        let body = state
            .body
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions body was cleared"))?
            .clone_ref(py);
        let headers = state.headers.as_ref().map(|value| value.clone_ref(py));
        (arguments, body, headers)
    };
    let call_id = scalar(arguments.bind(py), "litellm_call_id")?.unwrap_or_default();
    let messages = value(body.bind(py), "messages")?;
    let optional_params = body
        .bind(py)
        .iter()
        .filter_map(|(key, value)| match key.extract::<String>() {
            Ok(key) if key == "model" || key == "messages" => None,
            Ok(key) => Some(from_py(&value).map(|value| (key, value))),
            Err(error) => Some(Err(error)),
        })
        .collect::<PyResult<Map<String, Value>>>()?;
    let extra_headers = headers
        .as_ref()
        .map(|value| from_py(value.bind(py)))
        .transpose()?;
    let mut state = state.borrow_mut(py);
    Ok(OwnedRequest {
        model: state
            .model
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions request was already sent"))?,
        messages,
        optional_params,
        api_key: state.api_key.take(),
        api_base: state.api_base.take(),
        custom_llm_provider: state.custom_llm_provider.take(),
        extra_headers,
        timeout: state.timeout.take(),
        call_id,
    })
}

async fn execute(
    request: OwnedRequest,
) -> ExecutedCall<litellm_core::chat_completions::types::ChatCompletionsResponse, Error> {
    let provider = request.custom_llm_provider.clone().unwrap_or_default();
    let context =
        CallLifecycleContext::new("chat_completion", &request.model, provider, request.call_id);
    chat_completions_with_terminal(
        ChatCompletionsRequest {
            model: &request.model,
            messages: request.messages,
            optional_params: request.optional_params,
            api_key: request.api_key.as_deref(),
            api_base: request.api_base.as_deref(),
            custom_llm_provider: request.custom_llm_provider.as_deref(),
            extra_headers: request.extra_headers,
            timeout: request.timeout,
        },
        context,
    )
    .await
}

fn store_result(
    py: Python<'_>,
    state: &Py<ChatCompletionsState>,
    executed: ExecutedCall<litellm_core::chat_completions::types::ChatCompletionsResponse, Error>,
) -> PyResult<Py<PyAny>> {
    state.borrow_mut(py).terminal = Some(executed.terminal().clone());
    match executed {
        ExecutedCall::Success { response, .. } => {
            Ok(Pythonized(response).into_pyobject(py)?.unbind().into_any())
        }
        ExecutedCall::Failure { error, .. } => Err(chat_completions_error_to_pyerr(error)),
    }
}

#[pyfunction]
fn send(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<Bound<'_, PyAny>> {
    let request = take_request(py, &state)?;
    litellm_python_interop::run_async_py(py, async move {
        let executed = run_async_value(
            async move { Ok::<_, std::convert::Infallible>(execute(request).await) },
            |never| match never {},
        )
        .await?;
        Python::attach(|py| store_result(py, &state, executed))
    })
}

#[pyfunction]
fn send_sync(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<Py<PyAny>> {
    let request = take_request(py, &state)?;
    let executed = run_sync_value(
        py,
        async move { Ok::<_, std::convert::Infallible>(execute(request).await) },
        |never| match never {},
    )?;
    store_result(py, &state, executed)
}

#[pyfunction]
fn terminal_record(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<Py<PyAny>> {
    let terminal = state.borrow(py).terminal.clone().ok_or_else(|| {
        PyRuntimeError::new_err("chat completions terminal record is unavailable")
    })?;
    to_py(py, &terminal)
}

fn validate_arguments(arguments: &Bound<'_, PyDict>) -> PyResult<()> {
    let messages = arguments
        .get_item("messages")?
        .ok_or_else(|| PyValueError::new_err("chat completions requires messages"))?;
    if !messages.is_instance_of::<pyo3::types::PyList>() {
        return Err(PyTypeError::new_err("messages must be a list"));
    }
    Ok(())
}

#[pyfunction]
fn chat_completions(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    runner(py)?
        .getattr("_drive_sync")?
        .call1((arguments, bindings(py)?))
}

#[pyfunction]
fn achat_completions(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    runner(py)?
        .getattr("_drive_async")?
        .call1((arguments, bindings(py)?))
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, custom_llm_provider=None))]
fn chat_completions_decline(
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] messages: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<
        Map<String, Value>,
    >,
    custom_llm_provider: Option<String>,
) -> Option<String> {
    chat_completions_decline_reason(
        &model,
        custom_llm_provider.as_deref(),
        messages,
        &optional_params.unwrap_or_default(),
    )
    .map(str::to_string)
}

fn runner(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static RUNNER: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = RUNNER.get(py) {
        return Ok(module.bind(py));
    }
    let module = py.import("litellm.rust_bridge.chat_completions")?;
    Ok(RUNNER.get_or_init(py, || module.unbind()).bind(py))
}

fn bindings(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static BINDINGS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = BINDINGS.get(py) {
        return Ok(module.bind(py));
    }
    let module = PyModule::new(py, "_chat_completions_bindings")?;
    module.add("Lifecycle", py.get_type::<ChatCompletionsLifecycle>())?;
    module.add("invoke", wrap_pyfunction!(invoke, &module)?)?;
    module.add("prepare", wrap_pyfunction!(prepare, &module)?)?;
    module.add("send", wrap_pyfunction!(send, &module)?)?;
    module.add("send_sync", wrap_pyfunction!(send_sync, &module)?)?;
    module.add(
        "terminal_record",
        wrap_pyfunction!(terminal_record, &module)?,
    )?;
    Ok(BINDINGS.get_or_init(py, || module.unbind()).bind(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, wrap_pyfunction!(chat_completions, module)?)?;
    crate::routes::definition::add_function(module, wrap_pyfunction!(achat_completions, module)?)?;
    crate::routes::definition::add_function(
        module,
        wrap_pyfunction!(chat_completions_decline, module)?,
    )?;
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
    fn snapshot(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<Py<PyAny>> {
        let request = take_request(py, &state)?;
        to_py(
            py,
            &(
                request.messages,
                request.optional_params,
                request.extra_headers,
            ),
        )
    }

    #[test]
    #[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
    fn callback_roots_survive_rebinding_and_cycles_are_collected() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "chat_test").unwrap();
            module
                .add_function(wrap_pyfunction!(prepare, &module).unwrap())
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
        assert kwargs['input'] is messages
        assert self.body['messages'] is messages
        assert self.body['stop'] is stops
        assert self.headers is headers
        messages[0]['content'] = 'edited'
        stops.append('second')
        self.headers['x-hook'] = 'edited'
        view['complete_input_dict'] = {'replacement': True}
        view['headers'] = {'replacement': 'true'}

messages = [{'role': 'user', 'content': 'original'}]
stops = ['first']
headers = {}
opaque = Opaque()
logger = Logger()
arguments = dict(model='claude-opus-5', messages=messages,
                 optional_params={'max_tokens': 16, 'stop': stops},
                 extra_headers=headers, api_key='test',
                 custom_llm_provider='anthropic', opaque=opaque,
                 litellm_logging_obj=logger)
state = native.prepare(arguments, logger)
wire_messages, wire_params, wire_headers = native.snapshot(state)
assert wire_messages[0]['content'] == 'edited'
assert wire_params['stop'] == ['first', 'second']
assert wire_headers == {'x-hook': 'edited'}
arguments['cycle'] = state
logger.body['cycle'] = state
headers['cycle'] = state
alive = weakref.ref(opaque)
del arguments, logger, opaque, headers
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
}
