use litellm_core::Error;
use litellm_core::chat_completions::lifecycle::{
    Admission, ChatCompletionsRoute, Observations, Operation, Options, machine,
};
use litellm_core::chat_completions::request::{
    build_pre_call_request_with_services, parse_messages, settle_pre_call_request_with_services,
};
use litellm_core::chat_completions::types::{
    ChatCompletionsRequest, ChatPreCallReadback, ChatPreCallRequest, PreCallHeadersPolicy,
};
use litellm_core::chat_completions::{
    chat_completions_decline_reason, execute_settled_with_terminal,
};
use litellm_core::lifecycle::{
    CallLifecycleContext, ErrorDisposition, ExecutedCall, Lifecycle, Outcome, TerminalRecord,
};
use litellm_python_interop::{
    Pythonized, case_insensitive_headers, from_py, run_async_value, run_sync_value, to_py,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use crate::driver::{ADDITIONAL_ARGS, API_BASE, API_KEY, COMPLETE_INPUT_DICT, HEADERS, INPUT};
use crate::errors::{RustBridgeDeclined, chat_completions_error_to_pyerr, core_error_to_pyerr};
use crate::marshal::optional_timeout;
use crate::retained::RequestRoots;

#[pyclass]
struct ChatCompletionsState {
    roots: Option<RequestRoots>,
    logging: Option<Py<PyAny>>,
    pre_call: Option<Py<PyDict>>,
    pending: Option<ChatPreCallRequest>,
    context: Option<CallLifecycleContext>,
    terminal: Option<TerminalRecord>,
}

#[pymethods]
impl ChatCompletionsState {
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
                state.context.take(),
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
    let messages = parse_messages(value(arguments, "messages")?)
        .map_err(|_| RustBridgeDeclined::new_err("unreadable message list"))?;
    Ok(Admission {
        model: scalar(arguments, "model")?
            .ok_or_else(|| PyValueError::new_err("chat completions requires model"))?,
        messages,
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
    crate::driver::invoke(py, operation, asynchronous, "chat completions", host)
}

#[pyfunction]
fn build_request(
    py: Python<'_>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
) -> PyResult<Py<ChatCompletionsState>> {
    let bag = arguments.bind(py);
    let admission = admission(bag)?;
    let api_key = scalar(bag, "api_key")?;
    let api_base = scalar(bag, "api_base")?;
    let timeout = optional_timeout(
        bag.get_item("timeout_seconds")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<f64>())
            .transpose()?,
    )?;
    let context = CallLifecycleContext::new(
        "chat_completion",
        &admission.model,
        admission.custom_llm_provider.as_deref().unwrap_or_default(),
        scalar(bag, "litellm_call_id")?.unwrap_or_default(),
    );
    let extra_headers = optional_map(bag, "extra_headers")?;
    let built = run_sync_value(
        py,
        async move {
            build_pre_call_request_with_services(
                crate::runtime::authorization_services().as_ref(),
                ChatCompletionsRequest {
                    model: &admission.model,
                    messages: admission.messages,
                    optional_params: admission.optional_params,
                    api_key: api_key.as_deref(),
                    api_base: api_base.as_deref(),
                    custom_llm_provider: admission.custom_llm_provider.as_deref(),
                    extra_headers,
                    timeout,
                },
            )
            .await
        },
        core_error_to_pyerr,
    )?;
    let body = match &built.body {
        litellm_core::lifecycle::PreCallBody::StructuredAtSend {
            callback: generated,
        } => {
            let body = PyDict::new(py);
            for (name, value) in generated {
                body.set_item(name, to_py(py, &value)?)?;
            }
            if let Some(params) = bag.get_item("optional_params")? {
                for name in &built.parameter_fields {
                    body.set_item(name, params.get_item(name)?)?;
                }
            }
            body.into_any()
        }
        litellm_core::lifecycle::PreCallBody::SerializedAtBuild {
            callback: logging_body,
            ..
        } => logging_body.into_pyobject(py)?.into_any(),
        litellm_core::lifecycle::PreCallBody::StructuredAtBuild { .. } => {
            return Err(PyRuntimeError::new_err(
                "chat structured-at-build body is not registered",
            ));
        }
    };
    let headers = PyDict::new(py);
    for (name, value) in &built.headers {
        headers.set_item(name, value)?;
    }
    let headers = match built.headers_policy {
        PreCallHeadersPolicy::PreserveInput => {
            match bag
                .get_item("extra_headers")?
                .filter(|value| !value.is_none())
            {
                Some(original) => {
                    original.call_method1("update", (&headers,))?;
                    original
                }
                None => headers.into_any(),
            }
        }
        PreCallHeadersPolicy::CaseInsensitive => case_insensitive_headers(&headers)?,
    };
    let additional = PyDict::new(py);
    additional.set_item(COMPLETE_INPUT_DICT, &body)?;
    additional.set_item(API_BASE, bag.get_item("api_base")?)?;
    additional.set_item(HEADERS, &headers)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(INPUT, bag.get_item("messages")?)?;
    kwargs.set_item(API_KEY, bag.get_item("logging_api_key")?)?;
    kwargs.set_item(ADDITIONAL_ARGS, additional)?;
    Py::new(
        py,
        ChatCompletionsState {
            roots: Some(RequestRoots::new(
                arguments,
                body.unbind(),
                headers.unbind(),
            )),
            logging: Some(logging),
            pre_call: Some(kwargs.unbind()),
            pending: Some(built),
            context: Some(context),
            terminal: None,
        },
    )
}

#[pyfunction]
fn pre_call(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<()> {
    let (logging, arguments) = {
        let state = state.borrow(py);
        let logging = state
            .logging
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions logging state was cleared"))?
            .clone_ref(py);
        let arguments = state
            .pre_call
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions pre-call state was cleared"))?
            .clone_ref(py);
        (logging, arguments)
    };
    logging
        .bind(py)
        .call_method(pyo3::intern!(py, "pre_call"), (), Some(arguments.bind(py)))?;
    Ok(())
}

struct OwnedRequest {
    request: ChatPreCallRequest,
    readback: ChatPreCallReadback,
    context: CallLifecycleContext,
}

fn take_request(py: Python<'_>, state: &Py<ChatCompletionsState>) -> PyResult<OwnedRequest> {
    let (pending, context, body, headers) = {
        let mut state = state.borrow_mut(py);
        let pending = state.pending.take().ok_or_else(|| {
            PyRuntimeError::new_err("chat completions request was already sent or cleared")
        })?;
        let context = state
            .context
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions context was cleared"))?;
        let roots = state
            .roots
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions roots were cleared"))?;
        (pending, context, roots.body(py), roots.headers(py))
    };
    let headers = headers
        .call_method0("items")?
        .try_iter()?
        .map(|item| item?.extract::<(String, String)>())
        .collect::<PyResult<_>>()?;
    let readback = match &pending.body {
        litellm_core::lifecycle::PreCallBody::StructuredAtSend { .. } => {
            ChatPreCallReadback::StructuredAtSend {
                body: litellm_core::lifecycle::WireBody::encode(
                    &from_py::<Value>(&body)?,
                    "chat completions request",
                )
                .map_err(core_error_to_pyerr)?,
                headers,
            }
        }
        litellm_core::lifecycle::PreCallBody::SerializedAtBuild { .. }
        | litellm_core::lifecycle::PreCallBody::StructuredAtBuild { .. } => {
            ChatPreCallReadback::CapturedAtBuild { headers }
        }
    };
    Ok(OwnedRequest {
        request: pending,
        readback,
        context,
    })
}

async fn execute(
    request: OwnedRequest,
) -> Result<
    ExecutedCall<litellm_core::chat_completions::types::ChatCompletionsResponse, Error>,
    Error,
> {
    let settled = settle_pre_call_request_with_services(
        crate::runtime::authorization_services().as_ref(),
        request.request,
        request.readback,
    )
    .await;
    Ok(execute_settled_with_terminal(settled?, request.context).await)
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
        let executed = run_async_value(execute(request), chat_completions_error_to_pyerr).await?;
        Python::attach(|py| store_result(py, &state, executed))
    })
}

#[pyfunction]
fn send_sync(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<Py<PyAny>> {
    let request = take_request(py, &state)?;
    let executed = run_sync_value(py, execute(request), chat_completions_error_to_pyerr)?;
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
    let Ok(messages) = parse_messages(messages) else {
        return Some("unreadable message list".to_string());
    };
    chat_completions_decline_reason(
        &model,
        custom_llm_provider.as_deref(),
        &messages,
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
    module.add("build_request", wrap_pyfunction!(build_request, &module)?)?;
    module.add("pre_call", wrap_pyfunction!(pre_call, &module)?)?;
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
        let (body, headers) = match &request.readback {
            ChatPreCallReadback::StructuredAtSend { body, headers } => {
                (body.as_bytes(), headers.as_slice())
            }
            ChatPreCallReadback::CapturedAtBuild { headers } => (
                request.request.body.authorized().unwrap().body(),
                headers.as_slice(),
            ),
        };
        to_py(
            py,
            &(serde_json::from_slice::<Value>(body).unwrap(), headers),
        )
    }

    #[test]
    #[ignore = "requires requests on PYTHONPATH"]
    fn bedrock_callback_headers_work_without_botocore() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "chat_headers_test").unwrap();
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
import sys
from collections.abc import MutableMapping
from unittest.mock import patch

class Logger:
    def pre_call(self, **kwargs):
        view = kwargs['additional_args']
        headers = view['headers']
        assert isinstance(headers, MutableMapping)
        assert headers is not original_headers
        assert headers['X-ORIGINAL'] == 'original'
        assert headers.get('CONTENT-TYPE') == 'application/json'
        assert 'AUTHORIZATION' in headers
        assert headers['Authorization'].startswith(expected_auth)
        headers.update({'x-original': 'edited', 'X-Added': 'added'})
        assert headers.setdefault('X-ORIGINAL', 'ignored') == 'edited'
        assert sum(key.lower() == 'x-original' for key in headers) == 1
        assert headers.pop('x-ADDED') == 'added'
        del headers['X-DELETE']
        copied = headers.copy()
        copied['x-original'] = 'copy edit'
        assert headers['X-Original'] == 'edited'
        assert isinstance(view['complete_input_dict'], str)
        view['headers'] = {'replacement': 'ignored'}
        view['complete_input_dict'] = 'replacement'

with patch.dict(sys.modules, {'botocore': None, 'botocore.awsrequest': None}):
    for api_key, expected_auth in [('test-token', 'Bearer test-token'), ('', 'AWS4-HMAC-SHA256 ')]:
        original_headers = {'X-Original': 'original', 'x-delete': 'delete'}
        arguments = dict(
            model='anthropic.claude-opus-5',
            messages=[{'role': 'user', 'content': 'original'}],
            optional_params={'maxTokens': 16, 'aws_region_name': 'us-west-2',
                             'aws_access_key_id': 'test-access-key',
                             'aws_secret_access_key': 'test-secret-key'},
            extra_headers=original_headers, api_key=api_key,
            custom_llm_provider='bedrock', api_base=None,
        )
        state = native.build_request(arguments, Logger())
        native.pre_call(state)
        wire_body, wire_headers = native.snapshot(state)
        assert wire_body['messages'][0]['content'][0]['text'] == 'original'
        headers = {key.lower(): value for key, value in wire_headers}
        assert headers['x-original'] == 'edited'
        assert 'x-delete' not in headers
        assert 'x-added' not in headers
        assert 'replacement' not in headers
        assert headers['authorization'].startswith(expected_auth)
        assert original_headers == {'X-Original': 'original', 'x-delete': 'delete'}
",
                Some(&globals),
                Some(&globals),
            )
            .unwrap();
        });
    }

    #[test]
    #[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
    fn callback_roots_survive_rebinding_and_cycles_are_collected() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "chat_test").unwrap();
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
        assert kwargs['input'] is messages
        assert self.body['messages'] is not messages
        assert self.body['stop_sequences'] is stops
        assert self.headers is headers
        messages[0]['content'] = 'edited'
        self.body['messages'][0]['content'][0]['text'] = 'body edit'
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
                 optional_params={'max_tokens': 16, 'stop_sequences': stops},
                 extra_headers=headers, api_key='test',
                 custom_llm_provider='anthropic', opaque=opaque,
                 litellm_logging_obj=logger)
state = native.build_request(arguments, logger)
native.pre_call(state)
wire_body, wire_headers = native.snapshot(state)
assert wire_body['messages'][0]['content'][0]['text'] == 'body edit'
assert wire_body['stop_sequences'] == ['first', 'second']
assert dict(wire_headers)['x-hook'] == 'edited'
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
