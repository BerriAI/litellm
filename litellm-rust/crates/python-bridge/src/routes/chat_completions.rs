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

use crate::errors::{RustBridgeDeclined, chat_completions_error_to_pyerr, core_error_to_pyerr};
use crate::marshal::optional_timeout;

#[pyclass]
struct ChatCompletionsState {
    arguments: Option<Py<PyDict>>,
    model: Option<String>,
    messages: Option<Value>,
    optional_params: Option<Map<String, Value>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    timeout: Option<std::time::Duration>,
    terminal: Option<TerminalRecord>,
}

#[pymethods]
impl ChatCompletionsState {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.arguments)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let roots = {
            let mut state = slf.borrow_mut();
            (state.arguments.take(), state.terminal.take())
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
    let (method, awaiting) = match operation {
        Operation::Setup => ("setup", false),
        Operation::DeploymentPre => ("deployment_pre", true),
        Operation::Prepare => ("prepare", false),
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
            return Err(PyRuntimeError::new_err(
                "chat completions lifecycle is complete",
            ));
        }
    };
    Ok((awaiting, host.getattr(py, method)?.call0(py)?))
}

#[pyfunction]
fn prepare(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Py<ChatCompletionsState>> {
    let bag = arguments.bind(py);
    let admission = admission(bag)?;
    let api_key = scalar(bag, "api_key")?;
    let api_base = scalar(bag, "api_base")?;
    let extra_headers = optional_map(bag, "extra_headers")?;
    let timeout = optional_timeout(
        bag.get_item("timeout_seconds")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<f64>())
            .transpose()?,
    )?;
    let logging = bag
        .get_item("litellm_logging_obj")?
        .filter(|value| !value.is_none())
        .ok_or_else(|| PyRuntimeError::new_err("chat completions logging was not initialized"))?;
    let complete_input = PyDict::new(py);
    complete_input.set_item("model", &admission.model)?;
    complete_input.set_item("messages", bag.get_item("messages")?)?;
    for (name, value) in &admission.optional_params {
        complete_input.set_item(name, Pythonized(value))?;
    }
    let additional = PyDict::new(py);
    additional.set_item("complete_input_dict", complete_input)?;
    additional.set_item("api_base", bag.get_item("api_base")?)?;
    additional.set_item("headers", bag.get_item("extra_headers")?)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("input", bag.get_item("messages")?)?;
    kwargs.set_item("api_key", bag.get_item("logging_api_key")?)?;
    kwargs.set_item("additional_args", additional)?;
    logging.call_method("pre_call", (), Some(&kwargs))?;
    Py::new(
        py,
        ChatCompletionsState {
            arguments: Some(arguments),
            model: Some(admission.model),
            messages: Some(admission.messages),
            optional_params: Some(admission.optional_params),
            api_key,
            api_base,
            custom_llm_provider: admission.custom_llm_provider,
            extra_headers,
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
    let mut state = state.borrow_mut(py);
    let call_id = state
        .arguments
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("chat completions state was cleared"))
        .and_then(|arguments| scalar(arguments.bind(py), "litellm_call_id"))?
        .unwrap_or_default();
    Ok(OwnedRequest {
        model: state
            .model
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("chat completions request was already sent"))?,
        messages: state.messages.take().unwrap(),
        optional_params: state.optional_params.take().unwrap(),
        api_key: state.api_key.take(),
        api_base: state.api_base.take(),
        custom_llm_provider: state.custom_llm_provider.take(),
        extra_headers: state.extra_headers.take(),
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
    driver(py)?.getattr("drive_sync")?.call1((arguments,))
}

#[pyfunction]
fn achat_completions(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    driver(py)?.getattr("drive_async")?.call1((arguments,))
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

fn driver(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static DRIVER: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = DRIVER.get(py) {
        return Ok(module.bind(py));
    }
    let module = crate::driver::compile(py, "chat_completions", HOST)?;
    module.add("_Lifecycle", py.get_type::<ChatCompletionsLifecycle>())?;
    module.add("_invoke", wrap_pyfunction!(invoke, &module)?)?;
    module.add("_prepare", wrap_pyfunction!(prepare, &module)?)?;
    module.add("_send", wrap_pyfunction!(send, &module)?)?;
    module.add("_send_sync", wrap_pyfunction!(send_sync, &module)?)?;
    module.add(
        "_terminal_record",
        wrap_pyfunction!(terminal_record, &module)?,
    )?;
    Ok(DRIVER.get_or_init(py, || module.unbind()).bind(py))
}

const HOST: &str = r#"
from datetime import datetime
from litellm import utils
from litellm.types.utils import CallTypes
from litellm.rust_bridge.chat_completions import build_model_response, initialize_logging, invoke_terminal

class Host:
    def __init__(self, arguments, asynchronous):
        self.machine = _Lifecycle(arguments, asynchronous, utils.is_internal_call.get())
        self.arguments = arguments
        self.current = arguments
        self.asynchronous = asynchronous
        self.logger = arguments.get('litellm_logging_obj')
        self.state = None
        self.response = None
        self.error = None
        self.start = datetime.now()
        self.end = None

    def setup(self):
        self.logger = initialize_logging(self.arguments, self.asynchronous)
        self.arguments['litellm_logging_obj'] = self.logger

    async def deployment_pre(self):
        modified = await utils.async_pre_call_deployment_hook(self.current, 'acompletion')
        if modified is not None:
            self.current = modified
        self.current['litellm_logging_obj'] = self.logger

    def prepare(self): self.state = _prepare(self.current)

    def send_sync(self):
        self.response = build_model_response(_send_sync(self.state), self.arguments['model_response'])
        self.end = datetime.now()

    async def send(self):
        self.response = build_model_response(await _send(self.state), self.arguments['model_response'])
        self.end = datetime.now()

    async def deployment_success(self):
        self.response = await utils.async_post_call_success_deployment_hook(self.current, self.response, CallTypes.acompletion)

    async def deployment_failure(self):
        await utils.async_post_call_failure_deployment_hook(self.current, self.error, 'acompletion')

    def terminal(self, action, value):
        record = _terminal_record(self.state) if self.state is not None else None
        return invoke_terminal(action, (self.arguments, self.current, self.state), self.logger, record, value, self.start, self.end)

    def sync_success(self): return self.terminal('sync_success', self.response)
    def async_success(self): return self.terminal('async_success', self.response)
    def sync_success_if_needed(self): return self.terminal('sync_success_if_needed', self.response)
    def sync_failure(self): return self.terminal('sync_failure', self.error)
    def async_failure(self): return self.terminal('async_failure', self.error)
    def restore(self): utils._restore_correlation_context_if_supported(self.logger)

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
        raise self.error
"#;

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
