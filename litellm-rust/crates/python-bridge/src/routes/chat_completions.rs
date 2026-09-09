use litellm_core::Error as CoreError;
use litellm_core::chat_completions::chat_completions_decline_reason;
use litellm_core::chat_completions::lifecycle::{
    Admission, ChatCompletionsRoute, Observations, Operation, Options, machine,
};
use litellm_core::chat_completions::request::parse_messages;
use litellm_core::chat_completions::types::{
    ChatCompletionsRequest, ChatPreCallReadback, ChatPreCallRequest, PreCallHeadersPolicy,
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

use crate::arguments::{ChatAdmissionArguments, ChatBuildArguments};
use crate::callbacks::pre_call_args;
use crate::errors::{Error, Route, chat_completions_error_to_pyerr, core_error_to_pyerr};
use crate::marshal::optional_timeout;
use crate::retained::{RequestRoots, RetainedCallback};

struct PendingChatRequest {
    request: ChatPreCallRequest,
    context: CallLifecycleContext,
}

#[pyclass]
struct ChatCompletionsState {
    callback: RetainedCallback,
    pending: Option<PendingChatRequest>,
    terminal: Option<TerminalRecord>,
}

#[pymethods]
impl ChatCompletionsState {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.callback.traverse(&visit)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let retained = {
            let mut state = slf.borrow_mut();
            (
                state.callback.clear(),
                state.pending.take(),
                state.terminal.take(),
            )
        };
        drop(retained);
    }
}

fn admission(arguments: &ChatAdmissionArguments<'_>) -> PyResult<Admission> {
    let messages = parse_messages(from_py(&arguments.messages)?)
        .map_err(|_| PyErr::from(Error::declined("unreadable message list")))?;
    Ok(Admission {
        model: arguments
            .model
            .clone()
            .ok_or_else(|| PyValueError::new_err("chat completions requires model"))?,
        messages,
        optional_params: arguments
            .optional_params
            .as_ref()
            .map(from_py::<Map<String, Value>>)
            .transpose()?
            .unwrap_or_default(),
        custom_llm_provider: arguments.custom_llm_provider.clone(),
    })
}

#[pyclass]
struct ChatCompletionsLifecycle {
    machine: Lifecycle<ChatCompletionsRoute>,
    asynchronous: bool,
    pending_operation: Option<litellm_core::lifecycle::program::OperationTicket>,
}

impl ChatCompletionsLifecycle {
    fn preparation_permit(
        &mut self,
    ) -> PyResult<litellm_core::lifecycle::program::PreparationPermit<ChatCompletionsRoute>> {
        let ticket = self.pending_operation.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no build operation is pending")
        })?;
        self.machine
            .preparation_permit(ticket)
            .map_err(core_error_to_pyerr)
    }

    fn provider_permit(
        &mut self,
    ) -> PyResult<litellm_core::lifecycle::program::ProviderPermit<ChatCompletionsRoute>> {
        let ticket = self.pending_operation.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no send operation is pending")
        })?;
        self.machine
            .provider_permit(ticket)
            .map_err(core_error_to_pyerr)
    }
}

#[pymethods]
impl ChatCompletionsLifecycle {
    #[new]
    fn new(
        arguments: &Bound<'_, PyDict>,
        asynchronous: bool,
        internal_call: bool,
    ) -> PyResult<Self> {
        require_messages(arguments)?;
        let arguments = arguments.extract::<ChatAdmissionArguments<'_>>()?;
        match machine(
            &admission(&arguments)?,
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
                pending_operation: None,
            }),
            Err(decline) => Err(Error::declined(decline.reason()).into()),
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
}

#[pyfunction]
fn invoke(
    py: Python<'_>,
    machine: Py<ChatCompletionsLifecycle>,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let (operation, asynchronous) = {
        let mut machine = machine.borrow_mut(py);
        let ticket = machine.machine.issue().map_err(core_error_to_pyerr)?;
        let operation = ticket.operation();
        machine.pending_operation = Some(ticket);
        (operation, machine.asynchronous)
    };
    crate::driver::invoke(py, operation, asynchronous, Route::ChatCompletions, host)
}

#[pyfunction]
fn build_request(
    py: Python<'_>,
    machine: Py<ChatCompletionsLifecycle>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
) -> PyResult<Py<ChatCompletionsState>> {
    let permit = machine.borrow_mut(py).preparation_permit()?;
    let bag = arguments.bind(py);
    require_messages(bag)?;
    let admission_arguments = bag.extract::<ChatAdmissionArguments<'_>>()?;
    let build_arguments = bag.extract::<ChatBuildArguments<'_>>()?;
    let admission = admission(&admission_arguments)?;
    let api_key = build_arguments.api_key.clone();
    let api_base = build_arguments
        .api_base
        .as_ref()
        .map(|value| value.extract::<String>())
        .transpose()?;
    let timeout = optional_timeout(build_arguments.timeout_seconds)?;
    let context = CallLifecycleContext::new(
        "chat_completion",
        &admission.model,
        admission.custom_llm_provider.as_deref().unwrap_or_default(),
        build_arguments.litellm_call_id.clone().unwrap_or_default(),
    );
    let extra_headers = build_arguments
        .extra_headers
        .as_ref()
        .map(from_py::<Map<String, Value>>)
        .transpose()?;
    let built = run_sync_value(
        py,
        async move {
            permit
                .chat_completions(
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
            if let Some(params) = &admission_arguments.optional_params {
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
        PreCallHeadersPolicy::PreserveInput => match &build_arguments.extra_headers {
            Some(original) => {
                original.call_method1(pyo3::intern!(py, "update"), (&headers,))?;
                original.clone()
            }
            None => headers.into_any(),
        },
        PreCallHeadersPolicy::CaseInsensitive => case_insensitive_headers(&headers)?,
    };
    let kwargs = pre_call_args(
        &admission_arguments.messages,
        build_arguments.logging_api_key.as_ref(),
        &body,
        build_arguments.api_base.as_ref(),
        &headers,
    )
    .into_pyobject(py)?;
    Py::new(
        py,
        ChatCompletionsState {
            callback: RetainedCallback::new(
                RequestRoots::new(arguments, body.unbind(), headers.unbind()),
                logging,
                kwargs.unbind(),
            ),
            pending: Some(PendingChatRequest {
                request: built,
                context,
            }),
            terminal: None,
        },
    )
}

#[pyfunction]
fn pre_call(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<()> {
    let (logging, arguments) = {
        let state = state.borrow(py);
        (
            state.callback.logging(py, Route::ChatCompletions)?,
            state.callback.pre_call(py, Route::ChatCompletions)?,
        )
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
    let (pending, body, headers) = {
        let mut state = state.borrow_mut(py);
        let pending = state
            .pending
            .take()
            .ok_or(Error::RequestConsumed(Route::ChatCompletions))?;
        let roots = state.callback.roots(Route::ChatCompletions)?;
        (pending, roots.body(py), roots.headers(py))
    };
    let headers = headers
        .call_method0("items")?
        .try_iter()?
        .map(|item| item?.extract::<(String, String)>())
        .collect::<PyResult<_>>()?;
    let readback = match &pending.request.body {
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
        request: pending.request,
        readback,
        context: pending.context,
    })
}

async fn execute(
    permit: litellm_core::lifecycle::program::ProviderPermit<ChatCompletionsRoute>,
    request: OwnedRequest,
) -> Result<
    ExecutedCall<litellm_core::chat_completions::types::ChatCompletionsResponse, CoreError>,
    CoreError,
> {
    Ok(permit
        .chat_completions(
            crate::runtime::authorization_services().as_ref(),
            request.request,
            request.readback,
            request.context,
        )
        .await)
}

fn store_result(
    py: Python<'_>,
    state: &Py<ChatCompletionsState>,
    executed: ExecutedCall<
        litellm_core::chat_completions::types::ChatCompletionsResponse,
        CoreError,
    >,
) -> PyResult<Py<PyAny>> {
    state.borrow_mut(py).terminal = Some(executed.terminal().clone());
    match executed {
        ExecutedCall::Success { response, .. } | ExecutedCall::Deferred { response, .. } => {
            Ok(Pythonized(response).into_pyobject(py)?.unbind().into_any())
        }
        ExecutedCall::Failure { error, .. } => Err(chat_completions_error_to_pyerr(error)),
    }
}

#[pyfunction]
fn send(
    py: Python<'_>,
    machine: Py<ChatCompletionsLifecycle>,
    state: Py<ChatCompletionsState>,
) -> PyResult<Bound<'_, PyAny>> {
    let permit = machine.borrow_mut(py).provider_permit()?;
    let request = take_request(py, &state)?;
    litellm_python_interop::run_async_py(py, async move {
        let executed =
            run_async_value(execute(permit, request), chat_completions_error_to_pyerr).await?;
        Python::attach(|py| store_result(py, &state, executed))
    })
}

#[pyfunction]
fn send_sync(
    py: Python<'_>,
    machine: Py<ChatCompletionsLifecycle>,
    state: Py<ChatCompletionsState>,
) -> PyResult<Py<PyAny>> {
    let permit = machine.borrow_mut(py).provider_permit()?;
    let request = take_request(py, &state)?;
    let executed = run_sync_value(
        py,
        execute(permit, request),
        chat_completions_error_to_pyerr,
    )?;
    store_result(py, &state, executed)
}

#[pyfunction]
fn terminal_record(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<Py<PyAny>> {
    let terminal = state
        .borrow(py)
        .terminal
        .clone()
        .ok_or(Error::TerminalUnavailable(Route::ChatCompletions))?;
    to_py(py, &terminal)
}

fn validate_arguments(arguments: &Bound<'_, PyDict>) -> PyResult<()> {
    let messages = require_messages(arguments)?;
    if !messages.is_instance_of::<pyo3::types::PyList>() {
        return Err(PyTypeError::new_err("messages must be a list"));
    }
    Ok(())
}

fn require_messages<'py>(arguments: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
    arguments
        .get_item("messages")?
        .ok_or_else(|| PyValueError::new_err("chat completions requires messages"))
}

#[pyfunction]
fn chat_completions(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    crate::driver::drive_sync(py, runner(py)?, arguments, bindings(py)?)
}

#[pyfunction]
fn achat_completions(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    validate_arguments(arguments.bind(py))?;
    crate::driver::drive_async(py, runner(py)?, arguments, bindings(py)?)
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
#[path = "../../tests/unit/routes/chat_completions.rs"]
mod tests;
