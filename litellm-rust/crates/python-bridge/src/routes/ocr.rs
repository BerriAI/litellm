//! Native OCR retains the entire Python argument graph through completion.
//! Missing native operations raise NotImplementedError before
//! callbacks; no Python preparation, auth, encoding, or provider transforms run.

use litellm_core::error::Error as CoreError;
use litellm_core::lifecycle::ocr::{NativeOutcome, Observations, OcrRoute, Operation, Options};
use litellm_core::lifecycle::{
    CallLifecycleContext, ErrorDisposition, ExecutedCall, Lifecycle, Outcome, TerminalRecord,
};
use litellm_core::ocr::types::{
    OcrAdmissionRequest, OcrDocumentProjection, OcrEndpoint, OcrPreCallRequest, SettledOcrRequest,
};
use litellm_core::routing_utils::provider::get_custom_llm_provider;
use litellm_python_interop::{Pythonized, from_py, to_py};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use crate::callbacks::pre_call_args;
use crate::errors::{Error, Route, core_error_to_pyerr, ocr_error_to_pyerr};
use crate::retained::{RequestRoots, RetainedCallback};
use litellm_auth_python::PythonAuth;
use litellm_python_interop::{run_async_value, run_sync_value};

#[pyclass]
struct OcrState {
    callback: RetainedCallback,
    endpoint: Option<OcrEndpoint>,
    asynchronous: bool,
    terminal: Option<TerminalRecord>,
}

#[pymethods]
impl OcrState {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.callback.traverse(&visit)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let retained = {
            let mut state = slf.borrow_mut();
            (
                state.callback.clear(),
                state.endpoint.take(),
                state.terminal.take(),
            )
        };
        drop(retained);
    }
}

fn scalar(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    optional_string(arguments, name).map(|value| value.filter(|value| !value.trim().is_empty()))
}

fn optional_string(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    arguments
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()
}

fn header_pairs(headers: &Bound<'_, PyDict>) -> PyResult<Vec<(String, String)>> {
    headers
        .iter()
        .map(|(name, value)| Ok((name.extract()?, value.extract()?)))
        .collect()
}

fn decode_request(
    py: Python<'_>,
    bag: &Bound<'_, PyDict>,
    credentials: litellm_auth::CredentialInputs,
) -> PyResult<OcrAdmissionRequest> {
    let document = bag
        .get_item("document")?
        .ok_or_else(|| PyValueError::new_err("OCR requires document"))?
        .cast_into::<PyDict>()?;
    let timeout_seconds = match bag.get_item("timeout")?.filter(|value| !value.is_none()) {
        None => py
            .import("litellm.constants")?
            .getattr("request_timeout")?
            .extract::<f64>()?,
        Some(timeout) => match timeout.extract::<f64>() {
            Ok(seconds) => seconds,
            Err(_) => timeout.getattr("read")?.extract::<f64>()?,
        },
    };
    let extra_headers = match bag
        .get_item("extra_headers")?
        .filter(|value| !value.is_none())
    {
        Some(headers) => header_pairs(headers.cast::<PyDict>()?)?,
        None => Vec::new(),
    };
    let model = scalar(bag, "model")?.ok_or_else(|| PyValueError::new_err("OCR requires model"))?;
    let custom_llm_provider = scalar(bag, "custom_llm_provider")?;
    let document_input = PyDict::new(py);
    for name in ["type", "document_url", "image_url"] {
        if let Some(value) = document.get_item(name)? {
            document_input.set_item(name, value)?;
        }
    }
    Ok(OcrAdmissionRequest {
        model,
        custom_llm_provider,
        api_key: optional_string(bag, "api_key")?,
        api_base: optional_string(bag, "api_base")?,
        extra_headers,
        timeout_seconds,
        request_format: scalar(bag, "req_format")?,
        document: from_py(document_input.as_any())?,
        credentials,
        vertex_project: scalar(bag, "vertex_project")?.or(scalar(bag, "vertex_ai_project")?),
        vertex_location: scalar(bag, "vertex_location")?.or(scalar(bag, "vertex_ai_location")?),
        stream: bag
            .get_item("stream")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
    })
}

fn request_error_to_pyerr(
    py: Python<'_>,
    error: CoreError,
    model: &str,
    custom_llm_provider: Option<&str>,
) -> PyErr {
    let resolved = get_custom_llm_provider(model, custom_llm_provider);
    ocr_error_to_pyerr(
        py,
        error,
        resolved.as_ref().map_or(model, |value| value.model),
        resolved
            .as_ref()
            .map_or("", |value| value.custom_llm_provider),
    )
}

#[pyclass]
struct OcrLifecycle {
    machine: Lifecycle<OcrRoute>,
    asynchronous: bool,
    pending_operation: Option<litellm_core::lifecycle::program::OperationTicket>,
}

impl OcrLifecycle {
    fn preparation_permit(
        &mut self,
    ) -> PyResult<litellm_core::lifecycle::program::PreparationPermit<OcrRoute>> {
        let ticket = self.pending_operation.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no build operation is pending")
        })?;
        self.machine
            .preparation_permit(ticket)
            .map_err(core_error_to_pyerr)
    }

    fn provider_permit(
        &mut self,
    ) -> PyResult<litellm_core::lifecycle::program::ProviderPermit<OcrRoute>> {
        let ticket = self.pending_operation.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no send operation is pending")
        })?;
        self.machine
            .provider_permit(ticket)
            .map_err(core_error_to_pyerr)
    }
}

#[pymethods]
impl OcrLifecycle {
    #[new]
    fn new(
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        logger: Option<&Bound<'_, PyAny>>,
        asynchronous: bool,
        internal_call: bool,
    ) -> PyResult<Self> {
        let auth = PythonAuth::from_arguments(py, arguments)?;
        let request = decode_request(py, arguments, auth.inputs().clone())?;
        let identity = |name: &str| -> PyResult<Option<String>> {
            if let Some(logger) = logger {
                match logger.getattr(name) {
                    Ok(value) => {
                        if let Ok(value) = value.extract::<String>() {
                            return Ok(Some(value));
                        }
                    }
                    Err(error)
                        if !error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) =>
                    {
                        return Err(error);
                    }
                    _ => {}
                }
            }
            scalar(arguments, name)
        };
        let machine = Lifecycle::new(
            &request,
            Options {
                asynchronous,
                internal_call,
                call_id: identity("litellm_call_id")?,
                trace_id: identity("litellm_trace_id")?,
                ..Options::default()
            },
        )
        .map_err(|error| {
            request_error_to_pyerr(
                py,
                error,
                &request.model,
                request.custom_llm_provider.as_deref(),
            )
        })?;
        match machine {
            NativeOutcome::Completed(machine) => Ok(Self {
                machine,
                asynchronous,
                pending_operation: None,
            }),
            NativeOutcome::Declined(decline) => {
                Err(PyNotImplementedError::new_err(decline.reason()))
            }
        }
    }

    fn identity(&self) -> (String, Option<String>) {
        let identity = self.machine.identity();
        (identity.call_id.clone(), identity.trace_id.clone())
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
    machine: Py<OcrLifecycle>,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let (operation, asynchronous) = {
        let mut machine = machine.borrow_mut(py);
        let ticket = machine.machine.issue().map_err(core_error_to_pyerr)?;
        let operation = ticket.operation();
        machine.pending_operation = Some(ticket);
        (operation, machine.asynchronous)
    };
    crate::driver::invoke(py, operation, asynchronous, Route::Ocr, host)
}

#[pyfunction]
fn build_request(
    py: Python<'_>,
    machine: Py<OcrLifecycle>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
    asynchronous: bool,
) -> PyResult<Py<OcrState>> {
    let permit = machine.borrow_mut(py).preparation_permit()?;
    let bag = arguments.bind(py);
    let auth = PythonAuth::from_arguments(py, bag)?;
    let request = decode_request(py, bag, auth.inputs().clone())?;
    let model = request.model.clone();
    let custom_llm_provider = request.custom_llm_provider.clone();
    let pre_call_request = py.detach(|| permit.ocr(request, &auth)).map_err(|error| {
        crate::errors::ocr_preparation_error_to_pyerr(
            py,
            error,
            auth.take_error(),
            &model,
            custom_llm_provider.as_deref(),
            bag,
        )
    })?;
    let document = bag
        .get_item("document")?
        .ok_or_else(|| PyValueError::new_err("OCR requires document"))?
        .cast_into::<PyDict>()?;
    let OcrPreCallRequest {
        endpoint,
        headers: draft_headers,
        body: draft_body,
        document_projection,
        parameter_fields,
        ..
    } = pre_call_request;
    let litellm_core::lifecycle::PreCallBody::StructuredAtSend {
        callback: draft_body,
    } = draft_body
    else {
        return Err(PyRuntimeError::new_err(
            "OCR requires a structured-at-send body",
        ));
    };
    let body = PyDict::new(py);
    for (name, value) in &draft_body {
        match (name.as_str(), document_projection) {
            ("document", OcrDocumentProjection::RetainedDocument) => {
                body.set_item(name, &document)?;
            }
            ("document", OcrDocumentProjection::ShallowCopyDocument) => {
                body.set_item(name, document.copy()?)?;
            }
            _ => body.set_item(name, to_py(py, value)?)?,
        }
    }
    let optional_params = PyDict::new(py);
    for &name in parameter_fields {
        if let Some(value) = bag.get_item(name)? {
            body.set_item(name, &value)?;
            optional_params.set_item(name, value)?;
        }
    }
    let headers = PyDict::new(py);
    for (name, value) in &draft_headers {
        headers.set_item(name, value)?;
    }
    let litellm_params = PyDict::new(py);
    litellm_params.set_item("litellm_call_id", bag.get_item("litellm_call_id")?)?;
    litellm_params.set_item(
        pyo3::intern!(py, "api_base"),
        bag.get_item(pyo3::intern!(py, "api_base"))?,
    )?;
    let update = PyDict::new(py);
    update.set_item("kwargs", bag)?;
    update.set_item("model", endpoint.model())?;
    update.set_item("optional_params", optional_params)?;
    update.set_item("litellm_params", litellm_params)?;
    update.set_item("custom_llm_provider", endpoint.custom_llm_provider())?;
    logging
        .bind(py)
        .call_method("update_from_kwargs", (), Some(&update))?;

    let pre_call = pre_call_args(
        "OCR document processing",
        bag.get_item("api_key")?,
        &body,
        endpoint.url(),
        &headers,
    )
    .into_pyobject(py)?;
    Py::new(
        py,
        OcrState {
            callback: RetainedCallback::new(
                RequestRoots::new(
                    arguments,
                    body.unbind().into_any(),
                    headers.unbind().into_any(),
                ),
                logging,
                pre_call.unbind(),
            ),
            endpoint: Some(endpoint),
            asynchronous,
            terminal: None,
        },
    )
}

#[pyfunction]
fn pre_call(py: Python<'_>, state: Py<OcrState>) -> PyResult<()> {
    let (logging, arguments) = {
        let state = state.borrow(py);
        (
            state.callback.logging(py, Route::Ocr)?,
            state.callback.pre_call(py, Route::Ocr)?,
        )
    };
    logging
        .bind(py)
        .call_method(pyo3::intern!(py, "pre_call"), (), Some(arguments.bind(py)))?;
    Ok(())
}

type OcrWireRequest = (SettledOcrRequest, String, String, bool);

fn request(py: Python<'_>, state: &Py<OcrState>) -> PyResult<OcrWireRequest> {
    let (endpoint, body, headers, asynchronous) = {
        let mut state = state.borrow_mut(py);
        let endpoint = state
            .endpoint
            .take()
            .ok_or(Error::RequestConsumed(Route::Ocr))?;
        let roots = state.callback.roots(Route::Ocr)?;
        let body = roots.body(py);
        let headers = roots.headers(py);
        (endpoint, body, headers, state.asynchronous)
    };
    let model = endpoint.model().to_string();
    let provider = endpoint.custom_llm_provider().to_string();
    let request = endpoint
        .settle(
            header_pairs(headers.cast::<PyDict>()?)?,
            litellm_core::lifecycle::PreCallBody::StructuredAtSend {
                callback: from_py(&body)?,
            },
        )
        .map_err(core_error_to_pyerr)?;
    Ok((request, model, provider, asynchronous))
}

#[pyfunction]
fn send(
    py: Python<'_>,
    machine: Py<OcrLifecycle>,
    state: Py<OcrState>,
) -> PyResult<Bound<'_, PyAny>> {
    let permit = machine.borrow_mut(py).provider_permit()?;
    let (request, model, provider, _asynchronous) = request(py, &state)?;
    litellm_python_interop::run_async_py(py, async move {
        let error_model = model.clone();
        let error_provider = provider.clone();
        let call_id = Python::attach(|py| {
            state
                .borrow(py)
                .callback
                .roots(Route::Ocr)
                .ok()
                .and_then(|roots| {
                    scalar(&roots.arguments(py), "litellm_call_id")
                        .ok()
                        .flatten()
                })
                .unwrap_or_default()
        });
        let executed = run_async_value(
            async move {
                Ok::<_, std::convert::Infallible>(
                    permit
                        .ocr(
                            request,
                            CallLifecycleContext::new("ocr", &model, &provider, call_id),
                        )
                        .await,
                )
            },
            |never| match never {},
        )
        .await?;
        let terminal = executed.terminal().clone();
        Python::attach(|py| state.borrow_mut(py).terminal = Some(terminal));
        match executed {
            ExecutedCall::Success { response, .. } | ExecutedCall::Deferred { response, .. } => {
                Ok(Pythonized(response))
            }
            ExecutedCall::Failure { error, .. } => Err(Python::attach(|py| {
                ocr_error_to_pyerr(py, error, &error_model, &error_provider)
            })),
        }
    })
}

#[pyfunction]
fn finish(py: Python<'_>, response: Py<PyDict>) -> PyResult<Py<PyAny>> {
    let fields = response.bind(py).copy()?;
    let native_response = fields.get_item(pyo3::intern!(py, "provider_native_response"))?;
    if native_response.is_some() {
        fields.del_item(pyo3::intern!(py, "provider_native_response"))?;
    }
    let response = py
        .import("litellm.llms.base_llm.ocr.transformation")?
        .getattr("OCRResponse")?
        .call((), Some(&fields))?;
    if let Some(native_response) = native_response.filter(|value| !value.is_none()) {
        response.call_method1(
            pyo3::intern!(py, "set_provider_native_response"),
            (native_response,),
        )?;
    }
    Ok(response.unbind())
}

#[pyfunction]
fn send_sync(
    py: Python<'_>,
    machine: Py<OcrLifecycle>,
    state: Py<OcrState>,
) -> PyResult<Py<PyAny>> {
    let permit = machine.borrow_mut(py).provider_permit()?;
    let (request, model, provider, _asynchronous) = request(py, &state)?;
    let error_model = model.clone();
    let error_provider = provider.clone();
    let call_id = state
        .borrow(py)
        .callback
        .roots(Route::Ocr)
        .ok()
        .and_then(|roots| {
            scalar(&roots.arguments(py), "litellm_call_id")
                .ok()
                .flatten()
        })
        .unwrap_or_default();
    let executed = run_sync_value(
        py,
        async move {
            Ok::<_, std::convert::Infallible>(
                permit
                    .ocr(
                        request,
                        CallLifecycleContext::new("ocr", &model, &provider, call_id),
                    )
                    .await,
            )
        },
        |never| match never {},
    )?;
    state.borrow_mut(py).terminal = Some(executed.terminal().clone());
    let response = match executed {
        ExecutedCall::Success { response, .. } | ExecutedCall::Deferred { response, .. } => {
            response
        }
        ExecutedCall::Failure { error, .. } => {
            return Err(ocr_error_to_pyerr(py, error, &error_model, &error_provider));
        }
    };
    let fields = to_py(py, &response)?.into_bound(py).cast_into::<PyDict>()?;
    let response = finish(py, fields.unbind());
    drop(state);
    response
}

#[pyfunction]
fn terminal_record(py: Python<'_>, state: Py<OcrState>) -> PyResult<Py<PyAny>> {
    let terminal = state
        .borrow(py)
        .terminal
        .clone()
        .ok_or(Error::TerminalUnavailable(Route::Ocr))?;
    to_py(py, &terminal)
}

#[pyfunction]
fn ocr(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    crate::driver::drive_sync(py, runner(py)?, arguments, bindings(py)?)
}

#[pyfunction]
fn aocr(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    crate::driver::drive_async(py, runner(py)?, arguments, bindings(py)?)
}

fn runner(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static RUNNER: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = RUNNER.get(py) {
        return Ok(module.bind(py));
    }
    let module = py.import("litellm.rust_bridge.ocr")?;
    Ok(RUNNER.get_or_init(py, || module.unbind()).bind(py))
}

fn bindings(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static BINDINGS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = BINDINGS.get(py) {
        return Ok(module.bind(py));
    }
    let module = PyModule::new(py, "_ocr_bindings")?;
    module.add("Lifecycle", py.get_type::<OcrLifecycle>())?;
    module.add("invoke", wrap_pyfunction!(invoke, &module)?)?;
    module.add("build_request", wrap_pyfunction!(build_request, &module)?)?;
    module.add("pre_call", wrap_pyfunction!(pre_call, &module)?)?;
    module.add("send", wrap_pyfunction!(send, &module)?)?;
    module.add("send_sync", wrap_pyfunction!(send_sync, &module)?)?;
    module.add("finish", wrap_pyfunction!(finish, &module)?)?;
    module.add(
        "terminal_record",
        wrap_pyfunction!(terminal_record, &module)?,
    )?;
    Ok(BINDINGS.get_or_init(py, || module.unbind()).bind(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!(ocr, module)?)?;
    crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!(aocr, module)?)?;
    Ok(())
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register(module)
}

#[cfg(test)]
#[path = "../../tests/unit/routes/ocr.rs"]
mod tests;
