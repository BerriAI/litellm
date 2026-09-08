//! Native OCR retains the entire Python argument graph through completion.
//! Missing native operations raise NotImplementedError before
//! callbacks; no Python preparation, auth, encoding, or provider transforms run.

use litellm_core::error::Error;
use litellm_core::lifecycle::ocr::{NativeOutcome, Observations, OcrRoute, Operation, Options};
use litellm_core::lifecycle::{
    CallLifecycleContext, ErrorDisposition, ExecutedCall, Lifecycle, Outcome, TerminalRecord,
};
use litellm_core::ocr::DefaultOcrServices;
use litellm_core::ocr::types::{
    OcrAdmissionRequest, OcrDocumentProjection, OcrDraft, OcrEndpoint, SettledOcrRequest,
};
use litellm_core::routing_utils::provider::get_custom_llm_provider;
use litellm_python_interop::{Pythonized, from_py, to_py};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use crate::driver::{ADDITIONAL_ARGS, API_BASE, API_KEY, COMPLETE_INPUT_DICT, HEADERS, INPUT};
use crate::errors::core_error_to_pyerr;
use crate::retained::RequestRoots;
use litellm_python_interop::{run_async_value, run_sync_value};

#[pyclass]
struct OcrState {
    roots: Option<RequestRoots>,
    logging: Option<Py<PyAny>>,
    pre_call: Option<Py<PyDict>>,
    endpoint: Option<OcrEndpoint>,
    asynchronous: bool,
    terminal: Option<TerminalRecord>,
}

#[pymethods]
impl OcrState {
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
                state.endpoint.take(),
                state.terminal.take(),
            )
        };
        drop(roots);
    }
}

fn ocr_error_to_pyerr(py: Python<'_>, error: Error, model: &str, provider: &str) -> PyErr {
    let status = match error {
        Error::Unsupported(message) => return PyRuntimeError::new_err(message),
        Error::Auth(_) => 401,
        Error::Http { status, .. } => status,
        Error::Network(_) | Error::Connect(_) => {
            return PyRuntimeError::new_err("OCR transport failed");
        }
        Error::InvalidResponse(_) => {
            return PyRuntimeError::new_err("Invalid OCR provider response");
        }
        other => return core_error_to_pyerr(other),
    };
    let class = match status {
        400 => "BadRequestError",
        401 => "AuthenticationError",
        403 => "PermissionDeniedError",
        404 => "NotFoundError",
        422 => "UnprocessableEntityError",
        429 => "RateLimitError",
        500 => "InternalServerError",
        502 => "BadGatewayError",
        503 => "ServiceUnavailableError",
        _ => "APIError",
    };
    let exception = || -> PyResult<PyErr> {
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            "message",
            format!("OCR provider request failed (HTTP {status})"),
        )?;
        kwargs.set_item("model", model)?;
        kwargs.set_item("llm_provider", provider)?;
        if class == "APIError" {
            kwargs.set_item("status_code", status)?;
        } else {
            let httpx = py.import("httpx")?;
            let request = httpx
                .getattr("Request")?
                .call1(("POST", "https://litellm.ai"))?;
            let response_kwargs = PyDict::new(py);
            response_kwargs.set_item("request", request)?;
            let response = httpx
                .getattr("Response")?
                .call((status,), Some(&response_kwargs))?;
            kwargs.set_item("response", response)?;
        }
        let instance = py
            .import("litellm.exceptions")?
            .getattr(class)?
            .call((), Some(&kwargs))?;
        Ok(PyErr::from_value(instance))
    };
    exception().unwrap_or_else(|error| error)
}

fn scalar(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    arguments
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()
        .map(|value| value.filter(|value| !value.trim().is_empty()))
}

fn header_pairs(headers: &Bound<'_, PyDict>) -> PyResult<Vec<(String, String)>> {
    headers
        .iter()
        .map(|(name, value)| Ok((name.extract()?, value.extract()?)))
        .collect()
}

fn decode_request(py: Python<'_>, bag: &Bound<'_, PyDict>) -> PyResult<OcrAdmissionRequest> {
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
        api_key: scalar(bag, "api_key")?,
        api_base: scalar(bag, "api_base")?,
        extra_headers,
        timeout_seconds,
        request_format: scalar(bag, "req_format")?,
        document: from_py(document_input.as_any())?,
        azure_ad_token: scalar(bag, "azure_ad_token")?,
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
    error: Error,
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
        let request = decode_request(py, arguments)?;
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
    machine: Py<OcrLifecycle>,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let (operation, asynchronous) = {
        let machine = machine.borrow(py);
        (machine.machine.operation(), machine.asynchronous)
    };
    crate::driver::invoke(py, operation, asynchronous, true, "OCR", host)
}

#[pyfunction]
fn prepare(
    py: Python<'_>,
    arguments: Py<PyDict>,
    logging: Py<PyAny>,
    asynchronous: bool,
) -> PyResult<Py<OcrState>> {
    let bag = arguments.bind(py);
    let request = decode_request(py, bag)?;
    let model = request.model.clone();
    let custom_llm_provider = request.custom_llm_provider.clone();
    let draft = py
        .detach(|| litellm_core::ocr::prepare::prepare(request))
        .map_err(|error| {
            request_error_to_pyerr(py, error, &model, custom_llm_provider.as_deref())
        })?;
    let document = bag
        .get_item("document")?
        .ok_or_else(|| PyValueError::new_err("OCR requires document"))?
        .cast_into::<PyDict>()?;
    let OcrDraft {
        endpoint,
        headers: draft_headers,
        body: draft_body,
        document_projection,
        parameter_fields,
    } = draft;
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

    let additional_args = PyDict::new(py);
    additional_args.set_item(COMPLETE_INPUT_DICT, &body)?;
    additional_args.set_item(API_BASE, endpoint.url())?;
    additional_args.set_item(HEADERS, &headers)?;
    let pre_call = PyDict::new(py);
    pre_call.set_item(INPUT, "OCR document processing")?;
    pre_call.set_item(API_KEY, bag.get_item("api_key")?)?;
    pre_call.set_item(ADDITIONAL_ARGS, additional_args)?;
    Py::new(
        py,
        OcrState {
            roots: Some(RequestRoots::new(
                arguments,
                body.unbind().into_any(),
                headers.unbind().into_any(),
            )),
            logging: Some(logging),
            pre_call: Some(pre_call.unbind()),
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
        let logging = state
            .logging
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("OCR logging state was cleared"))?
            .clone_ref(py);
        let arguments = state
            .pre_call
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("OCR pre-call state was cleared"))?
            .clone_ref(py);
        (logging, arguments)
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
            .ok_or_else(|| PyRuntimeError::new_err("OCR request was already sent or cleared"))?;
        let roots = state
            .roots
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("OCR roots were cleared"))?;
        let body = roots.body(py);
        let headers = roots.headers(py);
        (endpoint, body, headers, state.asynchronous)
    };
    let model = endpoint.model().to_string();
    let provider = endpoint.custom_llm_provider().to_string();
    let request = endpoint.settle(header_pairs(headers.cast::<PyDict>()?)?, from_py(&body)?);
    Ok((request, model, provider, asynchronous))
}

#[pyfunction]
fn send(py: Python<'_>, state: Py<OcrState>) -> PyResult<Bound<'_, PyAny>> {
    let (request, model, provider, asynchronous) = request(py, &state)?;
    litellm_python_interop::run_async_py(py, async move {
        let error_model = model.clone();
        let error_provider = provider.clone();
        let call_id = Python::attach(|py| {
            state
                .borrow(py)
                .roots
                .as_ref()
                .and_then(|roots| {
                    scalar(&roots.arguments(py), "litellm_call_id")
                        .ok()
                        .flatten()
                })
                .unwrap_or_default()
        });
        let executed = run_async_value(
            async move {
                let services = DefaultOcrServices;
                Ok::<_, std::convert::Infallible>(
                    litellm_core::ocr::ocr(
                        &services,
                        request,
                        Options {
                            asynchronous,
                            ..Options::default()
                        },
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
            ExecutedCall::Success { response, .. } => Ok(Pythonized(response)),
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
fn send_sync(py: Python<'_>, state: Py<OcrState>) -> PyResult<Py<PyAny>> {
    let (request, model, provider, asynchronous) = request(py, &state)?;
    let error_model = model.clone();
    let error_provider = provider.clone();
    let call_id = state
        .borrow(py)
        .roots
        .as_ref()
        .and_then(|roots| {
            scalar(&roots.arguments(py), "litellm_call_id")
                .ok()
                .flatten()
        })
        .unwrap_or_default();
    let executed = run_sync_value(
        py,
        async move {
            let services = DefaultOcrServices;
            Ok::<_, std::convert::Infallible>(
                litellm_core::ocr::ocr(
                    &services,
                    request,
                    Options {
                        asynchronous,
                        ..Options::default()
                    },
                    CallLifecycleContext::new("ocr", &model, &provider, call_id),
                )
                .await,
            )
        },
        |never| match never {},
    )?;
    state.borrow_mut(py).terminal = Some(executed.terminal().clone());
    let response = match executed {
        ExecutedCall::Success { response, .. } => response,
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
        .ok_or_else(|| PyRuntimeError::new_err("OCR terminal record is unavailable"))?;
    to_py(py, &terminal)
}

#[pyfunction]
fn ocr(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    runner(py)?
        .getattr("_drive_sync")?
        .call1((arguments, bindings(py)?))
}

#[pyfunction]
fn aocr(py: Python<'_>, arguments: Py<PyDict>) -> PyResult<Bound<'_, PyAny>> {
    runner(py)?
        .getattr("_drive_async")?
        .call1((arguments, bindings(py)?))
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
    module.add("prepare", wrap_pyfunction!(prepare, &module)?)?;
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
mod tests {
    use super::*;
    use litellm_core::integrations::custom_logger::CallbackTiming;
    use litellm_core::integrations::types::Usage;
    use litellm_core::lifecycle::{RouteProjection, TerminalClassification};
    use serde_json::json;

    #[test]
    #[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
    fn structured_errors_use_public_sdk_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let exceptions = py.import("litellm.exceptions").unwrap();
            for (status, class) in [
                (400, "BadRequestError"),
                (401, "AuthenticationError"),
                (403, "PermissionDeniedError"),
                (404, "NotFoundError"),
                (422, "UnprocessableEntityError"),
                (429, "RateLimitError"),
                (500, "InternalServerError"),
                (502, "BadGatewayError"),
                (503, "ServiceUnavailableError"),
                (504, "APIError"),
            ] {
                let error = ocr_error_to_pyerr(
                    py,
                    Error::Http {
                        status,
                        body: "private upstream content".into(),
                    },
                    "mistral-ocr-latest",
                    "mistral",
                );
                let value = error.value(py);
                assert!(
                    value
                        .is_instance(&exceptions.getattr(class).unwrap())
                        .unwrap(),
                    "HTTP {status}: expected {class}, got {error}"
                );
                assert_eq!(
                    value
                        .getattr("status_code")
                        .unwrap()
                        .extract::<u16>()
                        .unwrap(),
                    status
                );
                assert_eq!(
                    value.getattr("model").unwrap().extract::<String>().unwrap(),
                    "mistral-ocr-latest"
                );
                assert_eq!(
                    value
                        .getattr("llm_provider")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "mistral"
                );
                assert!(!error.to_string().contains("private upstream content"));
            }
            let error = ocr_error_to_pyerr(
                py,
                Error::Auth("private credential".into()),
                "model",
                "mistral",
            );
            assert!(
                error
                    .value(py)
                    .is_instance(&exceptions.getattr("AuthenticationError").unwrap())
                    .unwrap()
            );
            assert!(!error.to_string().contains("private credential"));
        });
    }

    #[test]
    fn unstructured_transport_errors_are_not_guessed_from_strings() {
        Python::initialize();
        Python::attach(|py| {
            for error in [
                Error::Network("timeout secret".into()),
                Error::Connect("401 secret".into()),
                Error::InvalidResponse("429 secret".into()),
            ] {
                let error = ocr_error_to_pyerr(py, error, "model", "mistral");
                assert!(error.is_instance_of::<PyRuntimeError>(py));
                assert!(!error.to_string().contains("secret"));
            }
        });
    }

    #[test]
    fn finish_does_not_modify_the_input_dictionary() {
        Python::initialize();
        Python::attach(|py| {
            let fields = PyDict::new(py);
            fields
                .set_item("provider_native_response", "native")
                .unwrap();
            fields.set_item("model", "model").unwrap();

            let _ = finish(py, fields.clone().unbind());

            assert_eq!(
                fields
                    .get_item("provider_native_response")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "native"
            );
        });
    }

    #[test]
    fn terminal_record_exports_core_timing() {
        Python::initialize();
        Python::attach(|py| {
            let state = Py::new(
                py,
                OcrState {
                    roots: None,
                    logging: None,
                    pre_call: None,
                    endpoint: None,
                    asynchronous: false,
                    terminal: Some(TerminalRecord {
                        call_id: "call-1".into(),
                        trace_id: None,
                        attempt: 1,
                        call_type: "ocr".into(),
                        model: "model".into(),
                        provider: "mistral".into(),
                        timing: CallbackTiming::new(10.25, 12.5),
                        usage: Usage::default(),
                        cost_inputs: Default::default(),
                        classification: TerminalClassification::Success,
                        projection: RouteProjection::Ocr {
                            value: json!({"pages": []}),
                        },
                    }),
                },
            )
            .unwrap();

            let record = terminal_record(py, state).unwrap();
            let timing = record.bind(py).get_item("timing").unwrap();
            assert_eq!(
                timing
                    .get_item("start_time")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                10.25
            );
            assert_eq!(
                timing
                    .get_item("end_time")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                12.5
            );
        });
    }

    #[test]
    #[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
    fn callback_decline_is_terminal_and_identity_is_reused() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "ocr_test").unwrap();
            module
                .add_function(wrap_pyfunction!(ocr, &module).unwrap())
                .unwrap();
            module
                .add_function(wrap_pyfunction!(aocr, &module).unwrap())
                .unwrap();
            let globals = PyDict::new(py);
            globals.set_item("native", module).unwrap();
            py.run(
                c"
import asyncio
import contextvars
import threading
from datetime import datetime

marker = contextvars.ContextVar('terminal_marker')

class Logger:
    litellm_call_id = 'supplied-call'
    litellm_trace_id = 'supplied-trace'

    def update_from_kwargs(self, **values):
        assert values['kwargs']['litellm_call_id'] == self.litellm_call_id
        assert values['kwargs']['litellm_trace_id'] == self.litellm_trace_id
        assert threading.get_ident() == self.thread
        marker.set('update')
        raise self.original

    def failure_handler(self, error, trace, start, end):
        assert error is self.original
        assert marker.get() == 'update'
        assert start <= end <= datetime.now()
        self.end = end
        self.calls.append('failure')

    async def async_failure_handler(self, error, trace, start, end):
        await asyncio.sleep(0)
        assert asyncio.current_task() is self.task
        assert marker.get() == 'update'
        assert error is self.original
        assert end is self.end
        self.calls.append('async_failure')

    def _restore_correlation_context(self):
        self.calls.append('restore')

async def exercise():
    for asynchronous in (False, True):
        logger = Logger()
        logger.thread = threading.get_ident()
        logger.task = asyncio.current_task()
        logger.calls = []
        logger.original = NotImplementedError('callback declined, not admission')
        arguments = dict(model='mistral/mistral-ocr-latest', api_key='test-key', timeout=1.0,
                         document={'type': 'document_url', 'document_url': 'https://example.test/doc.pdf'},
                         litellm_logging_obj=logger)
        try:
            if asynchronous:
                await native.aocr(arguments)
            else:
                native.ocr(arguments)
        except NotImplementedError as error:
            assert error is logger.original
        else:
            raise AssertionError('callback exception was lost')
        assert logger.calls == (['failure', 'async_failure', 'restore'] if asynchronous else ['failure', 'restore'])
        assert arguments['litellm_call_id'] == 'supplied-call'
        assert arguments['litellm_trace_id'] == 'supplied-trace'

asyncio.run(exercise())
",
                Some(&globals),
                Some(&globals),
            ).unwrap();
        });
    }

    #[test]
    #[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
    fn native_send_owns_state_without_the_python_driver() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "ocr_test").unwrap();
            module
                .add_function(wrap_pyfunction!(prepare, &module).unwrap())
                .unwrap();
            module
                .add_function(wrap_pyfunction!(pre_call, &module).unwrap())
                .unwrap();
            module
                .add_function(wrap_pyfunction!(send, &module).unwrap())
                .unwrap();
            let globals = PyDict::new(py);
            globals.set_item("native", module).unwrap();
            py.run(
                cr"
import asyncio
import gc
import weakref

class Logger:
    def update_from_kwargs(self, **values):
        pass
    def pre_call(self, **values):
        pass

async def exercise():
    received = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def respond(reader, writer):
        await reader.readuntil(b'\r\n\r\n')
        received.set()
        await release.wait()
        writer.close()
        await writer.wait_closed()
        closed.set()

    server = await asyncio.start_server(respond, '127.0.0.1', 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        logger = Logger()
        alive = weakref.ref(logger)
        state = native.prepare(dict(
            model='mistral/mistral-ocr-latest', api_key='test-key', timeout=5.0,
            api_base=f'http://127.0.0.1:{port}', litellm_logging_obj=logger,
            document={'type': 'document_url', 'document_url': 'https://example.test/doc.pdf'},
        ), logger, True)
        pending = native.send(state)
        del state, logger
        try:
            await asyncio.wait_for(received.wait(), 5)
            gc.collect()
            assert alive() is not None
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
            for _ in range(500):
                await asyncio.sleep(0.01)
                gc.collect()
                if alive() is None:
                    break
            assert alive() is None
        finally:
            release.set()
            await asyncio.wait_for(closed.wait(), 5)

asyncio.run(exercise())
",
                Some(&globals),
                Some(&globals),
            )
            .unwrap();
        });
    }

    #[pyfunction]
    fn snapshot(py: Python<'_>, state: Py<OcrState>) -> PyResult<Py<PyAny>> {
        let state = state.borrow(py);
        let roots = state
            .roots
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("OCR roots were cleared"))?;
        let headers = header_pairs(roots.headers(py).cast::<PyDict>()?)?;
        let body: serde_json::Value = from_py(&roots.body(py))?;
        to_py(py, &(headers, body))
    }

    #[test]
    #[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
    fn retains_identity_independent_wire_roots_and_collects_cycles() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "ocr_test").unwrap();
            module
                .add_function(wrap_pyfunction!(prepare, &module).unwrap())
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

class Timeout:
    read = 5.0

class Logger:
    def update_from_kwargs(self, **values):
        assert values['kwargs'] is arguments
        assert values['kwargs']['metadata'] is metadata
        assert values['kwargs']['opaque'] is opaque
        assert values['optional_params']['pages'] is pages
        self.calls = ['update']

    def pre_call(self, **values):
        self.calls.append('pre')
        view = values['additional_args']
        self.body = view['complete_input_dict']
        self.headers = view['headers']
        assert self.body['document'] is document
        assert self.body['pages'] is pages
        document['document_url'] = 'https://example.test/changed.pdf'
        pages.append(2)
        self.headers['x-hook'] = 'changed'
        view['complete_input_dict'] = {'replacement': True}
        view['headers'] = {'replacement': 'true'}

document = {'type': 'document_url', 'document_url': 'https://example.test/test.pdf'}
pages = [0]
metadata = {'nested': []}
opaque = Opaque()
logger = Logger()
arguments = dict(model='mistral/mistral-ocr-latest', document=document,
                 api_key='test-key', pages=pages, metadata=metadata,
                 opaque=opaque, litellm_logging_obj=logger, timeout=Timeout())
state = native.prepare(arguments, logger, False)
native.pre_call(state)
assert logger.calls == ['update', 'pre']
roots = gc.get_referents(state)
assert any(root is arguments for root in roots)
assert any(root is logger for root in roots)
assert any(root is logger.body for root in roots)
assert any(root is logger.headers for root in roots)
headers, body = native.snapshot(state)
assert body['document']['document_url'] == 'https://example.test/changed.pdf'
assert body['pages'] == [0, 2]
assert dict(headers)['x-hook'] == 'changed'
assert 'replacement' not in body and 'replacement' not in dict(headers)

arguments['cycle'] = state
logger.cycle = state
logger.body['cycle'] = state
logger.headers['cycle'] = state
alive = weakref.ref(opaque)
del roots, arguments, logger, opaque
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
    #[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
    fn async_callbacks_are_inline_and_unsupported_requests_never_call_them() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "ocr_test").unwrap();
            module
                .add_function(wrap_pyfunction!(ocr, &module).unwrap())
                .unwrap();
            module
                .add_function(wrap_pyfunction!(aocr, &module).unwrap())
                .unwrap();
            let globals = PyDict::new(py);
            globals.set_item("native", module).unwrap();
            py.run(
                cr"
import asyncio
import contextvars
import gc
import json
import threading
import weakref

marker = contextvars.ContextVar('ocr_marker')

class Opaque:
    pass

class Logger:
    def __init__(self):
        self.calls = []

    def failure_handler(self, error, trace, start, end):
        assert asyncio.current_task() is caller
        assert marker.get() == 'pre'
        self.error = error

    async def async_failure_handler(self, error, trace, start, end):
        await asyncio.sleep(0)
        assert asyncio.current_task() is caller
        assert error is self.error

    def update_from_kwargs(self, **values):
        assert asyncio.current_task() is caller
        assert threading.get_ident() == caller_thread
        assert marker.get() == 'caller'
        assert values['kwargs'] is not arguments
        assert values['kwargs']['opaque'] is arguments['opaque']
        self.calls.append('update')
        marker.set('updated')

    def pre_call(self, **values):
        assert asyncio.current_task() is caller
        assert threading.get_ident() == caller_thread
        assert marker.get() == 'updated'
        self.calls.append('pre')
        values['additional_args']['complete_input_dict']['pages'].append(3)
        marker.set('pre')

async def exercise():
    global arguments, caller, caller_thread
    caller = asyncio.current_task()
    caller_thread = threading.get_ident()
    marker.set('caller')
    logger = Logger()
    document = {'type': 'document_url', 'document_url': 'https://example.test/test.pdf'}
    arguments = dict(model='mistral/mistral-ocr-latest', document=document,
                     api_key='test-key', pages=[0], opaque=Opaque(),
                     litellm_logging_obj=logger)
    alive = weakref.ref(arguments['opaque'])
    received = asyncio.Event()
    errors = []

    async def respond(reader, writer):
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            length = next(int(line.split(b':', 1)[1]) for line in header.split(b'\r\n')
                          if line.lower().startswith(b'content-length:'))
            body = json.loads(await reader.readexactly(length))
            assert body['pages'] == [0, 3]
            assert 'opaque' not in body
            gc.collect()
            assert alive() is not None
            assert logger.calls == ['update', 'pre']
            globals().pop('arguments')
            gc.collect()
            assert alive() is not None
        except BaseException as error:
            errors.append(error)
        finally:
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nx')
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            received.set()

    server = await asyncio.start_server(respond, '127.0.0.1', 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        arguments['api_base'] = f'http://127.0.0.1:{port}'
        arguments['timeout'] = 5.0
        try:
            await native.aocr(arguments)
        except RuntimeError as error:
            assert error is logger.error
        else:
            raise AssertionError('expected upstream error')
        await asyncio.wait_for(received.wait(), 5)
    assert not errors, errors
    assert marker.get() == 'pre'
    assert logger.calls == ['update', 'pre']

    for model, doc in [
        ('azure_ai/doc-intelligence/prebuilt-read', document),
        ('vertex_ai/ocr', document),
        ('mistral/mistral-ocr-latest', {'type': 'file', 'file': Opaque()}),
    ]:
        logger.calls.clear()
        unsupported = dict(model=model, document=doc, api_key='test-key', timeout=5.0,
                           litellm_logging_obj=logger)
        for asynchronous in (False, True):
            try:
                if asynchronous:
                    await native.aocr(unsupported)
                else:
                    native.ocr(unsupported)
            except NotImplementedError:
                pass
            else:
                raise AssertionError('expected strict unsupported error')
            assert logger.calls == []

asyncio.run(exercise())
",
                Some(&globals),
                Some(&globals),
            )
            .unwrap();
        });
    }
}
