use litellm_core::call_lifecycle::host::{CallHost, HostLifecycle, HostPhase, HostStep};
use litellm_core::ocr::wire::{OcrWireRequest, decode_request};
use litellm_core::ocr::{OcrProviderResponse, PreparedOcrCall};
use litellm_python_interop::{PythonCoroutine, from_py, to_py};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::ocr_error_to_pyerr;
use crate::execution::run_sync_value;
use crate::lifecycle::{PythonCallState, PythonLifecycle, now};

#[pyclass]
struct PreparedTransport(Option<PreparedOcrCall>);

#[pyclass]
struct ProviderResponse(Option<OcrProviderResponse>);

struct PythonOcrHost {
    state: PythonCallState,
    request: Option<Py<PyAny>>,
    transport: Option<Py<PyAny>>,
    body: Option<Py<PyAny>>,
    headers: Option<Py<PyDict>>,
    document: Option<Py<PyAny>>,
    api_key: Option<Py<PyAny>>,
    failed_phase: Option<HostPhase>,
}

impl PythonOcrHost {
    fn request<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = self.state.kwargs.bind(py);
        let constructor = PyDict::new(py);
        for name in [
            "model",
            "document",
            "api_key",
            "api_base",
            "timeout",
            "custom_llm_provider",
            "extra_headers",
        ] {
            constructor.set_item(
                name,
                kwargs
                    .get_item(name)?
                    .unwrap_or_else(|| py.None().into_bound(py)),
            )?;
        }
        let extra = PyDict::new(py);
        for (key, value) in kwargs.iter() {
            if !constructor.contains(&key)? {
                extra.set_item(key, value)?;
            }
        }
        constructor.set_item("kwargs", extra)?;
        let request = py
            .import("litellm.rust_bridge.ocr")?
            .getattr("LiteLLMOcrRequest")?
            .call((), Some(&constructor))?;
        self.request = Some(request.clone().unbind());
        Ok(request)
    }

    fn prepare_transport(&mut self, py: Python<'_>) -> PyResult<HostStep<Py<PyAny>, Py<PyAny>>> {
        let request = self.request(py)?;
        let wire = py
            .import("litellm.rust_bridge.ocr_lifecycle")?
            .getattr("wire_request")?
            .call1((request,))?;
        self.document = Some(wire.get_item("document")?.unbind());
        self.api_key = Some(wire.get_item("api_key")?.unbind());
        let wire: OcrWireRequest = from_py(&wire)?;
        let request = decode_request(wire).map_err(ocr_error_to_pyerr)?;
        let future = async move {
            PreparedOcrCall::new(request)
                .await
                .map(|request| PreparedTransport(Some(request)))
                .map_err(ocr_error_to_pyerr)
        };
        if self.state.asynchronous {
            Ok(HostStep::Suspend(
                pyo3_async_runtimes::tokio::future_into_py(py, future)?.unbind(),
            ))
        } else {
            Ok(HostStep::Ready(
                Py::new(py, run_sync_value(py, future)?)?.into_any(),
            ))
        }
    }

    fn pre_call(&mut self, py: Python<'_>) -> PyResult<()> {
        let transport = self
            .transport
            .as_ref()
            .ok_or_else(missing_state)?
            .bind(py)
            .extract::<PyRef<PreparedTransport>>()?;
        let transport = transport.0.as_ref().ok_or_else(missing_state)?;
        let body: serde_json::Value = serde_json::from_slice(
            transport
                .http
                .body()
                .and_then(|body| body.as_bytes())
                .unwrap_or_default(),
        )
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let document_matches = self
            .document
            .as_ref()
            .map(|document| from_py::<serde_json::Value>(document.bind(py)))
            .transpose()?;
        let retain_document = body.get("document") == document_matches.as_ref();
        let body = to_py(py, &body)?.into_bound(py).cast_into::<PyDict>()?;
        if let Some(document) = &self.document
            && retain_document
        {
            body.set_item("document", document)?;
        }
        let headers = PyDict::new(py);
        for (name, value) in transport.http.headers() {
            headers.set_item(
                name.as_str(),
                value
                    .to_str()
                    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
            )?;
        }
        let logger = self.state.logger(py)?;
        let redact = py
            .import("litellm.rust_bridge.ocr")?
            .getattr("redact_logging_params")?;
        let update = PyDict::new(py);
        update.set_item("kwargs", redact.call1((&self.state.kwargs,))?)?;
        update.set_item("model", &transport.request().model)?;
        update.set_item(
            "optional_params",
            redact.call1((to_py(py, &transport.request().optional_params)?,))?,
        )?;
        let params = PyDict::new(py);
        params.set_item(
            "litellm_call_id",
            self.state.kwargs.bind(py).get_item("litellm_call_id")?,
        )?;
        params.set_item("api_base", transport.http.url().as_str())?;
        update.set_item("litellm_params", params)?;
        update.set_item("custom_llm_provider", transport.provider())?;
        logger.call_method("update_from_kwargs", (), Some(&update))?;
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", &body)?;
        additional.set_item("headers", &headers)?;
        additional.set_item("api_base", transport.http.url().as_str())?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", "OCR document processing")?;
        kwargs.set_item("api_key", &self.api_key)?;
        kwargs.set_item("additional_args", additional)?;
        self.body = Some(body.unbind().into_any());
        self.headers = Some(headers.unbind());
        logger.call_method("pre_call", (), Some(&kwargs))?;
        Ok(())
    }

    fn execute(&mut self, py: Python<'_>) -> PyResult<HostStep<Py<PyAny>, Py<PyAny>>> {
        let mut transport = self
            .transport
            .as_ref()
            .ok_or_else(missing_state)?
            .bind(py)
            .extract::<PyRefMut<PreparedTransport>>()?
            .0
            .take()
            .ok_or_else(missing_state)?;
        let body: serde_json::Value =
            from_py(self.body.as_ref().ok_or_else(missing_state)?.bind(py))?;
        *transport.http.body_mut() = Some(
            serde_json::to_vec(&body)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
                .into(),
        );
        let headers: std::collections::BTreeMap<String, String> = self
            .headers
            .as_ref()
            .ok_or_else(missing_state)?
            .extract(py)?;
        transport.http.headers_mut().clear();
        for (name, value) in headers {
            transport.http.headers_mut().insert(
                name.parse::<reqwest::header::HeaderName>()
                    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
                value
                    .parse()
                    .map_err(|error: reqwest::header::InvalidHeaderValue| {
                        pyo3::exceptions::PyValueError::new_err(error.to_string())
                    })?,
            );
        }
        let future = async move {
            transport
                .execute()
                .await
                .map(|response| ProviderResponse(Some(response)))
                .map_err(ocr_error_to_pyerr)
        };
        if self.state.asynchronous {
            Ok(HostStep::Suspend(
                pyo3_async_runtimes::tokio::future_into_py(py, future)?.unbind(),
            ))
        } else {
            Ok(HostStep::Ready(
                Py::new(py, run_sync_value(py, future)?)?.into_any(),
            ))
        }
    }
}

fn missing_state() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("missing native call state")
}

impl CallHost for PythonOcrHost {
    type Value = Py<PyAny>;
    type Error = PyErr;
    type Suspension = Py<PyAny>;

    fn invoke(&mut self, phase: HostPhase) -> PyResult<HostStep<Self::Value, Self::Suspension>> {
        Python::attach(|py| {
            match phase {
                HostPhase::Setup => self.state.setup(py)?,
                HostPhase::DeploymentPreCall => {
                    return Ok(HostStep::Suspend(
                        py.import("litellm.utils")?
                            .getattr("async_pre_call_deployment_hook")?
                            .call1((&self.state.kwargs, self.state.call_type))?
                            .unbind(),
                    ));
                }
                HostPhase::Prepare => self.state.prepare(py)?,
                HostPhase::PrepareTransport => return self.prepare_transport(py),
                HostPhase::PreCall => self.pre_call(py)?,
                HostPhase::Execute => return self.execute(py),
                HostPhase::PostCall => {
                    let response = self
                        .transport
                        .as_ref()
                        .ok_or_else(missing_state)?
                        .bind(py)
                        .extract::<PyRef<ProviderResponse>>()?;
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("api_key", &self.api_key)?;
                    kwargs.set_item(
                        "original_response",
                        &response.0.as_ref().ok_or_else(missing_state)?.text,
                    )?;
                    let additional = PyDict::new(py);
                    additional.set_item("complete_input_dict", &self.body)?;
                    kwargs.set_item("additional_args", additional)?;
                    self.state
                        .logger(py)?
                        .call_method("post_call", (), Some(&kwargs))?;
                }
                HostPhase::ConstructResponse => {
                    let response = self
                        .transport
                        .as_ref()
                        .ok_or_else(missing_state)?
                        .bind(py)
                        .extract::<PyRefMut<ProviderResponse>>()?
                        .0
                        .take()
                        .ok_or_else(missing_state)?;
                    let normalized = response
                        .normalize()
                        .map_err(ocr_error_to_pyerr)?
                        .into_json();
                    return Ok(HostStep::Ready(
                        py.import("litellm.rust_bridge.ocr")?
                            .getattr("_response")?
                            .call1((to_py(py, &normalized)?,))?
                            .unbind(),
                    ));
                }
                HostPhase::DeploymentPostCall => {
                    return Ok(HostStep::Suspend(
                        py.import("litellm.utils")?
                            .getattr("async_post_call_success_deployment_hook")?
                            .call1((
                                &self.state.kwargs,
                                &self.state.response,
                                self.state.call_type,
                            ))?
                            .unbind(),
                    ));
                }
                HostPhase::Finalize => self.state.finalize(py)?,
                HostPhase::Success => self.state.dispatch_success(py)?,
                HostPhase::MapFailure => {
                    if !matches!(
                        self.failed_phase,
                        Some(
                            HostPhase::PrepareTransport
                                | HostPhase::PreCall
                                | HostPhase::Execute
                                | HostPhase::PostCall
                                | HostPhase::ConstructResponse
                        )
                    ) {
                        return Ok(HostStep::Ready(py.None()));
                    }
                    let error = self
                        .state
                        .error
                        .as_ref()
                        .ok_or_else(missing_state)?
                        .clone_ref(py);
                    let request = self.request(py)?;
                    let mapped = py
                        .import("litellm.rust_bridge.ocr_lifecycle")?
                        .getattr("map_failure")?
                        .call1((error.value(py), request))?;
                    self.state.error = Some(PyErr::from_value(mapped));
                }
                HostPhase::DeploymentFailure => {
                    if !matches!(
                        self.failed_phase,
                        Some(
                            HostPhase::PrepareTransport
                                | HostPhase::PreCall
                                | HostPhase::Execute
                                | HostPhase::PostCall
                                | HostPhase::ConstructResponse
                        )
                    ) {
                        return Ok(HostStep::Ready(py.None()));
                    }
                    if let Some(error) = &self.state.error {
                        return Ok(HostStep::Suspend(
                            py.import("litellm.utils")?
                                .getattr("async_post_call_failure_deployment_hook")?
                                .call1((&self.state.kwargs, error.value(py), self.state.call_type))?
                                .unbind(),
                        ));
                    }
                }
                HostPhase::Failure | HostPhase::AsyncFailure => {
                    if let Some(awaitable) = self
                        .state
                        .dispatch_failure(py, phase == HostPhase::AsyncFailure)?
                    {
                        return Ok(HostStep::Suspend(awaitable));
                    }
                }
                HostPhase::Complete => {}
            }
            Ok(HostStep::Ready(py.None()))
        })
    }

    fn accept(&mut self, phase: HostPhase, value: Self::Value) -> PyResult<()> {
        Python::attach(|py| {
            match phase {
                HostPhase::DeploymentPreCall => {
                    self.state.kwargs = value.into_bound(py).cast_into::<PyDict>()?.unbind()
                }
                HostPhase::PrepareTransport => self.transport = Some(value),
                HostPhase::Execute => {
                    self.transport = Some(value);
                    self.state.end = Some(now(py)?);
                }
                HostPhase::ConstructResponse | HostPhase::DeploymentPostCall => {
                    self.state.response = Some(value)
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn retain_failure(&mut self, phase: HostPhase, error: PyErr) {
        self.failed_phase = Some(phase);
        self.state.error = Some(error);
        if self.state.end.is_none() {
            self.state.end = Python::attach(now).ok();
        }
    }

    fn is_cancellation(error: &PyErr) -> bool {
        Python::attach(|py| !error.is_instance_of::<pyo3::exceptions::PyException>(py))
    }

    fn finish(&mut self) -> PyResult<Self::Value> {
        if let Some(error) = self.state.error.take() {
            return Err(error);
        }
        self.state.response.take().ok_or_else(missing_state)
    }

    fn cleanup(&mut self) {
        Python::attach(|py| self.state.cleanup(py));
        self.request = None;
        self.transport = None;
        self.body = None;
        self.headers = None;
        self.document = None;
        self.api_key = None;
    }
}

#[pyfunction]
fn _ocr_lifecycle(
    py: Python<'_>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let host = PythonOcrHost {
        state: PythonCallState::new(
            py,
            kwargs.copy()?.unbind(),
            asynchronous,
            if asynchronous { "aocr" } else { "ocr" },
        )?,
        request: None,
        transport: None,
        body: None,
        headers: None,
        document: None,
        api_key: None,
        failed_phase: None,
    };
    let mut lifecycle = HostLifecycle::new(host, asynchronous);
    if asynchronous {
        return Ok(Py::new(py, PythonCoroutine::new(PythonLifecycle(lifecycle)))?.into_any());
    }
    match lifecycle.resume(None)? {
        HostStep::Ready(value) => Ok(value),
        HostStep::Suspend(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "sync call suspended",
        )),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_ocr_lifecycle, module)?)
}
