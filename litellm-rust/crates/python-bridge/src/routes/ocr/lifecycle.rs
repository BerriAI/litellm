use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_auth::ResolvedCredential;
use litellm_callbacks::protocol::{HostPhase, HostStep};
use litellm_callbacks_legacy::{OcrRequestFacts, OcrRequestPayload, PythonCallState};
use litellm_core::ocr::hooks::{OcrDuringCallRequest, OcrPostCallRequest, OcrPreCallRequest};
use litellm_core::ocr::{OcrAdmission, OcrCall, OcrClient, OcrHostOperation, OcrHostResult};

use super::callbacks;
use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::project::{OcrRetained, admitted_call, project_request};
use crate::lifecycle::{PythonHost, missing_state, run_call};

fn projected_mut(data: &mut OcrHostData) -> PyResult<&mut OcrRetained> {
    match data {
        OcrHostData::Projected(projected) => Ok(projected),
        _ => Err(missing_state()),
    }
}

struct PythonOcrHost {
    state: PythonCallState,
    phase: Option<HostPhase>,
    data: OcrHostData,
}

enum OcrHostData {
    Unprojected { request: Py<PyAny> },
    Projected(Box<OcrRetained>),
    Released,
}

impl PythonOcrHost {
    fn projected(&self) -> PyResult<&OcrRetained> {
        match &self.data {
            OcrHostData::Projected(projected) => Ok(projected),
            _ => Err(missing_state()),
        }
    }

    fn pre_call(
        &mut self,
        py: Python<'_>,
        request: OcrPreCallRequest,
    ) -> PyResult<OcrPreCallRequest> {
        let Self { state, data, .. } = self;
        let optional_params = request
            .optional_params
            .as_object()
            .ok_or_else(missing_state)?;
        projected_mut(data)?.callbacks.pre_call(
            py,
            state.kwargs(),
            optional_params,
            &request.document,
        )?;
        Ok(request)
    }

    fn read_document(&self, py: Python<'_>) -> PyResult<litellm_core::ocr::OcrFileContent> {
        self.projected()?
            .reader
            .as_ref()
            .ok_or_else(missing_state)?
            .read(py)
    }

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        let provider = self
            .projected()?
            .azure_ad_token_provider
            .as_ref()
            .ok_or_else(missing_state)?;
        provider.acquire(py)
    }

    fn during_call(
        &mut self,
        py: Python<'_>,
        request: OcrDuringCallRequest,
    ) -> PyResult<OcrDuringCallRequest> {
        let Self { state, data, .. } = self;
        let payload = projected_mut(data)?.callbacks.during_call(
            py,
            state.logger()?,
            state.kwargs(),
            OcrRequestFacts {
                model: &request.model,
                custom_llm_provider: &request.custom_llm_provider,
                url: &request.url,
                optional_params: &request.optional_params,
            },
            OcrRequestPayload {
                body: request.body,
                headers: request.headers,
            },
            &request.retained_fields,
        )?;
        Ok(OcrDuringCallRequest {
            body: payload.body,
            headers: payload.headers,
            ..request
        })
    }

    fn post_call(
        &self,
        py: Python<'_>,
        request: OcrPostCallRequest,
    ) -> PyResult<OcrPostCallRequest> {
        self.projected()?.callbacks.post_call(
            py,
            self.state.logger()?,
            &request.original_response,
        )?;
        Ok(request)
    }
}

impl PythonOcrHost {
    fn lifecycle(
        &mut self,
        py: Python<'_>,
        phase: HostPhase,
    ) -> PyResult<HostStep<OcrHostResult, Py<PyAny>>> {
        self.phase = Some(phase);
        match self.state.invoke(py, phase)? {
            HostStep::Suspend(awaitable) => Ok(HostStep::Suspend(awaitable)),
            HostStep::Ready(value) => {
                self.state.accept(py, phase, value)?;
                self.phase = None;
                Ok(HostStep::Ready(OcrHostResult::Lifecycle(Ok(()))))
            }
        }
    }
}

impl PythonHost for PythonOcrHost {
    type Call = OcrCall;

    fn asynchronous(&self) -> bool {
        self.state.asynchronous()
    }

    fn map_error(error: litellm_core::ocr::Error) -> PyErr {
        ocr_error_to_pyerr(error)
    }

    fn invoke(
        &mut self,
        py: Python<'_>,
        operation: OcrHostOperation,
    ) -> PyResult<HostStep<OcrHostResult, Py<PyAny>>> {
        if let Some(phase) = operation.phase() {
            return self.lifecycle(py, phase);
        }
        Ok(HostStep::Ready(match operation {
            OcrHostOperation::ProjectRequest => {
                let OcrHostData::Unprojected { request } = &self.data else {
                    return Err(missing_state());
                };
                let projection = project_request(request.bind(py), self.state.kwargs().bind(py))?;
                let has_token_provider = projection.retained.azure_ad_token_provider.is_some();
                self.data = OcrHostData::Projected(Box::new(projection.retained));
                OcrHostResult::Request(Ok((Box::new(projection.native), has_token_provider)))
            }
            OcrHostOperation::ReadDocument => OcrHostResult::Document(Ok(self.read_document(py)?)),
            OcrHostOperation::AcquireAzureAdToken => {
                OcrHostResult::AzureAdToken(Ok(self.acquire_azure_ad_token(py)?))
            }
            OcrHostOperation::PreCall(request) => {
                OcrHostResult::PreCall(Ok(self.pre_call(py, request)?))
            }
            OcrHostOperation::DuringCall(request) => {
                OcrHostResult::DuringCall(Ok(self.during_call(py, request)?))
            }
            OcrHostOperation::PostCall(request) => {
                OcrHostResult::PostCall(Ok(self.post_call(py, request)?))
            }
            OcrHostOperation::ConstructResponse(response) => {
                self.state
                    .record_response(py, callbacks::response(py, response.as_ref())?)?;
                OcrHostResult::Lifecycle(Ok(()))
            }
            OcrHostOperation::MapFailure(error) => {
                self.state
                    .record_failure(py, ocr_error_to_pyerr(error), false, None);
                let error = self.state.error().ok_or_else(missing_state)?;
                let (request, provider) = match &self.data {
                    OcrHostData::Unprojected { request } => (request.bind(py), ""),
                    OcrHostData::Projected(projected) => {
                        (projected.boundary_request.bind(py), projected.provider)
                    }
                    OcrHostData::Released => return Err(missing_state()),
                };
                let mapped = callbacks::map_failure(py, error, request, provider)?;
                self.state
                    .retain_error(py, PyErr::from_value(mapped.into_bound(py).into_any()));
                OcrHostResult::Lifecycle(Ok(()))
            }
            OcrHostOperation::Lifecycle(_)
            | OcrHostOperation::Success { .. }
            | OcrHostOperation::Failure { .. } => return Err(missing_state()),
        }))
    }

    fn resume(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<OcrHostResult> {
        let phase = self.phase.take().ok_or_else(missing_state)?;
        self.state.accept(py, phase, value)?;
        Ok(OcrHostResult::Lifecycle(Ok(())))
    }

    fn fail(&mut self, py: Python<'_>, error: PyErr, cancelled: bool) -> litellm_core::ocr::Error {
        let native = litellm_core::ocr::Error::InvalidRequest(error.to_string());
        self.state
            .record_failure(py, error, cancelled, self.phase.take());
        native
    }

    fn complete(
        &mut self,
        _: Python<'_>,
        _: litellm_core::ocr::LiteLLMOcrResponse,
    ) -> PyResult<Py<PyAny>> {
        self.state.take_response().ok_or_else(missing_state)
    }

    fn take_error(&mut self, py: Python<'_>) -> Option<PyErr> {
        self.state.take_error(py)
    }

    fn cleanup(&mut self, py: Python<'_>) {
        self.state.cleanup(py);
        self.data = OcrHostData::Released;
    }

    fn traverse(&self, visit: &pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        self.state.traverse(visit)?;
        match &self.data {
            OcrHostData::Unprojected { request } => visit.call(request),
            OcrHostData::Projected(projected) => {
                visit.call(&projected.boundary_request)?;
                if let Some(reader) = &projected.reader {
                    reader.traverse(visit)?;
                }
                if let Some(provider) = &projected.azure_ad_token_provider {
                    provider.traverse(visit)?;
                }
                projected.callbacks.traverse(visit)
            }
            OcrHostData::Released => Ok(()),
        }
    }
}

pub(super) struct BridgeOcrHooks;

impl litellm_core::ocr::hooks::OcrHooks for BridgeOcrHooks {
    fn intercepts_requests(&self) -> bool {
        true
    }
}

fn run_ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let client = OcrClient::shared().map_err(ocr_error_to_pyerr)?;
    let call = admitted_call(OcrCall::admit(
        client,
        OcrAdmission {
            asynchronous,
            ..OcrAdmission::all()
        },
    ))?;
    let host = PythonOcrHost {
        state: PythonCallState::new(
            py,
            args.unbind(),
            kwargs.copy()?.unbind(),
            asynchronous,
            if asynchronous { "aocr" } else { "ocr" },
        )?,
        phase: None,
        data: OcrHostData::Unprojected {
            request: request.unbind(),
        },
    };
    run_call(py, call, host)
}

#[pyfunction]
fn ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, false)
}

#[pyfunction]
fn aocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, true)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)
}
