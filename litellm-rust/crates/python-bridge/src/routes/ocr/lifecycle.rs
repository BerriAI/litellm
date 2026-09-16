use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_auth::ResolvedCredential;
use litellm_core::ocr::hooks::{OcrDuringCallRequest, OcrPostCallRequest, OcrPreCallRequest};
use litellm_core::ocr::{OcrAdmission, OcrCall, OcrClient, OcrHostOperation, OcrHostResult};
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};

use super::callbacks;
use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::project::{ProjectedOcrCall, ProjectedOcrFields, PythonOcrInput, admitted_call};
use crate::lifecycle::{
    OperationClass, PythonCallState, PythonRoute, missing_state, now, run_call,
};

struct PythonOcrHost {
    state: PythonCallState,
    data: OcrHostData,
}

enum OcrHostData {
    Unprojected { request: Py<PyAny> },
    Projected(Box<ProjectedOcrHost>),
    Released,
}

struct ProjectedOcrHost {
    fields: ProjectedOcrFields,
    pre_call: Option<callbacks::OcrLoggingFields>,
    callback_inputs: Option<Py<PyDict>>,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyDict>>,
}

impl PythonOcrHost {
    fn projected(&self) -> PyResult<&ProjectedOcrHost> {
        match &self.data {
            OcrHostData::Projected(projected) => Ok(projected),
            _ => Err(missing_state()),
        }
    }

    fn projected_mut(&mut self) -> PyResult<&mut ProjectedOcrHost> {
        match &mut self.data {
            OcrHostData::Projected(projected) => Ok(projected),
            _ => Err(missing_state()),
        }
    }

    fn pre_call(
        &mut self,
        py: Python<'_>,
        request: OcrPreCallRequest,
    ) -> PyResult<OcrPreCallRequest> {
        let kwargs = self.state.kwargs.bind(py);
        let callback_inputs = kwargs.copy()?;
        callback_inputs.set_item("document", &self.projected()?.fields.document)?;
        let projected = self.projected_mut()?;
        projected.callback_inputs = Some(callback_inputs.unbind());
        projected.pre_call = Some((&request).into());
        Ok(request)
    }

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        let provider = self
            .projected()?
            .fields
            .azure_ad_token_provider
            .as_ref()
            .ok_or_else(missing_state)?;
        provider.acquire(py)
    }

    fn python_pre_call(
        &mut self,
        py: Python<'_>,
        mut request: OcrDuringCallRequest,
    ) -> PyResult<OcrDuringCallRequest> {
        let projected = self.projected()?;
        let pre_call = projected.pre_call.as_ref().ok_or_else(missing_state)?;
        self.state.logger()?.update_ocr(
            py,
            &self.state.kwargs,
            pre_call,
            &projected.fields.secret_fields,
            &request.url,
        )?;
        let body = to_py(py, &request.body)?
            .into_bound(py)
            .cast_into::<PyDict>()?;
        let headers = PyDict::new(py);
        for (name, value) in &request.headers {
            headers.set_item(name, value)?;
        }
        let api_key = self.projected()?.fields.api_key.clone_ref(py);
        let projected = self.projected_mut()?;
        projected.body = Some(body.clone().unbind());
        projected.headers = Some(headers.clone().unbind());
        self.state
            .logger()?
            .pre_ocr(py, &Some(api_key), &body, &headers, &request.url)?;
        let headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        request.body = from_py(&body)?;
        request.headers = headers;
        Ok(request)
    }

    fn python_post_call(
        &mut self,
        py: Python<'_>,
        request: OcrPostCallRequest,
    ) -> PyResult<OcrPostCallRequest> {
        let projected = self.projected()?;
        self.state.logger()?.post_ocr(
            py,
            &request.original_response,
            projected.body.as_ref(),
            projected.headers.as_ref(),
        )?;
        Ok(request)
    }
}

impl PythonRoute for PythonOcrHost {
    type Call = OcrCall;

    fn state(&self) -> &PythonCallState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut PythonCallState {
        &mut self.state
    }

    fn classify(operation: &OcrHostOperation) -> OperationClass {
        operation
            .phase()
            .map_or(OperationClass::Route, OperationClass::Phase)
    }

    fn lifecycle_result() -> OcrHostResult {
        OcrHostResult::Lifecycle(Ok(()))
    }

    fn host_error(message: String) -> litellm_core::ocr::Error {
        litellm_core::ocr::Error::InvalidRequest(message)
    }

    fn map_error(error: litellm_core::ocr::Error) -> PyErr {
        ocr_error_to_pyerr(error)
    }

    fn invoke(&mut self, py: Python<'_>, operation: OcrHostOperation) -> PyResult<OcrHostResult> {
        Ok(match operation {
            OcrHostOperation::ProjectRequest => {
                let OcrHostData::Unprojected { request } = &self.data else {
                    return Err(missing_state());
                };
                let projected = ProjectedOcrCall::try_from(PythonOcrInput {
                    request: request.bind(py),
                    kwargs: self.state.kwargs.bind(py),
                })?;
                let has_token_provider = projected.fields.azure_ad_token_provider.is_some();
                let request = projected.request;
                self.data = OcrHostData::Projected(Box::new(ProjectedOcrHost {
                    fields: projected.fields,
                    pre_call: None,
                    callback_inputs: None,
                    body: None,
                    headers: None,
                }));
                OcrHostResult::Request(Ok((Box::new(request), has_token_provider)))
            }
            OcrHostOperation::AcquireAzureAdToken => {
                OcrHostResult::AzureAdToken(Ok(self.acquire_azure_ad_token(py)?))
            }
            OcrHostOperation::PreCall(request) => {
                OcrHostResult::PreCall(Ok(self.pre_call(py, request)?))
            }
            OcrHostOperation::DuringCall(request) => {
                OcrHostResult::DuringCall(Ok(self.python_pre_call(py, request)?))
            }
            OcrHostOperation::PostCall(request) => {
                OcrHostResult::PostCall(Ok(self.python_post_call(py, request)?))
            }
            OcrHostOperation::ConstructResponse(response) => {
                self.state.end = Some(now(py)?);
                self.state.response = Some(callbacks::response(py, response.as_ref())?);
                OcrHostResult::Lifecycle(Ok(()))
            }
            OcrHostOperation::MapFailure(error) => {
                if self.state.error.is_none() {
                    self.state.retain_error(py, ocr_error_to_pyerr(error));
                }
                if self.state.end.is_none() {
                    self.state.end = Some(now(py)?);
                }
                let error = self.state.error.as_ref().ok_or_else(missing_state)?;
                let (request, provider) = match &self.data {
                    OcrHostData::Unprojected { request } => (request.bind(py), ""),
                    OcrHostData::Projected(projected) => (
                        projected.fields.boundary_request.bind(py),
                        projected.fields.provider,
                    ),
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
        })
    }

    fn cleanup(&mut self) {
        self.data = OcrHostData::Released;
    }
    fn traverse(&self, visit: &pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        match &self.data {
            OcrHostData::Unprojected { request } => visit.call(request),
            OcrHostData::Projected(projected) => {
                visit.call(&projected.fields.boundary_request)?;
                visit.call(&projected.fields.document)?;
                visit.call(&projected.fields.api_key)?;
                if let Some(provider) = &projected.fields.azure_ad_token_provider {
                    provider.traverse(visit)?;
                }
                visit.call(&projected.callback_inputs)?;
                visit.call(&projected.body)?;
                visit.call(&projected.headers)
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

fn call(
    py: Python<'_>,
    args: Bound<'_, PyTuple>,
    kwargs: Option<Bound<'_, PyDict>>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let kwargs = kwargs.unwrap_or_else(|| PyDict::new(py));
    let name = if asynchronous { "aocr" } else { "ocr" };
    let request = py
        .import("litellm.rust_bridge.ocr")?
        .getattr("bind_request")?
        .call1((name, &args, &kwargs))?;
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
            name,
        )?,
        data: OcrHostData::Unprojected {
            request: request.unbind(),
        },
    };
    run_call(py, call, host)
}

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn ocr(
    py: Python<'_>,
    args: Bound<'_, PyTuple>,
    kwargs: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    call(py, args, kwargs, false)
}

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn aocr(
    py: Python<'_>,
    args: Bound<'_, PyTuple>,
    kwargs: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    call(py, args, kwargs, true)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    super::super::add_function(module, wrap_pyfunction!(ocr, module)?)?;
    super::super::add_function(module, wrap_pyfunction!(aocr, module)?)
}
