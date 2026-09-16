use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_auth::ResolvedCredential;
use litellm_core::ocr::hooks::{OcrDuringCallRequest, OcrPostCallRequest};
use litellm_core::ocr::{OcrAdmission, OcrCall, OcrClient, OcrHostOperation, OcrHostResult};
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};

use super::callbacks;
use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::project::{admitted_call, project};
use crate::auth::PythonTokenProvider;
use crate::lifecycle::{
    OperationClass, PythonCallState, PythonRoute, Signature, missing_state, now, run_call,
};

const SIGNATURE: Signature = Signature {
    name: "ocr",
    parameters: &[
        "model",
        "document",
        "api_key",
        "api_base",
        "timeout",
        "custom_llm_provider",
        "extra_headers",
    ],
    required: 2,
};

const ASYNC_SIGNATURE: Signature = Signature {
    name: "aocr",
    ..SIGNATURE
};

struct PythonOcrHost {
    state: PythonCallState,
    projected: Option<ProjectedOcrHost>,
}

struct ProjectedOcrHost {
    model: String,
    provider: &'static str,
    secret_fields: Vec<&'static str>,
    azure_ad_token_provider: Option<PythonTokenProvider>,
    pre_call: Option<callbacks::OcrLoggingFields>,
    payload: Option<CapturedOcrPayload>,
}

struct CapturedOcrPayload {
    body: Py<PyDict>,
    headers: Py<PyDict>,
}

impl CapturedOcrPayload {
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
}

impl PythonOcrHost {
    fn projected(&self) -> PyResult<&ProjectedOcrHost> {
        self.projected.as_ref().ok_or_else(missing_state)
    }

    fn projected_mut(&mut self) -> PyResult<&mut ProjectedOcrHost> {
        self.projected.as_mut().ok_or_else(missing_state)
    }

    fn project(&mut self, py: Python<'_>) -> PyResult<OcrHostResult> {
        let signature = if self.state.asynchronous {
            &ASYNC_SIGNATURE
        } else {
            &SIGNATURE
        };
        let arguments = signature.bind(self.state.args.bind(py), self.state.kwargs.bind(py))?;
        let projected = project(py, &arguments)?;
        let has_token_provider = projected.azure_ad_token_provider.is_some();
        self.projected = Some(ProjectedOcrHost {
            model: projected.request.model.clone(),
            provider: projected.request.provider_name(),
            secret_fields: projected.secret_fields,
            azure_ad_token_provider: projected.azure_ad_token_provider,
            pre_call: None,
            payload: None,
        });
        Ok(OcrHostResult::Request(Ok((
            Box::new(projected.request),
            has_token_provider,
        ))))
    }

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        self.projected()?
            .azure_ad_token_provider
            .as_ref()
            .ok_or_else(missing_state)?
            .acquire(py)
    }

    fn during_call(
        &mut self,
        py: Python<'_>,
        mut request: OcrDuringCallRequest,
    ) -> PyResult<OcrDuringCallRequest> {
        let projected = self.projected()?;
        let pre_call = projected.pre_call.as_ref().ok_or_else(missing_state)?;
        let logger = self.state.logger()?;
        logger.update_ocr(
            py,
            &self.state.kwargs,
            pre_call,
            &projected.secret_fields,
            &request.url,
        )?;
        let body = to_py(py, &request.body)?
            .into_bound(py)
            .cast_into::<PyDict>()?;
        let headers = PyDict::new(py);
        for (name, value) in &request.headers {
            headers.set_item(name, value)?;
        }
        logger.pre_ocr(
            py,
            request.api_key.as_deref(),
            &body,
            &headers,
            &request.url,
        )?;
        request.body = from_py(&body)?;
        request.headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        let projected = self.projected_mut()?;
        projected.payload = Some(CapturedOcrPayload {
            body: body.unbind(),
            headers: headers.unbind(),
        });
        Ok(request)
    }

    fn post_call(
        &mut self,
        py: Python<'_>,
        request: OcrPostCallRequest,
    ) -> PyResult<OcrPostCallRequest> {
        let projected = self.projected()?;
        let payload = projected.payload.as_ref();
        self.state.logger()?.post_ocr(
            py,
            &request.original_response,
            payload.map(|payload| &payload.body),
            payload.map(|payload| &payload.headers),
        )?;
        Ok(request)
    }

    fn map_failure(&mut self, py: Python<'_>, error: litellm_core::ocr::Error) -> PyResult<()> {
        if self.state.error.is_none() {
            self.state.retain_error(py, ocr_error_to_pyerr(error));
        }
        if self.state.end.is_none() {
            self.state.end = Some(now(py)?);
        }
        let error = self.state.error.as_ref().ok_or_else(missing_state)?;
        let (model, provider) = match &self.projected {
            Some(projected) => (projected.model.as_str(), projected.provider),
            None => ("", ""),
        };
        let mapped = callbacks::map_failure(py, error, model, provider, &self.state.kwargs)?;
        self.state
            .retain_error(py, PyErr::from_value(mapped.into_bound(py).into_any()));
        Ok(())
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
            OcrHostOperation::ProjectRequest => self.project(py)?,
            OcrHostOperation::AcquireAzureAdToken => {
                OcrHostResult::AzureAdToken(Ok(self.acquire_azure_ad_token(py)?))
            }
            OcrHostOperation::PreCall(request) => {
                self.projected_mut()?.pre_call = Some((&request).into());
                OcrHostResult::PreCall(Ok(request))
            }
            OcrHostOperation::DuringCall(request) => {
                OcrHostResult::DuringCall(Ok(self.during_call(py, request)?))
            }
            OcrHostOperation::PostCall(request) => {
                OcrHostResult::PostCall(Ok(self.post_call(py, request)?))
            }
            OcrHostOperation::ConstructResponse(response) => {
                self.state.end = Some(now(py)?);
                self.state.response = Some(callbacks::response(py, response.as_ref())?);
                OcrHostResult::Lifecycle(Ok(()))
            }
            OcrHostOperation::MapFailure(error) => {
                self.map_failure(py, error)?;
                OcrHostResult::Lifecycle(Ok(()))
            }
            OcrHostOperation::Lifecycle(_)
            | OcrHostOperation::Success { .. }
            | OcrHostOperation::Failure { .. } => return Err(missing_state()),
        })
    }

    fn cleanup(&mut self) {
        self.projected = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(projected) = &self.projected else {
            return Ok(());
        };
        if let Some(provider) = &projected.azure_ad_token_provider {
            provider.traverse(visit)?;
        }
        if let Some(payload) = &projected.payload {
            payload.traverse(visit)?;
        }
        Ok(())
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
    let signature = if asynchronous {
        &ASYNC_SIGNATURE
    } else {
        &SIGNATURE
    };
    signature.bind(&args, &kwargs)?;
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
            signature.name,
        )?,
        projected: None,
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
