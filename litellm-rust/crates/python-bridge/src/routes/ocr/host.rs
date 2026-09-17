use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use litellm_auth::ResolvedCredential;
use litellm_core::ocr::hooks::{OcrDuringCallRequest, OcrPostCallRequest};
use litellm_core::ocr::{OcrCall, OcrHostOperation, OcrHostResult};
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};

use super::document::PythonFileReader;
use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::project::project;
use super::{ASYNC_SIGNATURE, SIGNATURE, callbacks, errors};
use crate::auth::PythonTokenProvider;
use crate::lifecycle::{OperationClass, PythonCallState, PythonRoute, missing_state, now};

pub(super) struct PythonOcrHost {
    state: PythonCallState,
    projected: Option<ProjectedOcrHost>,
}

struct ProjectedOcrHost {
    model: String,
    provider: &'static str,
    secret_fields: Vec<&'static str>,
    azure_ad_token_provider: Option<PythonTokenProvider>,
    pre_call: Option<callbacks::OcrLoggingFields>,
    reader: Option<PythonFileReader>,
    reader_failed: bool,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyDict>>,
}

impl PythonOcrHost {
    pub(super) fn new(state: PythonCallState) -> Self {
        Self {
            state,
            projected: None,
        }
    }

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
            reader: projected.reader,
            reader_failed: false,
            body: None,
            headers: None,
        });
        Ok(OcrHostResult::Request(Ok((
            Box::new(
                projected
                    .request
                    .with_host_hooks(Arc::new(BridgeOcrHooks), None),
            ),
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
        logger.pre_ocr(py, request.api_key.as_deref(), &body, &headers, &request.url)?;
        request.body = from_py(&body)?;
        request.headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        let projected = self.projected_mut()?;
        projected.body = Some(body.unbind());
        projected.headers = Some(headers.unbind());
        Ok(request)
    }

    fn post_call(
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

    fn map_failure(&mut self, py: Python<'_>, error: litellm_core::ocr::Error) -> PyResult<()> {
        if self.state.end.is_none() {
            self.state.end = Some(now(py)?);
        }
        let (model, provider) = match &self.projected {
            Some(projected) => (projected.model.as_str(), projected.provider),
            None => ("", ""),
        };
        let mapped = match self.state.error.take() {
            Some(host_error) if self.projected.as_ref().is_some_and(|host| host.reader_failed) => {
                PyErr::from_value(host_error.into_bound(py).into_any())
            }
            Some(host_error) => errors::public_host_exception(py, &host_error, model, provider)?,
            None => errors::public_exception(py, error, model, provider)?,
        };
        self.state.retain_error(py, mapped);
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
            OcrHostOperation::ReadDocument => {
                let reader = self.projected_mut()?.reader.take().ok_or_else(missing_state)?;
                let result = reader.read(py);
                self.projected_mut()?.reader_failed = result.is_err();
                OcrHostResult::Document(Ok(result?))
            }
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

    fn traverse(&self, visit: &pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        let Some(projected) = &self.projected else {
            return Ok(());
        };
        if let Some(provider) = &projected.azure_ad_token_provider {
            provider.traverse(visit)?;
        }
        if let Some(reader) = &projected.reader {
            reader.traverse(visit)?;
        }
        visit.call(&projected.body)?;
        visit.call(&projected.headers)
    }
}

struct BridgeOcrHooks;

impl litellm_core::ocr::hooks::OcrHooks for BridgeOcrHooks {
    fn intercepts_requests(&self) -> bool {
        true
    }
}
