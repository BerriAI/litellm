use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use litellm_auth::ResolvedCredential;
use litellm_core::call_lifecycle::host::HostPhase;
use litellm_core::ocr::hooks::{OcrDuringCallRequest, OcrPostCallRequest};
use litellm_core::ocr::{OcrCall, OcrHostOperation, OcrHostResult, OcrProjectedRequest};
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};

use super::document::PythonFileReader;
use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::project::project;
use super::{callbacks, errors};
use crate::auth::PythonTokenProvider;
use crate::lifecycle::{PythonCallState, PythonRoute, Signature, missing_state};
use crate::marshal::Projection;

pub(super) struct PythonOcrHost {
    state: PythonCallState,
    signature: &'static Signature,
    retained: Option<OcrRetained>,
}

pub(super) struct OcrRetained {
    pub model: String,
    pub provider: &'static str,
    pub secret_fields: Vec<&'static str>,
    pub azure_ad_token_provider: Option<PythonTokenProvider>,
    pub payload: Option<PythonPayload>,
    pub reader: Option<PythonFileReader>,
}

pub(super) struct PythonPayload {
    pub body: Py<PyDict>,
    pub headers: Py<PyDict>,
}

impl PythonPayload {
    fn from_request(py: Python<'_>, request: &OcrDuringCallRequest) -> PyResult<Self> {
        let body = to_py(py, &request.body)?.into_bound(py).cast_into::<PyDict>()?;
        let headers = PyDict::new(py);
        for (name, value) in &request.headers {
            headers.set_item(name, value)?;
        }
        Ok(Self { body: body.unbind(), headers: headers.unbind() })
    }

    fn write_back(&self, py: Python<'_>, mut request: OcrDuringCallRequest) -> PyResult<OcrDuringCallRequest> {
        request.body = from_py(self.body.bind(py))?;
        request.headers = self.headers.bind(py)
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(request)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
}

impl PythonOcrHost {
    pub(super) fn new(state: PythonCallState, signature: &'static Signature) -> Self {
        Self {
            state,
            signature,
            retained: None,
        }
    }

    fn retained(&self) -> PyResult<&OcrRetained> {
        self.retained.as_ref().ok_or_else(missing_state)
    }

    fn retained_mut(&mut self) -> PyResult<&mut OcrRetained> {
        self.retained.as_mut().ok_or_else(missing_state)
    }

    fn project(&mut self, py: Python<'_>) -> PyResult<OcrHostResult> {
        let arguments = self.signature.bind(self.state.args.bind(py), self.state.kwargs.bind(py))?;
        let Projection { native, retained } = project(py, &arguments)?;
        let host_token_provider = retained.azure_ad_token_provider.is_some();
        self.retained = Some(retained);
        Ok(OcrHostResult::Request(Ok(OcrProjectedRequest {
            request: Box::new(native),
            intercepts_requests: true,
            host_token_provider,
        })))
    }

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        self.retained()?
            .azure_ad_token_provider
            .as_ref()
            .ok_or_else(missing_state)?
            .acquire(py)
    }

    fn during_call(
        &mut self,
        py: Python<'_>,
        request: OcrDuringCallRequest,
    ) -> PyResult<OcrDuringCallRequest> {
        let retained = self.retained()?;
        let logger = self.state.logger()?;
        callbacks::update_logging(
            py,
            logger,
            &self.state.kwargs,
            &request,
            &retained.secret_fields,
        )?;
        let payload = PythonPayload::from_request(py, &request)?;
        callbacks::pre_call(py, logger, &request, &payload)?;
        let request = payload.write_back(py, request)?;
        self.retained_mut()?.payload = Some(payload);
        Ok(request)
    }

    fn post_call(
        &mut self,
        py: Python<'_>,
        request: OcrPostCallRequest,
    ) -> PyResult<OcrPostCallRequest> {
        let payload = self.retained()?.payload.as_ref().ok_or_else(missing_state)?;
        callbacks::post_call(
            py,
            self.state.logger()?,
            &request.original_response,
            payload,
        )?;
        Ok(request)
    }

    fn map_failure(&mut self, py: Python<'_>, error: litellm_core::ocr::Error) -> PyResult<()> {
        self.state.finish(py)?;
        let (model, provider) = match &self.retained {
            Some(retained) => (retained.model.as_str(), retained.provider),
            None => ("", ""),
        };
        let mapped = match self.state.error.take() {
            Some(host_error) if matches!(error, litellm_core::ocr::Error::HostDocumentRead) => {
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

    fn phase(operation: &OcrHostOperation) -> Option<HostPhase> {
        operation.phase()
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
                let reader = self
                    .retained_mut()?
                    .reader
                    .take()
                    .ok_or_else(missing_state)?;
                match reader.read(py) {
                    Ok(content) => OcrHostResult::Document(Ok(content)),
                    Err(error) if error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                        self.state.retain_first_error(py, error, false);
                        OcrHostResult::Document(Err(litellm_core::ocr::Error::HostDocumentRead))
                    }
                    Err(error) => return Err(error),
                }
            }
            OcrHostOperation::AcquireAzureAdToken => {
                OcrHostResult::AzureAdToken(Ok(self.acquire_azure_ad_token(py)?))
            }
            OcrHostOperation::PreCall(request) => {
                OcrHostResult::PreCall(Ok(request))
            }
            OcrHostOperation::DuringCall(request) => {
                OcrHostResult::DuringCall(Ok(self.during_call(py, request)?))
            }
            OcrHostOperation::PostCall(request) => {
                OcrHostResult::PostCall(Ok(self.post_call(py, request)?))
            }
            OcrHostOperation::ConstructResponse(response) => {
                self.state.finish(py)?;
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
        self.retained = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(retained) = &self.retained else {
            return Ok(());
        };
        if let Some(provider) = &retained.azure_ad_token_provider {
            provider.traverse(visit)?;
        }
        if let Some(reader) = &retained.reader {
            reader.traverse(visit)?;
        }
        if let Some(payload) = &retained.payload {
            payload.traverse(visit)?;
        }
        Ok(())
    }
}
