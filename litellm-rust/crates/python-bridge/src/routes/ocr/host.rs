use litellm_auth::ResolvedCredential;
use litellm_core::ocr::{LiteLLMOcrResponse, Ocr, OcrOp, OcrOpResult};
use litellm_host_python::{RouteHost, missing_state, to_py};
use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::project::{OcrRetained, project_request};

enum OcrHostData {
    Unprojected,
    Projected(Box<OcrRetained>),
    Released,
}

/// The Python side of the OCR route: projects the prepared arguments, reads file-like
/// documents, acquires Azure AD tokens, and builds the public response and exception.
pub(super) struct OcrRouteHost {
    request: Py<PyAny>,
    data: OcrHostData,
}

impl OcrRouteHost {
    pub(super) fn new(request: Py<PyAny>) -> Self {
        Self {
            request,
            data: OcrHostData::Unprojected,
        }
    }

    fn projected(&self) -> PyResult<&OcrRetained> {
        match &self.data {
            OcrHostData::Projected(projected) => Ok(projected),
            _ => Err(missing_state()),
        }
    }

    fn read_document(&self, py: Python<'_>) -> PyResult<litellm_core::ocr::OcrFileContent> {
        self.projected()?
            .reader
            .as_ref()
            .ok_or_else(missing_state)?
            .read(py)
    }

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        self.projected()?
            .azure_ad_token_provider
            .as_ref()
            .ok_or_else(missing_state)?
            .acquire(py)
    }
}

impl RouteHost for OcrRouteHost {
    type Route = Ocr;

    fn invoke(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        op: OcrOp,
    ) -> PyResult<OcrOpResult> {
        match op {
            OcrOp::ProjectRequest => {
                let OcrHostData::Unprojected = self.data else {
                    return Err(missing_state());
                };
                let projection = project_request(self.request.bind(py), arguments)?;
                let caller_token = projection.retained.azure_ad_token_provider.is_some();
                self.data = OcrHostData::Projected(Box::new(projection.retained));
                Ok(OcrOpResult::Request {
                    request: Box::new(projection.native),
                    caller_token,
                })
            }
            OcrOp::ReadDocument => self.read_document(py).map(OcrOpResult::Document),
            OcrOp::AcquireAzureAdToken => self
                .acquire_azure_ad_token(py)
                .map(OcrOpResult::AzureAdToken),
        }
    }

    fn complete(&mut self, py: Python<'_>, response: LiteLLMOcrResponse) -> PyResult<Py<PyAny>> {
        py.import("litellm.rust_bridge.ocr.callbacks")?
            .getattr("response")?
            .call1((to_py(py, &response)?,))
            .map(Bound::unbind)
    }

    fn native_error(error: litellm_core::ocr::Error) -> PyErr {
        ocr_error_to_pyerr(error)
    }

    fn host_error(error: &PyErr) -> litellm_core::ocr::Error {
        litellm_core::ocr::Error::InvalidRequest(error.to_string())
    }

    fn map_failure(&self, py: Python<'_>, error: &PyErr) -> PyResult<PyErr> {
        let provider = match &self.data {
            OcrHostData::Projected(projected) => projected.provider,
            _ => "",
        };
        let mapped: Py<PyBaseException> = py
            .import("litellm.rust_bridge.ocr.callbacks")?
            .getattr("map_failure")?
            .call1((error.value(py), self.request.bind(py), provider))?
            .extract()?;
        Ok(PyErr::from_value(mapped.into_bound(py).into_any()))
    }

    fn close(&mut self, _: Python<'_>) {
        self.data = OcrHostData::Released;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)?;
        if let OcrHostData::Projected(projected) = &self.data {
            if let Some(reader) = &projected.reader {
                reader.traverse(visit)?;
            }
            if let Some(provider) = &projected.azure_ad_token_provider {
                provider.traverse(visit)?;
            }
        }
        Ok(())
    }
}
