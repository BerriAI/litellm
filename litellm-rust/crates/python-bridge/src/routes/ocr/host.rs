use litellm_auth::ResolvedCredential;
use litellm_core::ocr::{LiteLLMOcrResponse, Ocr, OcrOp, OcrOpResult};
use litellm_host_python::{RouteHost, missing_state, to_py};
use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::project::{OcrHostHandles, project_request};
use crate::errors::ocr_error_to_pyerr;

enum OcrHostData {
    Unprojected,
    Projected(Box<OcrHostHandles>),
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

    fn handles(&self) -> PyResult<&OcrHostHandles> {
        match &self.data {
            OcrHostData::Projected(handles) => Ok(handles),
            _ => Err(missing_state()),
        }
    }

    fn read_document(&self, py: Python<'_>) -> PyResult<litellm_core::ocr::OcrFileContent> {
        self.handles()?
            .reader
            .as_ref()
            .ok_or_else(missing_state)?
            .read(py)
    }

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        self.handles()?
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
                let (request, handles) = project_request(self.request.bind(py), arguments)?;
                let caller_token = handles.azure_ad_token_provider.is_some();
                self.data = OcrHostData::Projected(Box::new(handles));
                Ok(OcrOpResult::Request {
                    request: Box::new(request),
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
        py.import("litellm.rust_bridge.ocr.route_host")?
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
            OcrHostData::Projected(handles) => handles.provider,
            _ => "",
        };
        let mapped: Py<PyBaseException> = py
            .import("litellm.rust_bridge.ocr.route_host")?
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
        if let OcrHostData::Projected(handles) = &self.data {
            if let Some(reader) = &handles.reader {
                reader.traverse(visit)?;
            }
            if let Some(provider) = &handles.azure_ad_token_provider {
                provider.traverse(visit)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::acquired(true)]
    #[case::provider_raised(false)]
    fn closing_releases_the_token_provider(#[case] succeeds: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("succeeds", succeeds).unwrap();
            py.run(
                c"
import gc
import weakref
class Provider:
    def __call__(self):
        if succeeds:
            return 'caller-token'
        raise ValueError('unavailable')
provider = Provider()
reference = weakref.ref(provider)
kwargs = {
    'model': 'azure_ai/mistral-ocr-latest',
    'custom_llm_provider': None,
    'document': {'type': 'document_url', 'document_url': 'https://example.com/a.pdf'},
    'api_key': None,
    'api_base': None,
    'extra_headers': None,
    'timeout': None,
    'azure_ad_token_provider': provider,
}
del provider
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let mut host = OcrRouteHost::new(py.None());
            let projected = host.invoke(py, &kwargs, OcrOp::ProjectRequest).unwrap();
            assert!(matches!(
                projected,
                OcrOpResult::Request {
                    caller_token: true,
                    ..
                }
            ));
            locals.del_item("kwargs").unwrap();
            drop(kwargs);
            assert_eq!(
                host.invoke(py, &PyDict::new(py), OcrOp::AcquireAzureAdToken)
                    .is_ok(),
                succeeds
            );
            let alive = || {
                py.run(c"gc.collect()", Some(&locals), Some(&locals))
                    .unwrap();
                !py.eval(c"reference()", Some(&locals), Some(&locals))
                    .unwrap()
                    .is_none()
            };
            assert!(alive());
            host.close(py);
            assert!(!alive());
        });
    }
}
