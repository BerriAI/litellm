use litellm_auth::ResolvedCredential;
use litellm_core::ocr::route::{Ocr, OcrOp, OcrProjection};
use litellm_host_python::{InvokeError, ProtocolHost, missing_state, to_py};
use litellm_llms::base_llm::ocr::{error::Error, transformation::LiteLLMOcrResponse};
use pyo3::{
    exceptions::{PyBaseException, PyException},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use super::{
    errors::to_pyerr as ocr_error_to_pyerr,
    project::{OcrHostHandles, project_request},
};

enum OcrHostData {
    Unprojected,
    Projected(Box<OcrHostHandles>),
    Released,
}

/// The Python side of the OCR route: projects the prepared arguments (reading a file-like
/// document as it goes), acquires Azure AD tokens, and builds the public response and
/// exception.
pub(super) struct OcrPythonHost {
    request: Py<PyAny>,
    data: OcrHostData,
}

impl OcrPythonHost {
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

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        self.handles()?
            .azure_ad_token_provider
            .as_ref()
            .ok_or_else(missing_state)?
            .acquire(py)
    }

    fn projection(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<OcrProjection> {
        let OcrHostData::Unprojected = self.data else {
            return Err(missing_state());
        };
        let (request, handles) = project_request(self.request.bind(py), arguments)?;
        let caller_token = handles.azure_ad_token_provider.is_some();
        self.data = OcrHostData::Projected(Box::new(handles));
        Ok(OcrProjection {
            request,
            caller_token,
        })
    }

    fn map_failure(&self, py: Python<'_>, error: PyErr) -> PyErr {
        if !error.is_instance_of::<PyException>(py) {
            return error;
        }
        let provider = match &self.data {
            OcrHostData::Projected(handles) => handles.provider,
            _ => "",
        };
        let mapped = py
            .import("litellm.rust_bridge.ocr.route_host")
            .and_then(|module| module.getattr("map_failure"))
            .and_then(|map| map.call1((error.value(py), self.request.bind(py), provider)))
            .and_then(|mapped| mapped.extract::<Py<PyBaseException>>().map_err(PyErr::from));
        match mapped {
            Ok(mapped) => PyErr::from_value(mapped.into_bound(py).into_any()),
            Err(_) => error,
        }
    }
}

impl ProtocolHost for OcrPythonHost {
    type Protocol = Ocr;
    type Failure = PyErr;

    fn project(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<OcrProjection, InvokeError<Error>> {
        self.projection(py, arguments)
            .map_err(|error| InvokeError::Python(self.map_failure(py, error)))
    }

    fn invoke(&mut self, py: Python<'_>, op: OcrOp) -> Result<(), InvokeError<Error>> {
        match op {
            OcrOp::AcquireAzureAdToken(reply) => self
                .acquire_azure_ad_token(py)
                .map(|token| reply.send(token))
                .map_err(|error| InvokeError::Python(self.map_failure(py, error))),
        }
    }

    fn complete(&mut self, py: Python<'_>, response: LiteLLMOcrResponse) -> PyResult<Py<PyAny>> {
        py.import("litellm.rust_bridge.ocr.route_host")?
            .getattr("response")?
            .call1((to_py(py, &response)?,))
            .map(Bound::unbind)
    }

    fn chunk(&mut self, _: Python<'_>, chunk: std::convert::Infallible) -> PyResult<Py<PyAny>> {
        match chunk {}
    }

    fn classify(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        if let Error::Secret(source) = &error
            && let Some(original) = crate::secrets::python_error(py, source)
        {
            return Ok(original);
        }
        Ok(self.map_failure(py, ocr_error_to_pyerr(error)))
    }

    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string())
    }

    fn close(&mut self, _: Python<'_>) {
        self.data = OcrHostData::Released;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)?;
        if let OcrHostData::Projected(handles) = &self.data
            && let Some(provider) = &handles.azure_ad_token_provider
        {
            provider.traverse(visit)?;
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
            let mut host = OcrPythonHost::new(py.None());
            assert!(host.project(py, &kwargs).unwrap().caller_token);
            locals.del_item("kwargs").unwrap();
            drop(kwargs);
            let (reply, _) = litellm_host::host::reply();
            assert_eq!(
                host.invoke(py, OcrOp::AcquireAzureAdToken(reply)).is_ok(),
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
