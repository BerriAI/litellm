use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError, PyTypeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use serde_json::Value;

use litellm_core::auth::{ResolvedCredential, SecretValue};
use litellm_core::ocr::LiteLLMOcrResponse;
use litellm_core::ocr::hooks::OcrPreCallRequest;
use litellm_python_interop::to_py_preserving_errors as to_py;

use crate::lifecycle::PythonLogger;

pub(super) struct AzureAdTokenProvider(Py<PyAny>);

impl AzureAdTokenProvider {
    pub(super) fn select(provider: Bound<'_, PyAny>) -> Option<Self> {
        (provider.is_callable() && provider.is_truthy().unwrap_or(false))
            .then(|| Self(provider.unbind()))
    }

    pub(super) fn acquire(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        let provider = self.0.bind(py);
        if !provider.is_callable() {
            return Err(PyTypeError::new_err(
                "Azure AD token provider must be callable",
            ));
        }
        let token = (|| {
            let token = provider.call0()?;
            if !token.is_instance_of::<PyString>() {
                let message = PyString::new(py, "Azure AD token must be a string, got {}")
                    .call_method1("format", (token.get_type(),))?;
                return Err(PyTypeError::new_err(message.unbind()));
            }
            Ok(token)
        })()
        .map_err(|error| {
            if error.is_instance_of::<PyTypeError>(py) || !error.is_instance_of::<PyException>(py) {
                return error;
            }
            match PyString::new(py, "Failed to get Azure AD token: {}")
                .call_method1("format", (error.value(py),))
            {
                Ok(message) => {
                    let wrapped = PyRuntimeError::new_err(message.unbind());
                    wrapped.set_context(py, Some(error.clone_ref(py)));
                    wrapped.set_cause(py, Some(error));
                    wrapped
                }
                Err(format_error) => {
                    format_error.set_context(py, Some(error));
                    format_error
                }
            }
        })?;
        Ok(ResolvedCredential::AccessToken {
            token: SecretValue::new(token.extract::<String>()?),
            expires_on: None,
        })
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}

impl PythonLogger {
    pub(crate) fn update_ocr(
        &self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        pre_call: &OcrPreCallRequest,
        url: &str,
    ) -> PyResult<()> {
        let redact = py
            .import("litellm.rust_bridge.ocr")?
            .getattr("redact_logging_params")?;
        let update = PyDict::new(py);
        update.set_item("kwargs", redact.call1((kwargs,))?.cast_into::<PyDict>()?)?;
        update.set_item("model", &pre_call.model)?;
        update.set_item(
            "optional_params",
            redact
                .call1((to_py(py, &pre_call.optional_params)?,))?
                .cast_into::<PyDict>()?,
        )?;
        let params = PyDict::new(py);
        params.set_item(
            "litellm_call_id",
            kwargs.bind(py).get_item("litellm_call_id")?,
        )?;
        params.set_item("api_base", url)?;
        update.set_item("litellm_params", params)?;
        update.set_item("custom_llm_provider", &pre_call.custom_llm_provider)?;
        self.object(py)
            .call_method("update_from_kwargs", (), Some(&update))?;
        Ok(())
    }

    pub(crate) fn pre_ocr(
        &self,
        py: Python<'_>,
        api_key: &Option<Py<PyAny>>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        additional.set_item("api_base", url)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", "OCR document processing")?;
        kwargs.set_item("api_key", api_key)?;
        kwargs.set_item("additional_args", additional)?;
        self.object(py).call_method("pre_call", (), Some(&kwargs))?;
        Ok(())
    }

    pub(crate) fn post_ocr(
        &self,
        py: Python<'_>,
        original_response: &Value,
        body: &Option<Py<PyDict>>,
        headers: &Option<Py<PyDict>>,
    ) -> PyResult<()> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("original_response", to_py(py, original_response)?)?;
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        kwargs.set_item("additional_args", additional)?;
        self.object(py)
            .call_method("post_call", (), Some(&kwargs))?;
        Ok(())
    }
}

pub(super) fn response(py: Python<'_>, response: &LiteLLMOcrResponse) -> PyResult<Py<PyAny>> {
    py.import("litellm.rust_bridge.ocr")?
        .getattr("_response")?
        .call1((to_py(py, response)?,))
        .map(Bound::unbind)
}

pub(super) fn map_failure(
    py: Python<'_>,
    error: &Py<PyBaseException>,
    request: &Bound<'_, PyAny>,
    provider: &str,
) -> PyResult<Py<PyBaseException>> {
    Ok(py
        .import("litellm.rust_bridge.ocr_lifecycle")?
        .getattr("map_failure")?
        .call1((error, request, provider))?
        .extract()?)
}

pub(super) fn timeout_seconds(py: Python<'_>, timeout: Py<PyAny>) -> PyResult<Option<f64>> {
    py.import("litellm.rust_bridge.timeouts")?
        .getattr("timeout_to_seconds")?
        .call1((timeout,))?
        .extract()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_callback_preserves_exception_identity_and_explicit_chaining() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class ProviderError(Exception):
    def __format__(self, specification):
        return 'unavailable'
ordinary = ProviderError('must use __format__')
type_error = TypeError('signature')
abort = KeyboardInterrupt('cancelled')
def provider(error):
    def acquire():
        raise error
    return acquire
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            for name in ["ordinary", "type_error", "abort"] {
                let original = locals.get_item(name).unwrap().unwrap();
                let callback = locals
                    .get_item("provider")
                    .unwrap()
                    .unwrap()
                    .call1((&original,))
                    .unwrap();
                let provider = AzureAdTokenProvider::select(callback).unwrap();
                let error = provider.acquire(py).unwrap_err();
                if name == "ordinary" {
                    assert!(error.is_instance_of::<PyRuntimeError>(py));
                    assert!(error.cause(py).unwrap().value(py).is(&original));
                    assert!(
                        error
                            .value(py)
                            .getattr("__context__")
                            .unwrap()
                            .is(&original)
                    );
                    assert_eq!(
                        error.value(py).str().unwrap().to_str().unwrap(),
                        "Failed to get Azure AD token: unavailable"
                    );
                } else {
                    assert!(error.value(py).is(&original));
                }
            }
        });
    }

    #[test]
    fn invalid_token_type_formatting_preserves_python_failure_semantics() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
failure = ValueError('formatting failed')
class TokenType(type):
    def __format__(cls, specification):
        raise failure
class Token(metaclass=TokenType):
    pass
def provider():
    return Token()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let provider =
                AzureAdTokenProvider::select(locals.get_item("provider").unwrap().unwrap())
                    .unwrap();
            let error = provider.acquire(py).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert!(
                error
                    .cause(py)
                    .unwrap()
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn token_string_extraction_errors_are_not_wrapped_as_callback_failures() {
        Python::initialize();
        Python::attach(|py| {
            let callback = py
                .eval(pyo3::ffi::c_str!("lambda: '\\ud800'"), None, None)
                .unwrap();
            let provider = AzureAdTokenProvider::select(callback).unwrap();
            let error = provider.acquire(py).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyUnicodeEncodeError>(py));
        });
    }
}
