use litellm_core::auth::{ResolvedCredential, SecretValue};
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyString;

#[derive(Clone, Copy)]
pub(crate) struct TokenProviderContract {
    callable_error: &'static str,
    token_type_error: &'static str,
    callback_error: &'static str,
}

pub(crate) const AZURE_AD_TOKEN_PROVIDER: TokenProviderContract = TokenProviderContract {
    callable_error: "Azure AD token provider must be callable",
    token_type_error: "Azure AD token must be a string, got {}",
    callback_error: "Failed to get Azure AD token: {}",
};

pub(crate) struct PythonTokenProvider {
    callback: Py<PyAny>,
    contract: TokenProviderContract,
}

impl PythonTokenProvider {
    pub(crate) fn select(
        provider: Bound<'_, PyAny>,
        contract: TokenProviderContract,
    ) -> Option<Self> {
        (provider.is_callable() && provider.is_truthy().unwrap_or(false)).then(|| Self {
            callback: provider.unbind(),
            contract,
        })
    }

    pub(crate) fn acquire(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        let provider = self.callback.bind(py);
        if !provider.is_callable() {
            return Err(PyTypeError::new_err(self.contract.callable_error));
        }
        let token = (|| {
            let token = provider.call0()?;
            if !token.is_instance_of::<PyString>() {
                let message = PyString::new(py, self.contract.token_type_error)
                    .call_method1("format", (token.get_type(),))?;
                return Err(PyTypeError::new_err(message.unbind()));
            }
            Ok(token)
        })()
        .map_err(|error| {
            if error.is_instance_of::<PyTypeError>(py) || !error.is_instance_of::<PyException>(py) {
                return error;
            }
            match PyString::new(py, self.contract.callback_error)
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

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.callback)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::PyDict;

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
                let provider =
                    PythonTokenProvider::select(callback, AZURE_AD_TOKEN_PROVIDER).unwrap();
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
            let provider = PythonTokenProvider::select(
                locals.get_item("provider").unwrap().unwrap(),
                AZURE_AD_TOKEN_PROVIDER,
            )
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
            let provider = PythonTokenProvider::select(callback, AZURE_AD_TOKEN_PROVIDER).unwrap();
            let error = provider.acquire(py).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyUnicodeEncodeError>(py));
        });
    }
}
