use std::sync::Mutex;

use litellm_auth::{AuthServiceError, CallerTokenProvider, SecretString};
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::Error;

#[derive(Clone, Copy)]
pub struct SecretProviderContract {
    pub value_name: &'static str,
    pub failure_context: &'static str,
}

pub struct PythonSecretProvider {
    callable: Py<PyAny>,
    contract: SecretProviderContract,
    error: Mutex<Option<PyErr>>,
}

impl PythonSecretProvider {
    pub fn new(callable: Py<PyAny>, contract: SecretProviderContract) -> Self {
        Self {
            callable,
            contract,
            error: Mutex::new(None),
        }
    }

    pub fn take_error(&self) -> Option<PyErr> {
        self.error.lock().unwrap().take()
    }

    fn retain_error(&self, error: Error, py: Python<'_>) -> AuthServiceError {
        *self.error.lock().unwrap() = Some(error.into_pyerr(py));
        AuthServiceError::CallerToken
    }

    fn call(&self, py: Python<'_>) -> Result<SecretString, Error> {
        let value = self.callable.bind(py).call0().map_err(|error| {
            if error.is_instance_of::<PyTypeError>(py) || !error.is_instance_of::<PyException>(py) {
                return Error::Python(error);
            }
            match error.value(py).str() {
                Ok(message) => Error::Callback {
                    failure_context: self.contract.failure_context,
                    message: message.to_string(),
                    source: error,
                },
                Err(error) => Error::Python(error),
            }
        })?;
        if !value.is_instance_of::<PyString>() {
            return Err(Error::InvalidReturn {
                value_name: self.contract.value_name,
                actual_type: value.get_type().to_string(),
            });
        }
        Ok(SecretString::new(value.extract::<String>()?))
    }
}

impl CallerTokenProvider for PythonSecretProvider {
    fn invoke(&self) -> Result<Option<SecretString>, AuthServiceError> {
        Python::attach(|py| {
            let callable = self.callable.bind(py);
            if !callable
                .is_truthy()
                .map_err(|error| self.retain_error(Error::Python(error), py))?
                || !callable.is_callable()
            {
                return Ok(None);
            }
            self.call(py)
                .map(Some)
                .map_err(|error| self.retain_error(error, py))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTRACT: SecretProviderContract = SecretProviderContract {
        value_name: "access credential",
        failure_context: "Credential callback failed",
    };

    #[test]
    fn retained_callable_is_invoked_again_without_caching() {
        Python::initialize();
        Python::attach(|py| {
            let provider = PythonSecretProvider::new(
                py.eval(
                    c"lambda values=iter(('first', 'second')): next(values)",
                    None,
                    None,
                )
                .unwrap()
                .unbind(),
                CONTRACT,
            );
            assert_eq!(provider.invoke().unwrap().unwrap().expose(), "first");
            assert_eq!(provider.invoke().unwrap().unwrap().expose(), "second");
            assert!(provider.take_error().is_none());
        });
    }

    #[test]
    fn contract_controls_errors_without_provider_specific_logic() {
        Python::initialize();
        Python::attach(|py| {
            let invalid = PythonSecretProvider::new(
                py.eval(c"lambda: 123", None, None).unwrap().unbind(),
                CONTRACT,
            );
            assert_eq!(invalid.invoke(), Err(AuthServiceError::CallerToken));
            assert_eq!(
                invalid.take_error().unwrap().value(py).to_string(),
                "access credential must be a string, got <class 'int'>"
            );

            let globals = pyo3::types::PyDict::new(py);
            let source = pyo3::exceptions::PyValueError::new_err("unavailable");
            globals.set_item("error", source.value(py)).unwrap();
            let failed = PythonSecretProvider::new(
                py.eval(
                    c"lambda: (_ for _ in ()).throw(error)",
                    Some(&globals),
                    None,
                )
                .unwrap()
                .unbind(),
                CONTRACT,
            );
            assert_eq!(failed.invoke(), Err(AuthServiceError::CallerToken));
            let wrapped = failed.take_error().unwrap();
            assert_eq!(
                wrapped.value(py).to_string(),
                "Credential callback failed: unavailable"
            );
            assert!(wrapped.cause(py).unwrap().value(py).is(source.value(py)));
        });
    }

    #[test]
    fn inactive_binding_differs_from_a_callback_returning_none() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                PythonSecretProvider::new(py.None(), CONTRACT).invoke(),
                Ok(None)
            );
            let provider = PythonSecretProvider::new(
                py.eval(c"lambda: None", None, None).unwrap().unbind(),
                CONTRACT,
            );
            assert_eq!(provider.invoke(), Err(AuthServiceError::CallerToken));
            assert!(
                provider
                    .take_error()
                    .unwrap()
                    .is_instance_of::<PyTypeError>(py)
            );
        });
    }
}
