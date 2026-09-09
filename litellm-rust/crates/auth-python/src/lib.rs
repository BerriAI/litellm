use litellm_auth::{
    AuthServiceError, AuthServices, AzureCredentialInputs, CallerCredential, CallerTokenProvider,
    CredentialInputs, SecretString,
};
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};
use std::sync::Mutex;

struct AzureTokenProvider {
    callable: Py<PyAny>,
    error: Mutex<Option<PyErr>>,
}

impl AzureTokenProvider {
    fn new(callable: Py<PyAny>) -> Self {
        Self {
            callable,
            error: Mutex::new(None),
        }
    }

    fn take_error(&self) -> Option<PyErr> {
        self.error.lock().unwrap().take()
    }

    fn retain_error(&self, error: PyErr) -> AuthServiceError {
        *self.error.lock().unwrap() = Some(error);
        AuthServiceError::CallerToken
    }

    fn call(&self, py: Python<'_>) -> PyResult<SecretString> {
        let value = self.callable.bind(py).call0()?;
        if !value.is_instance_of::<pyo3::types::PyString>() {
            return Err(PyTypeError::new_err(format!(
                "Azure AD token must be a string, got {}",
                value.get_type()
            )));
        }
        Ok(SecretString::new(value.extract::<String>()?))
    }
}

impl CallerTokenProvider for AzureTokenProvider {
    fn invoke(&self) -> Result<Option<SecretString>, AuthServiceError> {
        Python::attach(|py| {
            let callable = self.callable.bind(py);
            if !callable
                .is_truthy()
                .map_err(|error| self.retain_error(error))?
                || !callable.is_callable()
            {
                return Ok(None);
            }
            self.call(py).map(Some).map_err(|error| {
                if error.is_instance_of::<PyTypeError>(py)
                    || !error.is_instance_of::<PyException>(py)
                {
                    return self.retain_error(error);
                }
                let message = match error.value(py).str() {
                    Ok(message) => message.to_string(),
                    Err(error) => return self.retain_error(error),
                };
                let wrapped =
                    PyRuntimeError::new_err(format!("Failed to get Azure AD token: {message}"));
                wrapped.set_cause(py, Some(error));
                self.retain_error(wrapped)
            })
        })
    }
}

pub struct PythonAuth {
    inputs: CredentialInputs,
    azure_token_provider: Option<AzureTokenProvider>,
}

impl PythonAuth {
    pub fn from_arguments(py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<Self> {
        let azure_token_provider = arguments
            .get_item("azure_ad_token_provider")?
            .filter(|value| !value.is_none())
            .map(|value| AzureTokenProvider::new(value.unbind()));
        let inputs = CredentialInputs {
            azure: AzureCredentialInputs {
                token: optional_string(arguments, "azure_ad_token")?,
                has_token_provider: azure_token_provider.is_some(),
                tenant_id: optional_string(arguments, "tenant_id")?,
                client_id: optional_string(arguments, "client_id")?,
                has_client_secret: optional_string(arguments, "client_secret")?
                    .is_some_and(|value| !value.is_empty()),
                has_username: optional_string(arguments, "azure_username")?
                    .is_some_and(|value| !value.is_empty()),
                has_password: optional_string(arguments, "azure_password")?
                    .is_some_and(|value| !value.is_empty()),
                refresh: py
                    .import("litellm")?
                    .getattr("enable_azure_ad_token_refresh")?
                    .is(&PyBool::new(py, true)),
            },
        };
        Ok(Self {
            inputs,
            azure_token_provider,
        })
    }

    pub fn inputs(&self) -> &CredentialInputs {
        &self.inputs
    }

    pub fn take_error(&self) -> Option<PyErr> {
        self.azure_token_provider
            .as_ref()
            .and_then(AzureTokenProvider::take_error)
    }
}

impl AuthServices for PythonAuth {
    fn token_provider(&self, credential: CallerCredential) -> Option<&dyn CallerTokenProvider> {
        match credential {
            CallerCredential::AzureAdToken => self
                .azure_token_provider
                .as_ref()
                .map(|provider| provider as &dyn CallerTokenProvider),
        }
    }
}

fn optional_string(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    arguments
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_callable_is_invoked_again_without_caching() {
        Python::initialize();
        Python::attach(|py| {
            let provider = AzureTokenProvider::new(
                py.eval(
                    c"lambda tokens=iter(('first', 'second')): next(tokens)",
                    None,
                    None,
                )
                .unwrap()
                .unbind(),
            );
            assert_eq!(provider.invoke().unwrap().unwrap().expose(), "first");
            assert_eq!(provider.invoke().unwrap().unwrap().expose(), "second");
            assert!(provider.take_error().is_none());
        });
    }

    #[test]
    fn invalid_return_and_caller_errors_remain_distinguishable() {
        Python::initialize();
        Python::attach(|py| {
            let provider =
                AzureTokenProvider::new(py.eval(c"lambda: 123", None, None).unwrap().unbind());
            assert_eq!(provider.invoke(), Err(AuthServiceError::CallerToken));
            assert!(
                provider
                    .take_error()
                    .unwrap()
                    .is_instance_of::<PyTypeError>(py)
            );

            let error = pyo3::exceptions::PyValueError::new_err("unavailable");
            let globals = PyDict::new(py);
            globals.set_item("error", error.value(py)).unwrap();
            let provider = AzureTokenProvider::new(
                py.eval(
                    c"lambda: (_ for _ in ()).throw(error)",
                    Some(&globals),
                    None,
                )
                .unwrap()
                .unbind(),
            );
            assert_eq!(provider.invoke(), Err(AuthServiceError::CallerToken));
            let wrapped = provider.take_error().unwrap();
            assert!(wrapped.is_instance_of::<PyRuntimeError>(py));
            assert!(wrapped.cause(py).unwrap().value(py).is(error.value(py)));
            assert!(provider.take_error().is_none());
        });
    }

    #[test]
    fn inactive_binding_is_distinct_from_a_callback_returning_none() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(AzureTokenProvider::new(py.None()).invoke(), Ok(None));
            let provider =
                AzureTokenProvider::new(py.eval(c"lambda: None", None, None).unwrap().unbind());
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
