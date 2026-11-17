//! Credentials the caller supplies as Python callables, projected out of a route's
//! keyword arguments and acquired on the host's own thread when the call asks for one.

use crate::callable::wrap_failure;
use litellm_auth::ResolvedCredential;
use litellm_secret_value::SecretValue;

use pyo3::{
    exceptions::{PyException, PyTypeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyString},
};

const NOT_A_STRING: &str = "Azure AD token must be a string, got {}";
const FAILED: &str = "Failed to get Azure AD token: {}";

#[derive(Debug, thiserror::Error)]
#[error("Azure AD token provider must be callable")]
struct NotCallable;

impl From<NotCallable> for PyErr {
    fn from(error: NotCallable) -> Self {
        PyTypeError::new_err(litellm_error::Report::new(&error).to_string())
    }
}

/// The `azure_ad_token_provider` keyword argument, kept alive for the rest of the call.
pub(crate) struct CallerTokenProvider {
    provider: Py<PyAny>,
}

/// Reads `azure_ad_token_provider`, ignoring the falsy and non-callable values litellm's
/// public API has always accepted in its place.
pub(crate) fn azure_ad_token_provider(
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<Option<CallerTokenProvider>> {
    Ok(kwargs
        .get_item("azure_ad_token_provider")?
        .filter(|provider| provider.is_callable() && provider.is_truthy().unwrap_or(false))
        .map(|provider| CallerTokenProvider {
            provider: provider.unbind(),
        }))
}

impl CallerTokenProvider {
    pub(crate) fn acquire(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        self.acquire_or(
            py,
            |token| {
                let message = PyString::new(token.py(), NOT_A_STRING)
                    .call_method1("format", (token.get_type(),))?;
                Ok(PyTypeError::new_err(message.unbind()))
            },
            |py, error| wrap_failure::<()>(py, FAILED, Err(error)).unwrap_err(),
        )
    }

    /// Calls the provider. A token that is not a string raises what `not_a_string` builds, and any
    /// other failure, including one building that error, is handed to `failed`.
    pub(crate) fn acquire_or<'py>(
        &self,
        py: Python<'py>,
        not_a_string: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<PyErr>,
        failed: impl FnOnce(Python<'py>, PyErr) -> PyErr,
    ) -> PyResult<ResolvedCredential> {
        let provider = self.provider.bind(py);
        if !provider.is_callable() {
            return Err(NotCallable.into());
        }
        let token = (|| {
            let token = provider.call0()?;
            if !token.is_instance_of::<PyString>() {
                return Err(not_a_string(&token)?);
            }
            Ok(token)
        })()
        .map_err(|error| {
            if error.is_instance_of::<PyTypeError>(py) || !error.is_instance_of::<PyException>(py) {
                error
            } else {
                failed(py, error)
            }
        })?;
        Ok(ResolvedCredential::AccessToken {
            token: SecretValue::new(token.extract::<String>()?),
            expires_on: None,
        })
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.provider)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::{PyRuntimeError, PyUnicodeEncodeError};

    use super::*;

    fn kwargs<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
            .get_item("kwargs")
            .unwrap()
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }

    fn provider<'py>(py: Python<'py>, source: &std::ffi::CStr) -> CallerTokenProvider {
        azure_ad_token_provider(&kwargs(py, source))
            .unwrap()
            .expect("a callable provider should project")
    }

    #[test]
    fn an_acquired_token_becomes_an_access_credential_without_an_expiry() {
        Python::initialize();
        Python::attach(|py| {
            let provider = provider(
                py,
                c"kwargs = {'azure_ad_token_provider': lambda: 'ey.token'}",
            );
            assert_eq!(
                provider.acquire(py).unwrap(),
                ResolvedCredential::AccessToken {
                    token: SecretValue::new("ey.token"),
                    expires_on: None,
                }
            );
        });
    }

    #[test]
    fn a_failing_provider_is_reported_as_an_azure_token_failure() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class ProviderError(Exception):
    def __format__(self, specification):
        return 'unavailable'
original = ProviderError('must use __format__')
def acquire():
    raise original
kwargs = {'azure_ad_token_provider': acquire}
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let provider = azure_ad_token_provider(
                &locals
                    .get_item("kwargs")
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap(),
            )
            .unwrap()
            .unwrap();
            let error = provider.acquire(py).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                error.value(py).str().unwrap().to_str().unwrap(),
                "Failed to get Azure AD token: unavailable"
            );
            assert!(
                error
                    .cause(py)
                    .unwrap()
                    .value(py)
                    .is(locals.get_item("original").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn a_non_string_token_is_rejected_by_type_and_never_reported_as_a_provider_failure() {
        Python::initialize();
        Python::attach(|py| {
            let error = provider(py, c"kwargs = {'azure_ad_token_provider': lambda: 1}")
                .acquire(py)
                .unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
            let message = error.value(py).str().unwrap().to_str().unwrap().to_owned();
            assert!(
                message.starts_with("Azure AD token must be a string, got "),
                "{message}"
            );
            assert!(message.contains("int"), "{message}");
        });
    }

    #[test]
    fn a_token_type_that_cannot_be_rendered_reports_that_failure_with_the_original_attached() {
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
kwargs = {'azure_ad_token_provider': lambda: Token()}
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let provider = azure_ad_token_provider(
                &locals
                    .get_item("kwargs")
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap(),
            )
            .unwrap()
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
    fn an_undecodable_token_keeps_its_own_failure_instead_of_the_provider_report() {
        Python::initialize();
        Python::attach(|py| {
            let error = provider(
                py,
                c"kwargs = {'azure_ad_token_provider': lambda: '\\ud800'}",
            )
            .acquire(py)
            .unwrap_err();
            assert!(error.is_instance_of::<PyUnicodeEncodeError>(py));
        });
    }

    #[test]
    fn a_provider_that_stops_being_callable_after_projection_is_rejected_by_type() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class Provider:
    def __call__(self):
        return 'ey.token'
kwargs = {'azure_ad_token_provider': Provider()}
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let provider = azure_ad_token_provider(
                &locals
                    .get_item("kwargs")
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap(),
            )
            .unwrap()
            .expect("a callable provider should project");
            py.run(
                pyo3::ffi::c_str!("del Provider.__call__"),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let error = provider.acquire(py).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert_eq!(
                error.value(py).str().unwrap().to_str().unwrap(),
                "Azure AD token provider must be callable"
            );
        });
    }

    #[rstest::rstest]
    #[case::missing(c"kwargs = {}")]
    #[case::none(c"kwargs = {'azure_ad_token_provider': None}")]
    #[case::not_callable(c"kwargs = {'azure_ad_token_provider': 'not-callable'}")]
    #[case::falsy_callable(
        c"class Falsy:\n    def __call__(self):\n        return 'ey.token'\n    def __bool__(self):\n        return False\nkwargs = {'azure_ad_token_provider': Falsy()}\n"
    )]
    #[case::truthiness_error(
        c"class Unusable:\n    def __call__(self):\n        return 'ey.token'\n    def __bool__(self):\n        raise RuntimeError('cannot decide')\nkwargs = {'azure_ad_token_provider': Unusable()}\n"
    )]
    fn only_callable_and_truthy_providers_project(#[case] source: &std::ffi::CStr) {
        Python::initialize();
        Python::attach(|py| {
            assert!(
                azure_ad_token_provider(&kwargs(py, source))
                    .unwrap()
                    .is_none()
            );
        });
    }
}
