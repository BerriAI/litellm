//! Credentials the caller supplies as Python callables, projected out of a route's
//! keyword arguments and acquired on the host's own thread when the call asks for one.

use litellm_auth::{ResolvedCredential, SecretValue};
use litellm_host_python::wrap_failure;
use pyo3::{
    exceptions::PyTypeError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyString},
};

const NOT_CALLABLE: &str = "Azure AD token provider must be callable";
const NOT_A_STRING: &str = "Azure AD token must be a string, got {}";
const FAILED: &str = "Failed to get Azure AD token: {}";

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
        let provider = self.provider.bind(py);
        if !provider.is_callable() {
            return Err(PyTypeError::new_err(NOT_CALLABLE));
        }
        let token = wrap_failure(
            py,
            FAILED,
            (|| {
                let token = provider.call0()?;
                if !token.is_instance_of::<PyString>() {
                    let message = PyString::new(py, NOT_A_STRING)
                        .call_method1("format", (token.get_type(),))?;
                    return Err(PyTypeError::new_err(message.unbind()));
                }
                Ok(token)
            })(),
        )?;
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

    #[test]
    fn only_callable_and_truthy_providers_project() {
        Python::initialize();
        Python::attach(|py| {
            for source in [
                c"kwargs = {}",
                c"kwargs = {'azure_ad_token_provider': None}",
                c"kwargs = {'azure_ad_token_provider': 'not-callable'}",
                c"
class Falsy:
    def __call__(self):
        return 'ey.token'
    def __bool__(self):
        return False
kwargs = {'azure_ad_token_provider': Falsy()}
",
                c"
class Unusable:
    def __call__(self):
        return 'ey.token'
    def __bool__(self):
        raise RuntimeError('cannot decide')
kwargs = {'azure_ad_token_provider': Unusable()}
",
            ] {
                assert!(
                    azure_ad_token_provider(&kwargs(py, source))
                        .unwrap()
                        .is_none()
                );
            }
        });
    }
}
