use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use litellm_core::auth::{
    AuthError, AuthErrorKind, SecretString, TokenCredential, TokenLease, TokenProvider,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3_async_runtimes::TaskLocals;

pub struct PythonTokenProvider {
    callable: Py<PyAny>,
    locals: Option<TaskLocals>,
}

impl PythonTokenProvider {
    pub fn capture(py: Python<'_>, callable: Py<PyAny>) -> PyResult<Self> {
        if !callable.bind(py).is_callable() {
            return Err(PyTypeError::new_err("token provider must be callable"));
        }
        let locals = pyo3_async_runtimes::tokio::get_current_locals(py).ok();
        Ok(Self { callable, locals })
    }

    pub async fn resolve(&self) -> PyResult<TokenCredential> {
        let callable = Python::attach(|py| self.callable.clone_ref(py));
        let result = tokio::task::spawn_blocking(move || Python::attach(|py| callable.call0(py)))
            .await
            .map_err(|error| {
                PyRuntimeError::new_err(format!("token provider task failed: {error}"))
            })??;

        if let Some(token) = Python::attach(|py| parse_token_result(result.bind(py)))? {
            return Ok(token);
        }

        let locals = self.locals.as_ref().ok_or_else(|| {
            PyTypeError::new_err("awaitable token providers require an async LiteLLM entrypoint")
        })?;
        let future = Python::attach(|py| {
            if !result.bind(py).hasattr("__await__")? {
                return Err(PyTypeError::new_err(
                    "token provider must return a string, (string, expiry), or awaitable",
                ));
            }
            pyo3_async_runtimes::into_future_with_locals(locals, result.into_bound(py))
        })?;
        let awaited = future.await?;
        Python::attach(|py| {
            parse_token_result(awaited.bind(py))?.ok_or_else(|| {
                PyTypeError::new_err("awaitable token provider returned an invalid value")
            })
        })
    }
}

fn parse_token_result(value: &Bound<'_, PyAny>) -> PyResult<Option<TokenCredential>> {
    if let Ok(token) = value.extract::<String>() {
        if token.trim().is_empty() {
            return Err(PyTypeError::new_err(
                "token provider returned an empty token",
            ));
        }
        return Ok(Some(TokenCredential::NoStore(SecretString::new(token))));
    }
    let Ok(tuple) = value.cast::<PyTuple>() else {
        return Ok(None);
    };
    if tuple.len() != 2 {
        return Ok(None);
    }
    let token = tuple.get_item(0)?.extract::<String>()?;
    let expires_at = tuple.get_item(1)?.extract::<f64>()?;
    if token.trim().is_empty() || !expires_at.is_finite() {
        return Err(PyTypeError::new_err(
            "token provider returned an invalid token lease",
        ));
    }
    let now_epoch = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    Ok(Some(TokenCredential::Cached(TokenLease {
        token: SecretString::new(token),
        expires_at: Instant::now() + Duration::from_secs_f64((expires_at - now_epoch).max(0.0)),
    })))
}

#[async_trait]
impl TokenProvider for PythonTokenProvider {
    async fn token(&self) -> Result<TokenCredential, AuthError> {
        self.resolve().await.map_err(|_| {
            AuthError::new(
                AuthErrorKind::ExternalProviderFailed,
                "python_token_provider_failed",
                "Python token provider failed",
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use pyo3::types::PyDict;

    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn owned_python_provider_is_send_and_sync() {
        assert_send_sync::<PythonTokenProvider>();
    }

    #[test]
    fn synchronous_provider_returns_uncached_secret() {
        Python::initialize();
        let provider = Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                &CString::new("provider = lambda: 'secret-token'").unwrap(),
                None,
                Some(&locals),
            )?;
            PythonTokenProvider::capture(py, locals.get_item("provider")?.unwrap().unbind())
        })
        .unwrap();
        let credential = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(provider.resolve())
            .unwrap();
        let TokenCredential::NoStore(token) = credential else {
            panic!("plain tokens must not be cached")
        };
        assert_eq!(token.expose(), "secret-token");
        assert_eq!(format!("{token:?}"), "[redacted]");
    }

    #[test]
    fn original_python_exception_is_preserved_by_resolve() {
        Python::initialize();
        let provider = Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                &CString::new("def provider():\n    raise LookupError('credential missing')")
                    .unwrap(),
                None,
                Some(&locals),
            )?;
            PythonTokenProvider::capture(py, locals.get_item("provider")?.unwrap().unbind())
        })
        .unwrap();
        let error = pyo3_async_runtimes::tokio::get_runtime()
            .block_on(provider.resolve())
            .unwrap_err();
        Python::attach(|py| {
            assert!(error.is_instance_of::<pyo3::exceptions::PyLookupError>(py));
            assert_eq!(error.value(py).to_string(), "credential missing");
        });
    }
}
