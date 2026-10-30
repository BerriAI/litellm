use std::{pin::Pin, sync::Arc};

use futures_util::{Stream, StreamExt, stream};
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_redis_client::{BasicAuth, RedisError, RedisErrorKind, StreamingCredentialsProvider};
use pyo3::prelude::*;

const TOKEN_ONLY_USERNAME: &str = "default";

/// Adapts a redis-py `CredentialProvider` object. Each connection that subscribes calls its
/// `get_credentials` once, so every new connection pulls fresh credentials the way redis-py does.
pub(crate) struct PythonCredentialProvider {
    provider: Arc<Py<PyAny>>,
    context: PythonContext,
}

impl PythonCredentialProvider {
    pub(crate) fn new(py: Python<'_>, provider: Py<PyAny>) -> PyResult<Self> {
        Ok(Self {
            provider: Arc::new(provider),
            context: PythonContext::capture(py)?,
        })
    }

    async fn fetch(provider: Arc<Py<PyAny>>, context: PythonContext) -> Result<BasicAuth, RedisError> {
        let fetched = attach_blocking(context, move |py| -> PyResult<(Option<String>, String)> {
            let credentials: Vec<String> = provider.bind(py).call_method0("get_credentials")?.extract()?;
            match <[String; 2]>::try_from(credentials) {
                Ok([username, password]) => Ok((Some(username), password)),
                Err(credentials) => match <[String; 1]>::try_from(credentials) {
                    Ok([password]) => Ok((None, password)),
                    Err(credentials) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "get_credentials returned {} values, expected 1 or 2",
                        credentials.len()
                    ))),
                },
            }
        })
        .await
        .and_then(|fetched| fetched);
        match fetched {
            Ok((username, password)) => Ok(BasicAuth::new(
                username.unwrap_or_else(|| TOKEN_ONLY_USERNAME.to_owned()),
                password,
            )),
            Err(error) => Err(RedisError::from((
                RedisErrorKind::AuthenticationFailed,
                "the Redis credential_provider failed",
                Python::attach(|py| error.value(py).to_string()),
            ))),
        }
    }
}

impl StreamingCredentialsProvider for PythonCredentialProvider {
    fn subscribe(&self) -> Pin<Box<dyn Stream<Item = Result<BasicAuth, RedisError>> + Send + 'static>> {
        let fetch = Self::fetch(self.provider.clone(), self.context.clone());
        Box::pin(stream::once(fetch).chain(stream::pending()))
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;
    use litellm_redis_client::StreamingCredentialsProvider;
    use pyo3::{prelude::*, types::PyDict};
    use rstest::rstest;

    use super::PythonCredentialProvider;

    fn provider(source: &str) -> (PythonCredentialProvider, Py<PyDict>) {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(&std::ffi::CString::new(source).unwrap(), Some(&namespace), None).unwrap();
            let object = namespace.get_item("provider").unwrap().unwrap().unbind();
            (PythonCredentialProvider::new(py, object).unwrap(), namespace.unbind())
        })
    }

    #[rstest]
    #[case::username_and_password("('app', 'secret')", ("app", "secret"))]
    #[case::token_only_sends_the_default_user("('token',)", ("default", "token"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn each_subscription_pulls_fresh_credentials(#[case] reply: &str, #[case] expected: (&str, &str)) {
        let source = format!(
            "class Provider:\n    calls = 0\n    def get_credentials(self):\n        Provider.calls += 1\n        return {reply}\nprovider = Provider()\n"
        );
        let (provider, namespace) = provider(&source);

        let first = provider.subscribe().next().await.unwrap().unwrap();
        let second = provider.subscribe().next().await.unwrap().unwrap();

        assert_eq!((first.username(), first.password()), expected);
        assert_eq!((second.username(), second.password()), expected);
        let calls: i64 = Python::attach(|py| {
            namespace
                .bind(py)
                .get_item("Provider")
                .unwrap()
                .unwrap()
                .getattr("calls")
                .unwrap()
                .extract()
                .unwrap()
        });
        assert_eq!(calls, 2);
    }

    #[rstest]
    #[case::raises("class Provider:\n    def get_credentials(self):\n        raise RuntimeError('vault down')\nprovider = Provider()\n")]
    #[case::wrong_arity("class Provider:\n    def get_credentials(self):\n        return ('a', 'b', 'c')\nprovider = Provider()\n")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failing_provider_fails_the_connection(#[case] source: &str) {
        let (provider, _namespace) = provider(source);

        let failed = provider.subscribe().next().await.unwrap();

        assert!(failed.is_err());
    }
}
