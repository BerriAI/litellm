use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_secrets::{Error, SecretValue, source::SecretSource};
use pyo3::prelude::*;

use super::error::external_error;

/// Reads each secret through Python's `get_secret_str`, so the configured manager, the key
/// management settings and the environment fallback behave exactly as they do in Python.
pub(super) struct PythonSecrets {
    get_secret_str: Arc<Py<PyAny>>,
    context: PythonContext,
}

impl PythonSecrets {
    pub(super) fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self::reading_with(
            py.import("litellm.secret_managers.main")?
                .getattr("get_secret_str")?
                .unbind(),
            PythonContext::capture(py)?,
        ))
    }

    fn reading_with(get_secret_str: Py<PyAny>, context: PythonContext) -> Self {
        Self {
            get_secret_str: Arc::new(get_secret_str),
            context,
        }
    }
}

impl SecretSource for PythonSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        let get_secret_str = Arc::clone(&self.get_secret_str);
        let context = self.context.clone();
        let name = name.to_owned();
        Box::pin(async move {
            match attach_blocking(context, move |py| {
                get_secret_str
                    .bind(py)
                    .call1((name,))
                    .and_then(|value| value.extract::<Option<String>>())
                    .map(|value| value.map(SecretValue::new))
                    .map_err(|error| external_error(py, error))
            })
            .await
            {
                Ok(result) => result,
                Err(error) => Python::attach(|py| Err(external_error(py, error))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use litellm_secrets::source::SecretSource;
    use pyo3::{prelude::*, types::PyDict};
    use rstest::{fixture, rstest};

    use super::PythonSecrets;
    use crate::secrets::python_error;
    use litellm_host_python::PythonContext;

    #[fixture]
    fn namespace() -> Py<PyDict> {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"
import contextvars
import threading
read_on = None
request_var = contextvars.ContextVar('request_var', default=None)
seen_context_values = []
raised = KeyboardInterrupt('secret manager stopped')
def get_secret_str(name):
    global read_on
    read_on = threading.get_ident()
    seen_context_values.append(request_var.get())
    if name == 'RAISING':
        raise raised
    return {'MISTRAL_API_KEY': 'vault-key'}.get(name)
",
                Some(&namespace),
                None,
            )
            .unwrap();
            namespace.unbind()
        })
    }

    #[fixture]
    fn secrets(namespace: Py<PyDict>) -> (PythonSecrets, Py<PyDict>) {
        let (reader, context) = Python::attach(|py| {
            let namespace = namespace.bind(py);
            namespace
                .get_item("request_var")
                .unwrap()
                .unwrap()
                .call_method1("set", ("request-value",))
                .unwrap();
            (
                namespace
                    .get_item("get_secret_str")
                    .unwrap()
                    .unwrap()
                    .unbind(),
                PythonContext::capture(py).unwrap(),
            )
        });
        (PythonSecrets::reading_with(reader, context), namespace)
    }

    fn global<T: for<'a, 'py> FromPyObject<'a, 'py, Error: std::fmt::Debug>>(
        namespace: &Py<PyDict>,
        py: Python<'_>,
        name: &str,
    ) -> T {
        namespace
            .bind(py)
            .get_item(name)
            .unwrap()
            .unwrap()
            .extract()
            .unwrap()
    }

    #[rstest]
    #[case::found("MISTRAL_API_KEY", Some("vault-key"))]
    #[case::missing("OTHER", None)]
    #[tokio::test]
    async fn returns_what_get_secret_str_returns(
        secrets: (PythonSecrets, Py<PyDict>),
        #[case] name: &str,
        #[case] expected: Option<&str>,
    ) {
        let value = secrets.0.get_secret_str(name).await.unwrap();

        assert_eq!(value.as_ref().map(|value| value.expose()), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn exceptions_surface_as_the_original_python_object(
        secrets: (PythonSecrets, Py<PyDict>),
    ) {
        let error = secrets.0.get_secret_str("RAISING").await.unwrap_err();

        Python::attach(|py| {
            let surfaced = python_error(py, &error).expect("the Python exception is preserved");
            let raised: Py<PyAny> = global(&secrets.1, py, "raised");
            assert!(surfaced.value(py).is(raised.bind(py)));
        });
    }

    #[rstest]
    #[tokio::test]
    async fn reads_run_off_the_thread_polling_the_route(secrets: (PythonSecrets, Py<PyDict>)) {
        let polling: u64 = Python::attach(|py| {
            py.import("threading")
                .unwrap()
                .call_method0("get_ident")
                .unwrap()
                .extract()
                .unwrap()
        });

        secrets.0.get_secret_str("MISTRAL_API_KEY").await.unwrap();

        let read_on: u64 = Python::attach(|py| global(&secrets.1, py, "read_on"));
        assert_ne!(read_on, polling);
    }

    #[rstest]
    #[tokio::test]
    async fn reads_see_the_callers_contextvars(secrets: (PythonSecrets, Py<PyDict>)) {
        secrets.0.get_secret_str("MISTRAL_API_KEY").await.unwrap();

        let seen: Vec<String> = Python::attach(|py| global(&secrets.1, py, "seen_context_values"));
        assert_eq!(seen, vec!["request-value".to_owned()]);
    }
}
