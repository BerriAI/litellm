use std::{collections::HashMap, sync::Arc};

use futures_util::future::BoxFuture;
use litellm_core_utils::settings::Lookup;
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_secret_value::SecretValue;
use litellm_secrets::{
    Error,
    source::{SecretSource, Secrets},
};

use pyo3::{prelude::*, types::PyTuple};

use super::error::external_error;

const SECRET_MANAGER_MODULE: &str = "litellm.rust_bridge.host.secret_manager";

/// Reads secrets through the host module's `read_secrets`, so the configured manager, the key
/// management settings and the environment fallback behave exactly as they do in Python. A batch of
/// names is one host call.
pub(super) struct PythonSecrets {
    read_secrets: Arc<Py<PyAny>>,
    context: PythonContext,
}

impl PythonSecrets {
    pub(super) fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self::reading_with(
            py.import(SECRET_MANAGER_MODULE)?
                .getattr("read_secrets")?
                .unbind(),
            PythonContext::capture(py)?,
        ))
    }

    fn reading_with(read_secrets: Py<PyAny>, context: PythonContext) -> Self {
        Self {
            read_secrets: Arc::new(read_secrets),
            context,
        }
    }

    async fn read(&self, names: Vec<String>) -> Result<Vec<Option<SecretValue>>, Error> {
        let read_secrets = Arc::clone(&self.read_secrets);
        match attach_blocking(self.context.clone(), move |py| {
            PyTuple::new(py, names)
                .and_then(|names| read_secrets.bind(py).call1((names,)))
                .and_then(|values| values.extract::<Vec<Option<String>>>())
                .map(|values| {
                    values
                        .into_iter()
                        .map(|value| value.map(SecretValue::new))
                        .collect()
                })
                .map_err(|error| external_error(py, error))
        })
        .await
        {
            Ok(result) => result,
            Err(error) => Python::attach(|py| Err(external_error(py, error))),
        }
    }
}

struct ReadSecrets(HashMap<String, SecretValue>);

impl Lookup for ReadSecrets {
    fn get(&self, name: &str) -> Option<String> {
        self.0.get(name).map(|value| value.expose().to_owned())
    }
}

impl SecretSource for PythonSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        Box::pin(async move {
            Ok(self
                .read(vec![name.to_owned()])
                .await?
                .into_iter()
                .next()
                .flatten())
        })
    }

    fn resolve<'a>(&'a self, names: &'a [&str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        Box::pin(async move {
            let values = self
                .read(names.iter().map(|name| (*name).to_owned()).collect())
                .await?;
            let found = names
                .iter()
                .zip(values)
                .filter_map(|(name, value)| Some(((*name).to_owned(), value?)))
                .collect();
            Ok(Arc::new(ReadSecrets(found)) as Secrets)
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
calls = []
raised = KeyboardInterrupt('secret manager stopped')
values = {'MISTRAL_API_KEY': 'vault-key', 'MISTRAL_API_BASE': 'https://mistral.internal'}
def read_secrets(names):
    global read_on
    read_on = threading.get_ident()
    seen_context_values.append(request_var.get())
    calls.append(names)
    if 'RAISING' in names:
        raise raised
    return tuple(values.get(name) for name in names)
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
                    .get_item("read_secrets")
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
    async fn returns_what_read_secrets_returns(
        secrets: (PythonSecrets, Py<PyDict>),
        #[case] name: &str,
        #[case] expected: Option<&str>,
    ) {
        let value = secrets.0.get_secret_str(name).await.unwrap();

        assert_eq!(value.as_ref().map(|value| value.expose()), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn resolve_reads_every_name_in_one_python_call(secrets: (PythonSecrets, Py<PyDict>)) {
        let resolved = secrets
            .0
            .resolve(&["MISTRAL_API_KEY", "OTHER", "MISTRAL_API_BASE"])
            .await
            .unwrap();

        assert_eq!(
            resolved.get("MISTRAL_API_KEY").as_deref(),
            Some("vault-key")
        );
        assert_eq!(resolved.get("OTHER"), None);
        assert_eq!(
            resolved.get("MISTRAL_API_BASE").as_deref(),
            Some("https://mistral.internal")
        );
        let calls: Vec<Vec<String>> = Python::attach(|py| global(&secrets.1, py, "calls"));
        assert_eq!(
            calls,
            vec![vec![
                "MISTRAL_API_KEY".to_owned(),
                "OTHER".to_owned(),
                "MISTRAL_API_BASE".to_owned()
            ]]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn exceptions_surface_as_the_original_python_object(
        secrets: (PythonSecrets, Py<PyDict>),
    ) {
        let error = secrets
            .0
            .resolve(&["MISTRAL_API_KEY", "RAISING"])
            .await
            .err()
            .unwrap();

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
        secrets.0.resolve(&["MISTRAL_API_KEY"]).await.unwrap();

        let seen: Vec<String> = Python::attach(|py| global(&secrets.1, py, "seen_context_values"));
        assert_eq!(seen, vec!["request-value".to_owned()]);
    }
}
