use std::collections::BTreeMap;

use litellm_http::ClientVariant;
use litellm_spend_clickhouse::{Config, Error};
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
};

fn map_error(error: Error) -> PyErr {
    let message = litellm_error::Report::new(&error).to_string();
    if error.is_insert_too_large() {
        PyOverflowError::new_err(message)
    } else if error.is_request() {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}

#[pyclass(frozen)]
pub struct NativeClickHouseSpendConfig {
    inner: Config,
}

#[pymethods]
impl NativeClickHouseSpendConfig {
    #[new]
    fn new(database: String, url: &str, retention_days: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Config::new(database, url, retention_days).map_err(map_error)?,
        })
    }
}

#[pyclass]
pub struct NativeClickHouseSpendStorage {
    config: Config,
}

#[pymethods]
impl NativeClickHouseSpendStorage {
    #[new]
    fn new(config: PyRef<'_, NativeClickHouseSpendConfig>) -> Self {
        Self {
            config: config.inner.clone(),
        }
    }

    fn ensure_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let config = self.config.clone();
        crate::execution::run_async(
            py,
            async move { litellm_spend_clickhouse::ensure_schema(&client, &config).await },
            map_error,
        )
    }

    fn insert_rows<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] rows: Vec<
            BTreeMap<String, serde_json::Value>,
        >,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let config = self.config.clone();
        crate::execution::run_async(
            py,
            async move {
                let storage = config.storage();
                litellm_spend_clickhouse::insert_rows(
                    &client,
                    storage.writer(),
                    storage.database(),
                    rows,
                )
                .await
            },
            map_error,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    use litellm_storage_clickhouse::Error as StorageError;

    #[rstest]
    #[case::timestamp(Error::NonIntegerTimestamp { field: "start_time" }, "ValueError")]
    #[case::insert_limit(
        StorageError::InvalidLimit("CLICKHOUSE_TRACE_MAX_INSERT_BYTES").into(),
        "ValueError"
    )]
    #[case::insert_budget(StorageError::InsertTooLarge.into(), "OverflowError")]
    #[case::schema(StorageError::SchemaFailed(503).into(), "RuntimeError")]
    #[case::migration(
        Error::Migration(sqlx::migrate::MigrateError::VersionMismatch(6)),
        "RuntimeError"
    )]
    #[case::storage(
        Config::new("spend".into(), "ftp://localhost", 1).err().unwrap(),
        "RuntimeError"
    )]
    fn spend_failures_preserve_public_exception_types(
        #[case] error: Error,
        #[case] exception_name: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let message = litellm_error::Report::new(&error).to_string();
            let exception = map_error(error);
            assert_eq!(exception.get_type(py).name().unwrap(), exception_name);
            assert_eq!(
                exception.value(py).str().unwrap().to_str().unwrap(),
                message
            );
        });
    }

    #[test]
    fn spend_failures_render_their_cause() {
        let error = Config::new("spend".into(), "not a url", 1).err().unwrap();
        let cause = std::error::Error::source(&error).unwrap().to_string();
        Python::initialize();
        Python::attach(|py| {
            let message = map_error(error).value(py).str().unwrap().to_string();
            assert!(message.ends_with(&cause), "{message}");
        });
    }
}
