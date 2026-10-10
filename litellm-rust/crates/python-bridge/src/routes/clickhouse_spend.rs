use std::collections::BTreeMap;

use litellm_http::ClientVariant;
use litellm_spend_clickhouse::{Config, Error};
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
};

fn map_error(error: Error) -> PyErr {
    use litellm_storage_clickhouse::Error as StorageError;
    match error {
        Error::InsertTooLarge | Error::Storage(StorageError::InsertTooLarge) => {
            PyOverflowError::new_err(error.to_string())
        }
        Error::InvalidRow
        | Error::InvalidLimit(_)
        | Error::InvalidSchema
        | Error::Storage(
            StorageError::InvalidRow
            | StorageError::InvalidLimit(_)
            | StorageError::InvalidTable
            | StorageError::InvalidSchema
            | StorageError::EmptySql
            | StorageError::InvalidParameters
            | StorageError::InvalidQuery,
        ) => PyValueError::new_err(error.to_string()),
        Error::Migration(_) | Error::Storage(_) => PyRuntimeError::new_err(error.to_string()),
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

    #[rstest]
    #[case::row(Error::InvalidRow, "ValueError")]
    #[case::insert_limit(Error::InvalidLimit("CLICKHOUSE_TRACE_MAX_INSERT_BYTES"), "ValueError")]
    #[case::insert_timeout(
        Error::Storage(litellm_storage_clickhouse::Error::InvalidLimit(
            "CLICKHOUSE_INSERT_TIMEOUT_SECONDS"
        )),
        "ValueError"
    )]
    #[case::insert_budget(Error::InsertTooLarge, "OverflowError")]
    #[case::schema(
        Error::Storage(litellm_storage_clickhouse::Error::SchemaFailed(503)),
        "RuntimeError"
    )]
    #[case::migration(
        Error::Migration(sqlx::migrate::MigrateError::VersionMismatch(6)),
        "RuntimeError"
    )]
    #[case::storage(
        Error::Storage(litellm_storage_clickhouse::Error::InvalidUrl),
        "RuntimeError"
    )]
    fn spend_failures_preserve_public_exception_types(
        #[case] error: Error,
        #[case] exception_name: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let message = error.to_string();
            let exception = map_error(error);
            assert_eq!(exception.get_type(py).name().unwrap(), exception_name);
            assert_eq!(
                exception.value(py).str().unwrap().to_str().unwrap(),
                message
            );
        });
    }
}
