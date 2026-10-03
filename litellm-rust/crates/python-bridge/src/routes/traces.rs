use std::collections::BTreeMap;

use litellm_http::ClientVariant;
use litellm_traces::{QueryScope, ReadQuery, Tenant, query::named::ReadAccessParams};
use litellm_traces_clickhouse::{Config, Error, InsertTable, Parameter, QueryReaders};
use prost::Message;
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

#[derive(Message)]
struct OtlpErrorStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
}

#[pyfunction]
pub fn trace_encode_error<'py>(py: Python<'py>, message: &str) -> Bound<'py, PyBytes> {
    let status = OtlpErrorStatus {
        code: 0,
        message: message.to_owned(),
    };
    PyBytes::new(py, &status.encode_to_vec())
}

fn map_error(error: Error) -> PyErr {
    map_error_ref(&error)
}

fn map_error_ref(error: &Error) -> PyErr {
    use litellm_storage_clickhouse::Error as StorageError;

    match error {
        Error::Decode(litellm_traces::Error::TooLarge)
        | Error::InsertTooLarge
        | Error::ReadTooLarge => PyOverflowError::new_err(error.to_string()),
        Error::InvalidRow
        | Error::InvalidTable
        | Error::InvalidCursor(_)
        | Error::AmbiguousTrace
        | Error::TraceChanged
        | Error::Decode(_)
        | Error::InvalidSchema
        | Error::InvalidQuery
        | Error::InvalidParameters
        | Error::InvalidScope => PyValueError::new_err(error.to_string()),
        Error::Task
        | Error::SchemaFailed(_)
        | Error::SchemaTransport
        | Error::MissingSecret
        | Error::Busy
        | Error::ProvisionFailed(_)
        | Error::ProvisionTransport
        | Error::InvalidResponse => PyRuntimeError::new_err(error.to_string()),
        Error::Cached(source) => map_error_ref(source),
        Error::Storage(source) => match source {
            StorageError::InvalidRow
            | StorageError::InvalidTable
            | StorageError::InvalidSchema
            | StorageError::EmptySql
            | StorageError::InvalidParameters
            | StorageError::InvalidQuery => PyValueError::new_err(error.to_string()),
            StorageError::InsertTooLarge => PyOverflowError::new_err(error.to_string()),
            StorageError::InvalidUrl
            | StorageError::QueryFailed(_)
            | StorageError::InsertFailed(_)
            | StorageError::SchemaFailed(_)
            | StorageError::ResponseTooLarge
            | StorageError::InvalidResponse
            | StorageError::Transport => PyRuntimeError::new_err(error.to_string()),
        },
    }
}

fn map_sql_error(error: Error) -> PyErr {
    match error {
        Error::Storage(litellm_storage_clickhouse::Error::QueryFailed(400 | 404)) => {
            PyValueError::new_err(error.to_string())
        }
        error => map_error(error),
    }
}

#[pyclass(frozen)]
pub struct NativeTraceConfig {
    inner: Config,
}

#[pymethods]
impl NativeTraceConfig {
    #[new]
    fn new(
        database: String,
        url: &str,
        retention_days: u32,
        max_attribute_value_bytes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Config::new(database, url, retention_days, max_attribute_value_bytes)
                .map_err(map_error)?,
        })
    }
}

#[pyclass]
pub struct NativeTraceStorage {
    config: Config,
    query_readers: QueryReaders,
}

#[pymethods]
impl NativeTraceStorage {
    #[new]
    fn new(config: PyRef<'_, NativeTraceConfig>) -> PyResult<Self> {
        Ok(Self {
            query_readers: QueryReaders::new(
                config.inner.storage().writer().clone(),
                config.inner.storage().database().to_owned(),
            ),
            config: config.inner.clone(),
        })
    }

    fn ensure_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().writer().clone();
        let database = self.config.storage().database().to_owned();
        let retention_days = self.config.retention_days();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces_clickhouse::ensure_schema(
                    &client,
                    &connection,
                    &database,
                    retention_days,
                )
                .await
            },
            map_error,
        )
    }

    fn insert_rows<'py>(
        &self,
        py: Python<'py>,
        table: &str,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] rows: Vec<
            BTreeMap<String, serde_json::Value>,
        >,
    ) -> PyResult<Bound<'py, PyAny>> {
        let table = InsertTable::parse(table).map_err(map_error)?;
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().writer().clone();
        let database = self.config.storage().database().to_owned();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces_clickhouse::insert_rows(&client, &connection, &database, table, rows)
                    .await
            },
            map_error,
        )
    }

    fn ingest<'py>(
        &self,
        py: Python<'py>,
        payload: &[u8],
        content_type: Option<String>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] tenant: Tenant,
    ) -> PyResult<Bound<'py, PyAny>> {
        let payload = payload.to_vec();
        let max_value_bytes = self.config.max_attribute_value_bytes();
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().writer().clone();
        let database = self.config.storage().database().to_owned();
        crate::execution::run_async(
            py,
            async move {
                let rows = tokio::task::spawn_blocking(move || {
                    litellm_traces::decode_otlp(&payload, content_type.as_deref()).map(|spans| {
                        litellm_traces_clickhouse::span_rows(spans, &tenant, max_value_bytes)
                    })
                })
                .await
                .map_err(|_| Error::Task)??;
                let count = rows.len();
                litellm_traces_clickhouse::insert_shared_rows(
                    &client,
                    &connection,
                    &database,
                    InsertTable::OtelTraces,
                    rows,
                )
                .await?;
                Ok(count)
            },
            map_error,
        )
    }

    #[pyo3(signature = (scope, start_ms, end_ms, cursor, limit))]
    fn list_traces<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: ReadAccessParams,
        start_ms: i64,
        end_ms: i64,
        cursor: Option<String>,
        limit: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces_clickhouse::list_traces(
                    &client,
                    &connection,
                    &scope,
                    start_ms,
                    end_ms,
                    cursor.as_deref(),
                    limit,
                )
                .await
            },
            map_error,
        )
    }

    #[pyo3(signature = (trace_id, scope, trace_ref, cursor=None, page_size=None))]
    fn get_trace<'py>(
        &self,
        py: Python<'py>,
        trace_id: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: ReadAccessParams,
        trace_ref: String,
        cursor: Option<String>,
        page_size: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        crate::execution::run_async(
            py,
            async move {
                if let Some(page_size) = page_size {
                    litellm_traces_clickhouse::get_trace_page(
                        &client,
                        &connection,
                        &scope,
                        &trace_id,
                        &trace_ref,
                        cursor.as_deref(),
                        page_size,
                    )
                    .await
                } else if cursor.is_some() {
                    Err(Error::InvalidParameters)
                } else {
                    litellm_traces_clickhouse::get_trace(
                        &client,
                        &connection,
                        &scope,
                        &trace_id,
                        &trace_ref,
                    )
                    .await
                }
            },
            map_error,
        )
    }

    fn get_span<'py>(
        &self,
        py: Python<'py>,
        trace_id: String,
        span_id: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: ReadAccessParams,
        trace_ref: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces_clickhouse::get_span(
                    &client,
                    &connection,
                    &scope,
                    &trace_id,
                    &span_id,
                    &trace_ref,
                )
                .await
            },
            map_error,
        )
    }

    #[pyo3(signature = (trace_id, span_id, scope, trace_ref, cursor))]
    fn get_span_error<'py>(
        &self,
        py: Python<'py>,
        trace_id: String,
        span_id: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: ReadAccessParams,
        trace_ref: String,
        cursor: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces_clickhouse::get_span_error(
                    &client,
                    &connection,
                    &scope,
                    &trace_id,
                    &span_id,
                    &trace_ref,
                    cursor.as_deref(),
                )
                .await
            },
            map_error,
        )
    }

    fn query_sql<'py>(
        &self,
        py: Python<'py>,
        sql: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        secret: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        if sql.trim().is_empty() {
            return Err(map_error(
                litellm_storage_clickhouse::Error::EmptySql.into(),
            ));
        }
        let readers = self.query_readers.clone();
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        crate::execution::run_async(
            py,
            async move {
                let _permit = readers.acquire()?;
                let connection = readers.connection(&client, &scope, &secret).await?;
                litellm_traces_clickhouse::query_sql(&client, &connection, &sql).await
            },
            map_sql_error,
        )
    }

    fn query_help<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        secret: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let readers = self.query_readers.clone();
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        crate::execution::run_async(
            py,
            async move {
                let _permit = readers.acquire()?;
                let connection = readers.connection(&client, &scope, &secret).await?;
                litellm_traces_clickhouse::query_help(&client, &connection).await
            },
            map_sql_error,
        )
    }

    fn query<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] parameters: BTreeMap<
            String,
            Parameter,
        >,
    ) -> PyResult<Bound<'py, PyAny>> {
        let query =
            ReadQuery::parse(query).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let connection = self.config.storage().reader().clone();
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        crate::execution::run_async(
            py,
            async move {
                litellm_traces_clickhouse::execute_named_read(
                    &client,
                    &connection,
                    query,
                    &parameters,
                )
                .await
            },
            map_error,
        )
    }
}

/// The `otel_traces` rows an export would be stored as, without writing them.
#[pyfunction]
pub fn trace_span_rows<'py>(
    py: Python<'py>,
    body: &[u8],
    content_type: Option<&str>,
    #[pyo3(from_py_with = litellm_host_python::from_py_argument)] tenant: Tenant,
    max_attribute_value_bytes: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = py
        .detach(|| {
            litellm_traces::decode_otlp(body, content_type).map(|spans| {
                litellm_traces_clickhouse::span_rows(spans, &tenant, max_attribute_value_bytes)
            })
        })
        .map_err(|error| map_error(error.into()))?;
    litellm_host_python::Pythonized(rows).into_pyobject(py)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::row(Error::InvalidRow, "ValueError")]
    #[case::insert_budget(Error::InsertTooLarge, "OverflowError")]
    #[case::scope(Error::InvalidScope, "ValueError")]
    #[case::schema(Error::SchemaFailed(503), "RuntimeError")]
    #[case::reader(Error::MissingSecret, "RuntimeError")]
    #[case::storage(
        Error::Storage(litellm_storage_clickhouse::Error::InvalidUrl),
        "RuntimeError"
    )]
    #[case::cached_scope(Error::Cached(std::sync::Arc::new(Error::InvalidScope)), "ValueError")]
    fn trace_failures_preserve_public_exception_types(
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

    #[rstest]
    #[case::invalid_sql(400, "ValueError")]
    #[case::missing_table(404, "ValueError")]
    #[case::unavailable(503, "RuntimeError")]
    fn wrapped_query_status_preserves_public_exception_type(
        #[case] status: u16,
        #[case] exception_name: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = Error::Storage(litellm_storage_clickhouse::Error::QueryFailed(status));
            let message = error.to_string();
            let exception = map_sql_error(error);
            assert_eq!(exception.get_type(py).name().unwrap(), exception_name);
            assert_eq!(
                exception.value(py).str().unwrap().to_str().unwrap(),
                message
            );
        });
    }

    #[rstest]
    #[case::decode_budget(Error::Decode(litellm_traces::Error::TooLarge), "OverflowError")]
    #[case::invalid_export(Error::Decode(litellm_traces::Error::InvalidPayload), "ValueError")]
    #[case::cursor(Error::InvalidCursor("trace"), "ValueError")]
    #[case::ambiguous(Error::AmbiguousTrace, "ValueError")]
    #[case::changed_snapshot(Error::TraceChanged, "ValueError")]
    #[case::read_budget(Error::ReadTooLarge, "OverflowError")]
    fn trace_read_and_ingest_failures_preserve_public_exception_types(
        #[case] error: Error,
        #[case] exception_name: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                map_error(error).get_type(py).name().unwrap(),
                exception_name
            );
        });
    }
}
