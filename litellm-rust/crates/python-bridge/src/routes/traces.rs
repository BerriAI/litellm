use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use litellm_http::ClientVariant;
use litellm_traces::{
    QueryScope, Tenant,
    api::TraceQueryWindow,
    search::{RunField, RunFilter, RunSearch},
    store::{RunOrder, SpanPart, TextRange},
};
use litellm_traces_cache::{PageRequest, ReadError, TraceReader, resolve_run_window};
use litellm_traces_clickhouse::{ClickHouseTraces, Config, Error, InsertTable, QueryReaders};
use prost::Message;
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

pyo3::import_exception!(litellm.rust_bridge.trace.errors, TraceChanged);
pyo3::import_exception!(litellm.rust_bridge.trace.errors, TraceQueryError);

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
        Error::Decode(litellm_traces::Error::TooLarge) | Error::InsertTooLarge => {
            PyOverflowError::new_err(error.to_string())
        }
        Error::InvalidRow
        | Error::InvalidLimit(_)
        | Error::InvalidTable
        | Error::Decode(_)
        | Error::InvalidSchema
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
            | StorageError::InvalidLimit(_)
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

fn map_read_error(error: ReadError<Error>) -> PyErr {
    match error {
        error @ (ReadError::InvalidParameters
        | ReadError::InvalidCursor(_)
        | ReadError::AmbiguousTrace) => PyValueError::new_err(error.to_string()),
        error @ ReadError::TraceChanged => TraceChanged::new_err(error.to_string()),
        error @ ReadError::TooLarge => PyOverflowError::new_err(error.to_string()),
        error @ ReadError::Encode(_) => PyRuntimeError::new_err(error.to_string()),
        ReadError::Store(error) => map_error_ref(&error),
    }
}

fn run_window(
    start_ms: Option<i64>,
    end_ms: Option<i64>,
    as_of_ms: Option<u64>,
    cursor: Option<&str>,
) -> PyResult<TraceQueryWindow> {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    resolve_run_window(
        start_ms,
        end_ms,
        as_of_ms,
        cursor,
        TraceQueryWindow {
            start_ms: now_ms as i64 - 86_400_000,
            end_ms: now_ms as i64,
            as_of_ms: now_ms.saturating_sub(1),
        },
    )
    .map_err(map_read_error)
}

fn run_filter(window: TraceQueryWindow, q: &str, trace_refs: Vec<String>) -> PyResult<RunFilter> {
    Ok(RunFilter {
        start_ms: window.start_ms,
        end_ms: window.end_ms,
        as_of_ms: window.as_of_ms,
        search: RunSearch::parse(q).map_err(|error| PyValueError::new_err(error.to_string()))?,
        trace_refs,
    })
}

fn parsed<T: std::str::FromStr>(kind: &str, value: &str) -> PyResult<T> {
    value
        .parse()
        .map_err(|_| PyValueError::new_err(format!("unknown {kind} {value}")))
}

fn map_sql_error(error: Error) -> PyErr {
    map_sql_error_ref(&error)
}

fn map_sql_error_ref(error: &Error) -> PyErr {
    match error {
        Error::Storage(litellm_storage_clickhouse::Error::QueryFailed(failure)) => {
            TraceQueryError::new_err((
                sql_failure_kind(failure),
                failure.code,
                failure.message.clone(),
            ))
        }
        Error::Storage(litellm_storage_clickhouse::Error::ResponseTooLarge) => {
            TraceQueryError::new_err((
                "limited",
                None::<u32>,
                "Query exceeded the response size limit",
            ))
        }
        Error::Cached(source) => map_sql_error_ref(source),
        error => map_error_ref(error),
    }
}

fn sql_failure_kind(failure: &litellm_storage_clickhouse::QueryFailure) -> &'static str {
    // https://github.com/ClickHouse/ClickHouse/blob/v26.9.6.6-stable/src/Common/ErrorCodes.cpp
    match failure.code {
        Some(158 | 159 | 160 | 167 | 168 | 191 | 202 | 229 | 241 | 290 | 307 | 396 | 776) => {
            "limited"
        }
        Some(
            6 | 34 | 35 | 36 | 42 | 43 | 44 | 46 | 47 | 48 | 50 | 53 | 60 | 62 | 63 | 69 | 70 | 72
            | 73 | 78 | 80 | 81 | 115 | 164 | 291 | 344 | 392 | 452 | 472 | 497,
        ) => "rejected",
        Some(192 | 193 | 194 | 516) => "unavailable",
        _ if matches!(failure.status, 408 | 413 | 429) => "limited",
        _ if matches!(failure.status, 400 | 404 | 405 | 406 | 411 | 415 | 422) => "rejected",
        _ => "unavailable",
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
    reader: Arc<TraceReader>,
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
            reader: Arc::new(TraceReader::new(
                litellm_storage_clickhouse::READ_LIMITS.response_bytes,
            )),
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

    #[pyo3(signature = (scope, start_ms, end_ms, q, cursor, limit, order, trace_refs=Vec::new(), as_of_ms=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "one parameter per Python argument"
    )]
    fn list_traces<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        q: &str,
        cursor: Option<String>,
        limit: u32,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] order: RunOrder,
        trace_refs: Vec<String>,
        as_of_ms: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let window = run_window(start_ms, end_ms, as_of_ms, cursor.as_deref())?;
        let filter = run_filter(window, q, trace_refs)?;
        let page = PageRequest { cursor, limit };
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader
                    .list_traces(&store, &scope, &filter, order, &page)
                    .await
            },
            map_read_error,
        )
    }

    #[pyo3(signature = (scope, start_ms, end_ms, q, trace_refs=Vec::new(), as_of_ms=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "one parameter per Python argument"
    )]
    fn count_traces<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        q: &str,
        trace_refs: Vec<String>,
        as_of_ms: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let window = run_window(start_ms, end_ms, as_of_ms, None)?;
        let filter = run_filter(window, q, trace_refs)?;
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader.count_traces(&store, &scope, &filter).await
            },
            map_read_error,
        )
    }

    /// `tail` reads the last `max_chars` characters instead of starting at `offset`.
    #[pyo3(signature = (trace_id, trace_ref, span_ids, part, scope, offset=0, max_chars=None, tail=false, contains=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "one parameter per Python argument"
    )]
    fn span_text<'py>(
        &self,
        py: Python<'py>,
        trace_id: String,
        trace_ref: String,
        span_ids: Vec<String>,
        part: &str,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        offset: u64,
        max_chars: Option<u64>,
        tail: bool,
        contains: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let part: SpanPart = parsed("span part", part)?;
        let range = match (tail, max_chars) {
            (true, Some(chars)) => TextRange::Last { chars },
            (true, None) => return Err(PyValueError::new_err("tail reads need max_chars")),
            (false, max_chars) => TextRange::From { offset, max_chars },
        };
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader
                    .span_text(
                        &store, &scope, &trace_id, &trace_ref, span_ids, part, range, contains,
                    )
                    .await
            },
            map_read_error,
        )
    }

    #[pyo3(signature = (scope, start_ms, end_ms, q, buckets, as_of_ms=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "one parameter per Python argument"
    )]
    fn trace_histogram<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        q: &str,
        buckets: u32,
        as_of_ms: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let window = run_window(start_ms, end_ms, as_of_ms, None)?;
        let filter = run_filter(window, q, Vec::new())?;
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader.histogram(&store, &scope, &filter, buckets).await
            },
            map_read_error,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one parameter per Python argument"
    )]
    #[pyo3(signature = (scope, start_ms, end_ms, q, field, contains, limit, as_of_ms=None))]
    fn run_values<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        q: &str,
        field: &str,
        contains: &str,
        limit: u32,
        as_of_ms: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let field = field
            .parse::<RunField>()
            .map_err(|_| PyValueError::new_err(format!("unknown run field {field}")))?;
        let window = run_window(start_ms, end_ms, as_of_ms, None)?;
        let filter = run_filter(window, q, Vec::new())?;
        let contains = contains.to_owned();
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader
                    .values(&store, &scope, &filter, field, &contains, limit)
                    .await
            },
            map_read_error,
        )
    }

    #[pyo3(signature = (trace_id, scope, trace_ref, cursor=None, page_size=None))]
    fn get_trace<'py>(
        &self,
        py: Python<'py>,
        trace_id: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        trace_ref: String,
        cursor: Option<String>,
        page_size: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                if let Some(page_size) = page_size {
                    reader
                        .get_trace_page(
                            &store,
                            &scope,
                            &trace_id,
                            &trace_ref,
                            cursor.as_deref(),
                            page_size,
                        )
                        .await
                } else if cursor.is_some() {
                    Err(ReadError::InvalidParameters)
                } else {
                    reader
                        .get_trace(&store, &scope, &trace_id, &trace_ref)
                        .await
                }
            },
            map_read_error,
        )
    }

    fn get_span<'py>(
        &self,
        py: Python<'py>,
        trace_id: String,
        span_id: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        trace_ref: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader
                    .get_span(&store, &scope, &trace_id, &span_id, &trace_ref)
                    .await
            },
            map_read_error,
        )
    }

    #[pyo3(signature = (trace_id, span_id, scope, trace_ref, cursor))]
    fn get_span_error<'py>(
        &self,
        py: Python<'py>,
        trace_id: String,
        span_id: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        trace_ref: String,
        cursor: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader
                    .get_span_error(
                        &store,
                        &scope,
                        &trace_id,
                        &span_id,
                        &trace_ref,
                        cursor.as_deref(),
                    )
                    .await
            },
            map_read_error,
        )
    }

    fn get_trace_metadata<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader.get_trace_metadata(&store, &scope, &id).await
            },
            map_read_error,
        )
    }

    #[pyo3(signature = (scope, id, cursor, page_size))]
    fn get_trace_spans<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        id: String,
        cursor: Option<String>,
        page_size: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader
                    .get_trace_spans(&store, &scope, &id, cursor.as_deref(), page_size)
                    .await
            },
            map_read_error,
        )
    }

    fn get_span_by_id<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        id: String,
        span_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader.get_span_by_id(&store, &scope, &id, &span_id).await
            },
            map_read_error,
        )
    }

    #[pyo3(signature = (scope, id, span_id, cursor=None))]
    fn get_span_error_by_id<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        id: String,
        span_id: String,
        cursor: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.config.storage().reader().clone();
        let reader = Arc::clone(&self.reader);
        crate::execution::run_async(
            py,
            async move {
                let store = ClickHouseTraces::new(client, connection);
                reader
                    .get_span_error_by_id(&store, &scope, &id, &span_id, cursor.as_deref())
                    .await
            },
            map_read_error,
        )
    }

    fn query_sql<'py>(
        &self,
        py: Python<'py>,
        sql: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] scope: QueryScope,
        secret: String,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] params: BTreeMap<
            String,
            litellm_traces::api::SqlParameter,
        >,
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
                litellm_traces_clickhouse::query_sql_with_params(
                    &client,
                    &connection,
                    &sql,
                    &params,
                )
                .await
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
    use rstest::rstest;

    use super::*;

    fn initialize_python_path(py: Python<'_>) {
        let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .to_str()
            .unwrap();
        pyo3::types::PyModule::import(py, "sys")
            .unwrap()
            .getattr("path")
            .unwrap()
            .call_method1("insert", (0, repository))
            .unwrap();
    }

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
    #[case::syntax(500, Some(62), "rejected")]
    #[case::unknown_column(500, Some(47), "rejected")]
    #[case::readonly(500, Some(164), "rejected")]
    #[case::denied_table(403, Some(497), "rejected")]
    #[case::settings_constraint(500, Some(452), "rejected")]
    #[case::memory_limit(500, Some(241), "limited")]
    #[case::timeout(408, Some(159), "limited")]
    #[case::result_limit(500, Some(396), "limited")]
    #[case::invalid_sql_status(400, None, "rejected")]
    #[case::unavailable(503, None, "unavailable")]
    #[case::invalid_reader_credentials(403, Some(516), "unavailable")]
    fn raw_query_failures_preserve_diagnostics_and_curated_exception_type(
        #[case] status: u16,
        #[case] code: Option<u32>,
        #[case] kind: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            initialize_python_path(py);
            let error = Error::Storage(litellm_storage_clickhouse::Error::QueryFailed(
                litellm_storage_clickhouse::QueryFailure {
                    status,
                    code,
                    message: "engine diagnostic".into(),
                },
            ));
            let curated = map_error_ref(&error);
            assert_eq!(curated.get_type(py).name().unwrap(), "RuntimeError");
            let exception = map_sql_error(error);
            assert_eq!(exception.get_type(py).name().unwrap(), "TraceQueryError");
            assert_eq!(
                exception
                    .value(py)
                    .getattr("kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                kind
            );
            assert_eq!(
                exception
                    .value(py)
                    .getattr("database_code")
                    .unwrap()
                    .extract::<Option<u32>>()
                    .unwrap(),
                code
            );
            assert_eq!(
                exception.value(py).str().unwrap().to_str().unwrap(),
                "engine diagnostic"
            );
        });
    }

    #[rstest]
    fn raw_query_transport_failures_preserve_generic_exception_type() {
        Python::initialize();
        Python::attach(|py| {
            let exception =
                map_sql_error(Error::Storage(litellm_storage_clickhouse::Error::Transport));
            assert_eq!(exception.get_type(py).name().unwrap(), "RuntimeError");
        });
    }

    #[rstest]
    #[case::decode_budget(Error::Decode(litellm_traces::Error::TooLarge), "OverflowError")]
    #[case::invalid_export(Error::Decode(litellm_traces::Error::InvalidPayload), "ValueError")]
    #[case::invalid_decode_limit(
        Error::Decode(litellm_traces::Error::InvalidLimit("OTLP_MAX_SPANS")),
        "ValueError"
    )]
    fn trace_ingest_failures_preserve_public_exception_types(
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

    #[rstest]
    #[case::invalid_parameters(ReadError::InvalidParameters, "ValueError")]
    #[case::invalid_cursor(ReadError::InvalidCursor("trace"), "ValueError")]
    #[case::ambiguous(ReadError::AmbiguousTrace, "ValueError")]
    #[case::changed_snapshot(ReadError::TraceChanged, "TraceChanged")]
    #[case::read_budget(ReadError::TooLarge, "OverflowError")]
    #[case::encode(
        ReadError::Encode(Arc::new(serde_json::Error::io(std::io::Error::other("invalid")))),
        "RuntimeError"
    )]
    #[case::store(ReadError::Store(Arc::new(Error::InvalidScope)), "ValueError")]
    fn trace_read_failures_preserve_public_exception_types(
        #[case] error: ReadError<Error>,
        #[case] exception_name: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            initialize_python_path(py);
            let message = error.to_string();
            let exception = map_read_error(error);
            assert_eq!(exception.get_type(py).name().unwrap(), exception_name);
            assert_eq!(
                exception.value(py).str().unwrap().to_str().unwrap(),
                message
            );
        });
    }
}
