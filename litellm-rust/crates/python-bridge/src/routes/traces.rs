use std::collections::BTreeMap;

use litellm_http::ClientVariant;
use litellm_traces::{Connection, Error, InsertTable, Parameter, ReadQuery};
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
};

fn map_error(error: Error) -> PyErr {
    match error {
        Error::InvalidRow
        | Error::InvalidTable
        | Error::InvalidSchema
        | Error::EmptySql
        | Error::InvalidQuery => PyValueError::new_err(error.to_string()),
        Error::InsertTooLarge => PyOverflowError::new_err(error.to_string()),
        Error::InvalidUrl
        | Error::QueryFailed(_)
        | Error::InsertFailed(_)
        | Error::SchemaFailed(_)
        | Error::ResponseTooLarge
        | Error::InvalidResponse
        | Error::Transport => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyclass]
pub struct NativeTraceStorage {
    database: String,
    writer: Connection,
    reader: Option<Connection>,
}

#[pymethods]
impl NativeTraceStorage {
    #[new]
    #[pyo3(signature = (database, url, reader_url=None))]
    fn new(database: String, url: &str, reader_url: Option<&str>) -> PyResult<Self> {
        litellm_traces::schema_statements(&database, 1, 1).map_err(map_error)?;
        Ok(Self {
            writer: Connection::writer(url).map_err(map_error)?,
            reader: reader_url
                .map(|value| Connection::reader(value, &database))
                .transpose()
                .map_err(map_error)?,
            database,
        })
    }

    fn ensure_schema<'py>(
        &self,
        py: Python<'py>,
        trace_retention_days: u32,
        spend_log_retention_days: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.writer.clone();
        let database = self.database.clone();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces::ensure_schema(
                    &client,
                    &connection,
                    &database,
                    trace_retention_days,
                    spend_log_retention_days,
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
        let connection = self.writer.clone();
        let database = self.database.clone();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces::insert_rows(&client, &connection, &database, table, rows).await
            },
            map_error,
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
        let query = ReadQuery::parse(query).map_err(map_error)?;
        let connection = self.reader.clone().ok_or_else(|| {
            PyRuntimeError::new_err("Trace reads require a separate ClickHouse reader URL")
        })?;
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        crate::execution::run_async(
            py,
            async move {
                litellm_traces::execute_named_read(&client, &connection, query, &parameters).await
            },
            map_error,
        )
    }
}

#[pyfunction]
pub fn trace_decode_otlp<'py>(
    py: Python<'py>,
    body: &[u8],
    content_type: Option<&str>,
    content_encoding: Option<&str>,
    max_decompressed_bytes: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let spans = py
        .detach(|| {
            litellm_traces::decode_otlp(
                body,
                content_type,
                content_encoding,
                max_decompressed_bytes,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    litellm_host_python::Pythonized(spans).into_pyobject(py)
}
