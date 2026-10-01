use std::collections::{BTreeMap, HashMap};

use litellm_http::ClientVariant;
use litellm_traces::{Connection, Error, InsertTable, Parameter, ReadQuery, Shared};
use prost::Message;
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList, PyMapping, PyString},
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
    #[pyo3(signature = (database, url, reader_url = None))]
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
        #[pyo3(from_py_with = insert_rows_from_py)] rows: Vec<litellm_traces::InsertRow>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let table = InsertTable::parse(table).map_err(map_error)?;
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.writer.clone();
        let database = self.database.clone();
        crate::execution::run_async(
            py,
            async move {
                litellm_traces::insert_shared_rows(&client, &connection, &database, table, rows)
                    .await
            },
            map_error,
        )
    }

    fn lens_query<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] parameters: BTreeMap<
            String,
            Parameter,
        >,
    ) -> PyResult<Bound<'py, PyAny>> {
        let query = litellm_traces::LensQuery::parse(name).map_err(map_error)?;
        let connection = self.reader.clone().ok_or_else(|| {
            PyRuntimeError::new_err("Trace reads require a separate ClickHouse reader URL")
        })?;
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        crate::execution::run_async(
            py,
            async move {
                litellm_traces::execute_read(&client, &connection, query.sql(), &parameters).await
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
) -> PyResult<Bound<'py, PyAny>> {
    let spans = py
        .detach(|| litellm_traces::decode_otlp(body, content_type))
        .map_err(|error| match error {
            litellm_traces::DecodeError::TooLarge => PyOverflowError::new_err(error.to_string()),
            _ => PyValueError::new_err(error.to_string()),
        })?;
    spans_to_py(py, &spans).map(Bound::into_any)
}

fn insert_rows_from_py(value: &Bound<'_, PyAny>) -> PyResult<Vec<litellm_traces::InsertRow>> {
    let mut resources: HashMap<usize, (Bound<'_, PyAny>, Shared<serde_json::Value>)> =
        HashMap::new();
    value
        .try_iter()?
        .map(|row| {
            let row = row?;
            let mut fields = BTreeMap::new();
            for item in row.cast::<PyMapping>()?.items()?.iter() {
                let (key, value): (String, Bound<'_, PyAny>) = item.extract()?;
                let converted = if matches!(
                    key.as_str(),
                    "ResourceAttributes" | "ScopeName" | "ScopeVersion"
                ) {
                    let identity = value.as_ptr() as usize;
                    match resources.entry(identity) {
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            Shared::clone(&entry.get().1)
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let converted = Shared::new(litellm_host_python::from_py_argument::<
                                serde_json::Value,
                            >(&value)?);
                            entry.insert((value, Shared::clone(&converted)));
                            converted
                        }
                    }
                } else {
                    Shared::new(litellm_host_python::from_py_argument(&value)?)
                };
                fields.insert(key, converted);
            }
            Ok(fields)
        })
        .collect()
}

fn spans_to_py<'py>(
    py: Python<'py>,
    spans: &[litellm_traces::DecodedSpan],
) -> PyResult<Bound<'py, PyList>> {
    let mut resources = HashMap::new();
    let mut scopes = HashMap::new();
    let result = PyList::empty(py);
    for span in spans {
        let identity = span.resource_attributes.identity();
        let resource = match resources.entry(identity) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                litellm_host_python::Pythonized(&*span.resource_attributes).into_pyobject(py)?,
            ),
        };
        let row = PyDict::new(py);
        row.set_item("trace_id", &span.trace_id)?;
        row.set_item("span_id", &span.span_id)?;
        row.set_item("parent_span_id", &span.parent_span_id)?;
        row.set_item("trace_state", &span.trace_state)?;
        row.set_item("name", &span.name)?;
        row.set_item("kind", &span.kind)?;
        row.set_item("resource_attributes", &*resource)?;
        for (key, value) in [
            ("scope_name", &span.scope_name),
            ("scope_version", &span.scope_version),
        ] {
            let value = scopes
                .entry(value.identity())
                .or_insert_with(|| PyString::new(py, value));
            row.set_item(key, &*value)?;
        }
        row.set_item("attributes", &span.attributes)?;
        row.set_item("start_ns", span.start_ns)?;
        row.set_item("end_ns", span.end_ns)?;
        row.set_item("status_code", &span.status_code)?;
        row.set_item("status_message", &span.status_message)?;
        row.set_item(
            "events",
            litellm_host_python::Pythonized(&span.events).into_pyobject(py)?,
        )?;
        result.append(row)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn insert_projection_preserves_identity_without_merging_equal_resources() {
        Python::initialize();
        Python::attach(|py| {
            let resource = PyDict::new(py);
            resource.set_item("service.name", "shared").unwrap();
            let equal_resource = resource.copy().unwrap();
            let rows = PyList::empty(py);
            for value in [&resource, &resource, &equal_resource] {
                let row = PyDict::new(py);
                row.set_item("ResourceAttributes", value).unwrap();
                rows.append(row).unwrap();
            }
            let projected = insert_rows_from_py(rows.as_any()).unwrap();
            assert!(Shared::shares_storage_with(
                &projected[0]["ResourceAttributes"],
                &projected[1]["ResourceAttributes"]
            ));
            assert!(!Shared::shares_storage_with(
                &projected[0]["ResourceAttributes"],
                &projected[2]["ResourceAttributes"]
            ));
            assert_eq!(projected[0], projected[2]);
        });
    }

    #[rstest]
    fn shared_conversion_preserves_every_decoded_field() {
        Python::initialize();
        Python::attach(|py| {
            let spans = litellm_traces::decode_otlp(
                include_bytes!("../../../../../tests/test_litellm/tracing/fixtures/langsmith_deep_agent_export.json"),
                Some("application/json"),
            ).unwrap();
            let expected = litellm_host_python::Pythonized(&spans)
                .into_pyobject(py)
                .unwrap();
            let actual = spans_to_py(py, &spans).unwrap();
            assert!(actual.eq(expected).unwrap());
        });
    }

    #[rstest]
    #[ignore = "set OTLP_BENCH_BODY to a JSON export and run with --ignored --nocapture"]
    fn profile_decode_and_python_conversion() {
        let body = std::fs::read(std::env::var("OTLP_BENCH_BODY").unwrap()).unwrap();
        Python::initialize();
        Python::attach(|py| {
            for _ in 0..5 {
                let start = std::time::Instant::now();
                let spans = py
                    .detach(|| litellm_traces::decode_otlp(&body, Some("application/json")))
                    .unwrap();
                let decoded = start.elapsed();
                let start = std::time::Instant::now();
                let converted = spans_to_py(py, &spans).unwrap();
                let converted_in = start.elapsed();
                assert_eq!(converted.len(), spans.len());
                eprintln!(
                    "decode_us={} python_us={}",
                    decoded.as_micros(),
                    converted_in.as_micros()
                );
            }
        });
    }
}
