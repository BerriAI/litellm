use std::collections::BTreeMap;

use litellm_host_python::{FromPythonCache, ToPythonCache};
use litellm_http::ClientVariant;
use litellm_storage_clickhouse::Storage;
use litellm_traces::{Error, InsertTable, Parameter, ReadQuery, Shared};
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
    storage: Storage,
}

#[pymethods]
impl NativeTraceStorage {
    #[new]
    #[pyo3(signature = (database, url, reader_url = None))]
    fn new(database: String, url: &str, reader_url: Option<&str>) -> PyResult<Self> {
        litellm_traces::schema_statements(&database, 1, 1).map_err(map_error)?;
        Ok(Self {
            storage: Storage::new(database, url, reader_url).map_err(map_error)?,
        })
    }

    fn ensure_schema<'py>(
        &self,
        py: Python<'py>,
        trace_retention_days: u32,
        spend_log_retention_days: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        let connection = self.storage.writer().clone();
        let database = self.storage.database().to_owned();
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
        let connection = self.storage.writer().clone();
        let database = self.storage.database().to_owned();
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
        let connection = self.storage.reader().cloned().ok_or_else(|| {
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
        let connection = self.storage.reader().cloned().ok_or_else(|| {
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
    let mut resources = FromPythonCache::default();
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
                    resources
                        .get_or_try_insert_with(&value, |value| {
                            litellm_host_python::from_py_argument::<serde_json::Value>(value)
                                .map(Shared::new)
                        })?
                        .clone()
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
    let mut resources = ToPythonCache::default();
    let mut scopes = ToPythonCache::default();
    let result = PyList::empty(py);
    for span in spans {
        let resource = resources
            .get_or_try_insert_with(span.resource_attributes.as_ref(), |value| {
                litellm_host_python::Pythonized(value).into_pyobject(py)
            })?;
        let row = PyDict::new(py);
        row.set_item("trace_id", &span.trace_id)?;
        row.set_item("span_id", &span.span_id)?;
        row.set_item("parent_span_id", &span.parent_span_id)?;
        row.set_item("trace_state", &span.trace_state)?;
        row.set_item("name", &span.name)?;
        row.set_item("kind", &span.kind)?;
        row.set_item("resource_attributes", resource)?;
        for (key, value) in [
            ("scope_name", &span.scope_name),
            ("scope_version", &span.scope_version),
        ] {
            let value = scopes.get_or_try_insert_with(value.as_ref(), |value| {
                Ok(PyString::new(py, value).into_any())
            })?;
            row.set_item(key, value)?;
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
        row.set_item(
            "normalized",
            litellm_host_python::Pythonized(&span.normalized).into_pyobject(py)?,
        )?;
        row.set_item(
            "consumed_attributes",
            litellm_host_python::Pythonized(&span.consumed_attributes).into_pyobject(py)?,
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
}

#[pyfunction]
pub fn trace_normalized_field_definitions<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    litellm_host_python::Pythonized(litellm_traces::NORMALIZED_FIELD_DEFINITIONS).into_pyobject(py)
}
