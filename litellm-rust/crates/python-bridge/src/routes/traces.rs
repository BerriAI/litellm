use std::collections::BTreeMap;

use litellm_http::ClientVariant;
use litellm_traces::{Connection, Error, Parameter};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

fn map_error(error: Error) -> PyErr {
    match error {
        Error::InvalidSchema | Error::EmptySql => PyValueError::new_err(error.to_string()),
        Error::InvalidUrl
        | Error::QueryFailed(_)
        | Error::ResponseTooLarge
        | Error::InvalidResponse
        | Error::Transport => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyfunction]
pub fn trace_schema_statements(
    py: Python<'_>,
    database: String,
    trace_retention_days: u32,
    spend_log_retention_days: u32,
) -> PyResult<Vec<String>> {
    py.detach(|| {
        litellm_traces::schema_statements(&database, trace_retention_days, spend_log_retention_days)
    })
    .map_err(map_error)
}

#[pyfunction]
pub fn trace_query<'py>(
    py: Python<'py>,
    url: &str,
    database: &str,
    user: &str,
    password: &str,
    sql: String,
    #[pyo3(from_py_with = litellm_host_python::from_py_argument)] parameters: BTreeMap<
        String,
        Parameter,
    >,
) -> PyResult<Bound<'py, PyAny>> {
    let connection = Connection::configured(url, database, user, password).map_err(map_error)?;
    let client = crate::http::host_client(py, ClientVariant::NoRedirect)?;
    litellm_host_python::run_async(
        py,
        async move { litellm_traces::execute_read(&client, &connection, &sql, &parameters).await },
        map_error,
    )
}
