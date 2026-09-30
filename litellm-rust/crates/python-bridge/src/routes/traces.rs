use litellm_traces::{Connection, Error, ListQuery, Query};
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyAny,
};

fn map_error(error: Error) -> PyErr {
    match error {
        Error::SchemaPending => PyNotImplementedError::new_err(error.to_string()),
        Error::InvalidUrl => PyRuntimeError::new_err(error.to_string()),
        Error::InvalidListQuery | Error::InvalidIdentifier | Error::EmptySql => {
            PyValueError::new_err(error.to_string())
        }
        Error::QueryFailed(_)
        | Error::ResponseTooLarge
        | Error::InvalidResponse
        | Error::Transport => PyRuntimeError::new_err(error.to_string()),
    }
}

fn dispatch<'py>(
    py: Python<'py>,
    clickhouse_url: &str,
    request: Query,
) -> PyResult<Bound<'py, PyAny>> {
    let clickhouse_url = clickhouse_url.to_owned();
    litellm_host_python::run_async(
        py,
        async move {
            let connection = Connection::parse(&clickhouse_url)?;
            litellm_traces::query(connection, request).await
        },
        map_error,
    )
}

#[pyfunction]
#[pyo3(signature = (clickhouse_url, start_ms, end_ms, service=None, status=None, search=None, cursor=None, limit=50))]
#[expect(
    clippy::too_many_arguments,
    reason = "the list API has seven typed query fields"
)]
pub fn list_traces<'py>(
    py: Python<'py>,
    clickhouse_url: &str,
    start_ms: i64,
    end_ms: i64,
    service: Option<String>,
    status: Option<String>,
    search: Option<String>,
    cursor: Option<String>,
    limit: u8,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch(
        py,
        clickhouse_url,
        Query::List(ListQuery {
            start_ms,
            end_ms,
            service,
            status,
            search,
            cursor,
            limit,
        }),
    )
}

#[pyfunction]
pub fn get_trace<'py>(
    py: Python<'py>,
    clickhouse_url: &str,
    trace_id: String,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch(py, clickhouse_url, Query::Trace { trace_id })
}

#[pyfunction]
pub fn get_span<'py>(
    py: Python<'py>,
    clickhouse_url: &str,
    trace_id: String,
    span_id: String,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch(py, clickhouse_url, Query::Span { trace_id, span_id })
}
