use std::sync::Arc;

use jiff::Timestamp;
use litellm_repositories::{AuditLogRepository, NewAuditLog};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::value::RawValue;

use crate::db::{NativeDatabase, map_error};

/// One audit log row as Python passes it: `(id, updated_at, changed_by, changed_by_api_key, action, table_name,
/// object_id, before_value, updated_values)`, with the time in microseconds since the epoch and the two values
/// as JSON text.
type AuditLogRow = (
    String,
    i64,
    Option<String>,
    Option<String>,
    String,
    String,
    String,
    Option<String>,
    Option<String>,
);

fn json_document(column: &str, text: Option<String>) -> PyResult<Option<Box<RawValue>>> {
    text.map(RawValue::from_string)
        .transpose()
        .map_err(|error| PyValueError::new_err(format!("{column} is not JSON: {error}")))
}

fn new_audit_log(row: AuditLogRow) -> PyResult<NewAuditLog> {
    let (
        id,
        updated_at,
        changed_by,
        changed_by_api_key,
        action,
        table_name,
        object_id,
        before_value,
        updated_values,
    ) = row;
    Ok(NewAuditLog {
        id,
        updated_at: Timestamp::from_microsecond(updated_at).map_err(|error| {
            PyValueError::new_err(format!("updated_at is out of range: {error}"))
        })?,
        changed_by,
        changed_by_api_key,
        action,
        table_name,
        object_id,
        before_value: json_document("before_value", before_value)?,
        updated_values: json_document("updated_values", updated_values)?,
    })
}

/// The audit log repository.
#[pyclass(frozen)]
pub(crate) struct RustAuditLogRepository(Arc<AuditLogRepository>);

#[pymethods]
impl RustAuditLogRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(AuditLogRepository::new(
            database.repository_database()?,
        ))))
    }

    fn insert<'py>(&self, py: Python<'py>, row: AuditLogRow) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let row = new_audit_log(row)?;
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository.insert(&row).await.map_err(map_error)
            })?,
        )
    }

    /// Inserts every row in one statement, so either all of them are stored or none is.
    fn insert_many<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<AuditLogRow>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let rows = rows
            .into_iter()
            .map(new_audit_log)
            .collect::<PyResult<Vec<_>>>()?;
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository.insert_many(&rows).await.map_err(map_error)
            })?,
        )
    }
}
