use std::sync::Arc;

use litellm_repositories::EndUserRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The end user repository. `find_with_budget_and_object_permission` answers the row with its budget and object permission as JSON text.
#[pyclass(frozen)]
pub(crate) struct RustEndUserRepository(Arc<EndUserRepository>);

#[pymethods]
impl RustEndUserRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(EndUserRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find_with_budget_and_object_permission<'py>(
        &self,
        py: Python<'py>,
        end_user_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                super::json(
                    repository
                        .find_with_budget_and_object_permission(&end_user_id)
                        .await
                        .map_err(map_error)?
                        .as_ref(),
                )
            })?,
        )
    }

    /// Up to `limit` ids of end users whose row restricts something.
    fn find_restricted_ids<'py>(&self, py: Python<'py>, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository
                    .find_restricted_ids(limit)
                    .await
                    .map_err(map_error)
            })?,
        )
    }
}
