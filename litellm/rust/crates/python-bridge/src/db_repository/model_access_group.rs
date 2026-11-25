use std::sync::Arc;

use litellm_repositories::ModelAccessGroupRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The model access group budget repository.
#[pyclass(frozen)]
pub(crate) struct RustModelAccessGroupRepository(Arc<ModelAccessGroupRepository>);

#[pymethods]
impl RustModelAccessGroupRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(ModelAccessGroupRepository::new(
            database.repository_database()?,
        ))))
    }

    /// Up to `limit` names of groups that have a budget row.
    fn names<'py>(&self, py: Python<'py>, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository.names(limit).await.map_err(map_error)
            })?,
        )
    }

    /// `(access_group_name, spend, max_budget, has_budget_windows)` for each of `names` with a budget row.
    fn budgets<'py>(&self, py: Python<'py>, names: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let budgets = repository.budgets(&names).await.map_err(map_error)?;
                Ok(budgets
                    .into_iter()
                    .map(|budget| {
                        (
                            budget.access_group_name,
                            budget.spend,
                            budget.max_budget,
                            budget.has_budget_windows,
                        )
                    })
                    .collect::<Vec<_>>())
            })?,
        )
    }
}
