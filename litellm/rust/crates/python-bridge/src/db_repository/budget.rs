use std::sync::Arc;

use litellm_repositories::{BudgetRepository, Side};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The budget repository. Each lookup answers the budget row as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustBudgetRepository(Arc<BudgetRepository>);

#[pymethods]
impl RustBudgetRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(BudgetRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find_end_user_default_budget<'py>(
        &self,
        py: Python<'py>,
        budget_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                super::json(
                    repository
                        .find_end_user_default(&budget_id)
                        .await
                        .map_err(map_error)?
                        .as_ref(),
                )
            })?,
        )
    }

    #[pyo3(signature = (budget_id, *, use_writer=false))]
    fn find_team_member_default_budget<'py>(
        &self,
        py: Python<'py>,
        budget_id: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let side = if use_writer { Side::Writer } else { Side::Read };
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let budget = repository
                    .find_team_member_default(&budget_id, side)
                    .await
                    .map_err(map_error)?;
                super::json(budget.as_ref())
            })?,
        )
    }
}
