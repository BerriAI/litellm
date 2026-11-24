use std::sync::Arc;

use litellm_repositories::BudgetWindowSpendRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The budget window spend repository. `find` answers one window's start and spend as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustBudgetWindowSpendRepository(Arc<BudgetWindowSpendRepository>);

#[pymethods]
impl RustBudgetWindowSpendRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(BudgetWindowSpendRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (entity_type, entity_id, window_duration, *, use_writer=false))]
    fn find<'py>(
        &self,
        py: Python<'py>,
        entity_type: String,
        entity_id: String,
        window_duration: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let side = super::side(use_writer);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let window = repository
                    .find(&entity_type, &entity_id, &window_duration, side)
                    .await
                    .map_err(map_error)?;
                super::json(window.as_ref())
            })?,
        )
    }
}
