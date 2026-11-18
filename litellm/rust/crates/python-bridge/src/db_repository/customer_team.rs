use std::sync::Arc;

use litellm_repositories::CustomerTeamRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The customer team repository. `find_with_budgets` answers a customer team with its budget and budget override as
/// JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustCustomerTeamRepository(Arc<CustomerTeamRepository>);

#[pymethods]
impl RustCustomerTeamRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(CustomerTeamRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find_with_budgets<'py>(
        &self,
        py: Python<'py>,
        customer_team_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let customer_team = repository
                    .find_with_budgets(&customer_team_id)
                    .await
                    .map_err(map_error)?;
                super::json(customer_team.as_ref())
            })?,
        )
    }
}
