use std::sync::Arc;

use litellm_repositories::OrganizationRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The organization repository. `find_with_optional_budget` answers an organization, with its budget on request, as JSON text, or
/// `None`; `find_up_to_two_by_alias` answers up to two organizations sharing an alias, without budgets, as JSON text each.
#[pyclass(frozen)]
pub(crate) struct RustOrganizationRepository(Arc<OrganizationRepository>);

#[pymethods]
impl RustOrganizationRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(OrganizationRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (organization_id, *, include_budget_table=false))]
    fn find_with_optional_budget<'py>(
        &self,
        py: Python<'py>,
        organization_id: String,
        include_budget_table: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let organization = repository
                    .find_with_optional_budget(&organization_id, include_budget_table)
                    .await
                    .map_err(map_error)?;
                super::json(organization.as_ref())
            })?,
        )
    }

    fn find_up_to_two_by_alias<'py>(
        &self,
        py: Python<'py>,
        alias: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let found = repository
                    .find_up_to_two_by_alias(&alias)
                    .await
                    .map_err(map_error)?;
                super::json_list(&found)
            })?,
        )
    }
}
