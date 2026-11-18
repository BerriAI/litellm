use std::sync::Arc;

use litellm_repositories::CustomerOrganizationRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The customer organization repository. `find_with_budget` answers a customer organization with its budget as JSON
/// text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustCustomerOrganizationRepository(Arc<CustomerOrganizationRepository>);

#[pymethods]
impl RustCustomerOrganizationRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(CustomerOrganizationRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find_with_budget<'py>(
        &self,
        py: Python<'py>,
        customer_org_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let customer_organization = repository
                    .find_with_budget(&customer_org_id)
                    .await
                    .map_err(map_error)?;
                super::json(customer_organization.as_ref())
            })?,
        )
    }
}
