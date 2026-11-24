use std::sync::Arc;

use litellm_repositories::AppCustomerOrganizationRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The app customer organization repository. `find` answers a customer organization's row within an app as JSON
/// text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustAppCustomerOrganizationRepository(Arc<AppCustomerOrganizationRepository>);

#[pymethods]
impl RustAppCustomerOrganizationRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(AppCustomerOrganizationRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find<'py>(
        &self,
        py: Python<'py>,
        app_id: String,
        customer_org_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let organization = repository
                    .find(&app_id, &customer_org_id)
                    .await
                    .map_err(map_error)?;
                super::json(organization.as_ref())
            })?,
        )
    }
}
