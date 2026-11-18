use std::sync::Arc;

use litellm_repositories::AppCustomerMemberRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The app customer member repository. `find` answers a customer member's row within an app as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustAppCustomerMemberRepository(Arc<AppCustomerMemberRepository>);

#[pymethods]
impl RustAppCustomerMemberRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(AppCustomerMemberRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find<'py>(
        &self,
        py: Python<'py>,
        app_id: String,
        customer_member_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let member = repository
                    .find(&app_id, &customer_member_id)
                    .await
                    .map_err(map_error)?;
                super::json(member.as_ref())
            })?,
        )
    }
}
