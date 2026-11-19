use std::sync::Arc;

use litellm_repositories::CustomerMemberRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The customer member repository. `find_with_budgets` answers a customer member with its budget and budget override
/// as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustCustomerMemberRepository(Arc<CustomerMemberRepository>);

#[pymethods]
impl RustCustomerMemberRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(CustomerMemberRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find_with_budgets<'py>(
        &self,
        py: Python<'py>,
        customer_member_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let customer_member = repository
                    .find_with_budgets(&customer_member_id)
                    .await
                    .map_err(map_error)?;
                super::json(customer_member.as_ref())
            })?,
        )
    }
}
