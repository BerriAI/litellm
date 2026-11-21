use std::sync::Arc;

use litellm_repositories::ObjectPermissionRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The object permission repository. `find_by_object_permission_id` answers an object permission as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustObjectPermissionRepository(Arc<ObjectPermissionRepository>);

#[pymethods]
impl RustObjectPermissionRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(ObjectPermissionRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (object_permission_id, *, use_writer=false))]
    fn find_by_object_permission_id<'py>(
        &self,
        py: Python<'py>,
        object_permission_id: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let side = super::side(use_writer);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let permission = repository
                    .find_by_object_permission_id(&object_permission_id, side)
                    .await
                    .map_err(map_error)?;
                super::json(permission.as_ref())
            })?,
        )
    }
}
