use std::sync::Arc;

use litellm_repositories::AppRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The app repository. `find` answers an app row as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustAppRepository(Arc<AppRepository>);

#[pymethods]
impl RustAppRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(AppRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find<'py>(&self, py: Python<'py>, app_id: String) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let app = repository.find(&app_id).await.map_err(map_error)?;
                super::json(app.as_ref())
            })?,
        )
    }
}
