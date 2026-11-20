use std::sync::Arc;

use litellm_repositories::TagRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The tag repository. `names` lists tag names for the registry; `find_many_with_budgets` answers the tags that exist, each with
/// its budget, as JSON text.
#[pyclass(frozen)]
pub(crate) struct RustTagRepository(Arc<TagRepository>);

#[pymethods]
impl RustTagRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(TagRepository::new(
            database.repository_database()?,
        ))))
    }

    /// Up to `limit` tag names.
    fn names<'py>(&self, py: Python<'py>, limit: i64) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository.names(limit).await.map_err(map_error)
            })?,
        )
    }

    fn find_many_with_budgets<'py>(
        &self,
        py: Python<'py>,
        names: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let tags = repository
                    .find_many_with_budgets(&names)
                    .await
                    .map_err(map_error)?;
                tags.iter()
                    .map(litellm_host_python::json_dumps)
                    .collect::<PyResult<Vec<_>>>()
            })?,
        )
    }
}
