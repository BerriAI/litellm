use std::sync::Arc;

use litellm_repositories::ProjectRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The project repository. `find_with_budget` answers a project with its budget as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustProjectRepository(Arc<ProjectRepository>);

#[pymethods]
impl RustProjectRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(ProjectRepository::new(
            database.repository_database()?,
        ))))
    }

    fn find_with_budget<'py>(
        &self,
        py: Python<'py>,
        project_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let project = repository
                    .find_with_budget(&project_id)
                    .await
                    .map_err(map_error)?;
                super::json(project.as_ref())
            })?,
        )
    }
}
