use std::sync::Arc;

use litellm_repositories::{KeyOwnerIds, KeyOwnersRepository};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The key owners repository. `find` answers the user, team, membership, organization and project rows as JSON
/// text of one object keyed `user_row` through `project_row`, each `None` when not asked for or missing.
#[pyclass(frozen)]
pub(crate) struct RustKeyOwnersRepository(Arc<KeyOwnersRepository>);

#[pymethods]
impl RustKeyOwnersRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(KeyOwnersRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (*, user_id, team_id, membership_user_id, organization_id, project_id))]
    fn find<'py>(
        &self,
        py: Python<'py>,
        user_id: Option<String>,
        team_id: Option<String>,
        membership_user_id: Option<String>,
        organization_id: Option<String>,
        project_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let owners = repository
                    .find(KeyOwnerIds {
                        user_id: user_id.as_deref(),
                        team_id: team_id.as_deref(),
                        membership_user_id: membership_user_id.as_deref(),
                        organization_id: organization_id.as_deref(),
                        project_id: project_id.as_deref(),
                    })
                    .await
                    .map_err(map_error)?;
                litellm_host_python::json_dumps(&owners)
            })?,
        )
    }
}
