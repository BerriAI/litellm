use std::sync::Arc;

use litellm_repositories::{Side, TeamMembershipRepository};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The team membership repository. `find_with_budget` answers a membership with its member budget as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustTeamMembershipRepository(Arc<TeamMembershipRepository>);

#[pymethods]
impl RustTeamMembershipRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(TeamMembershipRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (user_id, team_id, *, use_writer=false))]
    fn find_with_budget<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        team_id: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let side = if use_writer { Side::Writer } else { Side::Read };
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let membership = repository
                    .find_with_budget(&user_id, &team_id, side)
                    .await
                    .map_err(map_error)?;
                super::json(membership.as_ref())
            })?,
        )
    }
}
