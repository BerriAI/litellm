use std::sync::Arc;

use litellm_repositories::{Side, TeamRepository};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The team repository. `find_with_model_aliases` answers a team with its model aliases as JSON text, or `None`; `find_up_to_two_by_alias` answers
/// up to two teams sharing an alias, as JSON text each.
#[pyclass(frozen)]
pub(crate) struct RustTeamRepository(Arc<TeamRepository>);

#[pymethods]
impl RustTeamRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(TeamRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (team_id, *, use_writer=false))]
    fn find_with_model_aliases<'py>(
        &self,
        py: Python<'py>,
        team_id: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let side = if use_writer { Side::Writer } else { Side::Read };
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let team = repository
                    .find_with_model_aliases(&team_id, side)
                    .await
                    .map_err(map_error)?;
                super::json(team.as_ref())
            })?,
        )
    }

    fn find_up_to_two_by_alias<'py>(
        &self,
        py: Python<'py>,
        alias: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let found = repository
                    .find_up_to_two_by_alias(&alias)
                    .await
                    .map_err(map_error)?;
                super::json_list(&found)
            })?,
        )
    }
}
