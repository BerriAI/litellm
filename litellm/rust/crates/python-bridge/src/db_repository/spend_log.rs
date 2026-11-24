use std::sync::Arc;

use chrono::{DateTime, Utc};
use litellm_repositories::{SpendLogOwner, SpendLogRepository};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The spend log repository. Each window sum answers the spend one key, team or user logged since a UTC
/// `datetime`, as a float.
#[pyclass(frozen)]
pub(crate) struct RustSpendLogRepository(Arc<SpendLogRepository>);

impl RustSpendLogRepository {
    fn window_spend<'py>(
        &self,
        py: Python<'py>,
        owner: SpendLogOwner,
        owner_id: String,
        since: DateTime<Utc>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository
                    .window_spend(owner, &owner_id, since)
                    .await
                    .map_err(map_error)
            })?,
        )
    }
}

#[pymethods]
impl RustSpendLogRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(SpendLogRepository::new(
            database.repository_database()?,
        ))))
    }

    fn key_window_spend<'py>(
        &self,
        py: Python<'py>,
        api_key: String,
        since: DateTime<Utc>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.window_spend(py, SpendLogOwner::Key, api_key, since)
    }

    fn team_window_spend<'py>(
        &self,
        py: Python<'py>,
        team_id: String,
        since: DateTime<Utc>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.window_spend(py, SpendLogOwner::Team, team_id, since)
    }

    fn user_window_spend<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        since: DateTime<Utc>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.window_spend(py, SpendLogOwner::User, user_id, since)
    }
}
