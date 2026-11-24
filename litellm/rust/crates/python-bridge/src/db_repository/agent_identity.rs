use std::sync::Arc;

use litellm_repositories::AgentIdentityRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The agent identity repository.
#[pyclass(frozen)]
pub(crate) struct RustAgentIdentityRepository(Arc<AgentIdentityRepository>);

#[pymethods]
impl RustAgentIdentityRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(AgentIdentityRepository::new(
            database.repository_database()?,
        ))))
    }

    /// Whether the identity of `agent_id` was stamped as authenticated now. False when it is no longer the
    /// active one at `revision`, or its agent is disabled or no longer identity managed.
    fn record_authentication<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        revision: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository
                    .record_authentication(&agent_id, &revision)
                    .await
                    .map_err(map_error)
            })?,
        )
    }
}
