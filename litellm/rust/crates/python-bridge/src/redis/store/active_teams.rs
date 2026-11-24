use std::{sync::Arc, time::Duration};

use litellm_redis_store::{
    active_teams::{ActiveMember, ActiveTeams},
    system_clock,
};
use pyo3::prelude::*;

use super::{kind_error, python_clock, redis_async, remote};
use crate::redis::client::NativeRedisClient;

/// One caller active on a model and the priority class it last sent with.
#[pyclass(frozen, get_all, name = "ActiveMember")]
pub(crate) struct PyActiveMember {
    member: String,
    priority_class: String,
}

fn members(listed: Vec<ActiveMember>) -> Vec<PyActiveMember> {
    listed
        .into_iter()
        .map(
            |ActiveMember {
                 member,
                 priority_class,
             }| PyActiveMember {
                member,
                priority_class,
            },
        )
        .collect()
}

/// The callers active on each model, behind `ActiveTeamStore`.
#[pyclass(frozen, name = "NativeActiveTeams")]
pub(crate) struct NativeActiveTeams(Arc<ActiveTeams>);

#[pymethods]
impl NativeActiveTeams {
    #[new]
    #[pyo3(signature = (client, clock=None))]
    fn new(client: Option<PyRef<'_, NativeRedisClient>>, clock: Option<Py<PyAny>>) -> Self {
        Self(Arc::new(ActiveTeams::with_clock(
            client.map(|client| remote(&client)),
            clock.map_or_else(system_clock, python_clock),
        )))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn touch<'py>(
        &self,
        py: Python<'py>,
        model: String,
        member: String,
        priority_class: String,
        window_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let window = Duration::from_secs(window_seconds);
        redis_async(py, "async_register_script", kind_error, async move {
            store
                .touch(&model, &member, &priority_class, window)
                .await
                .map(members)
        })
    }

    /// `touch` without Redis, answered in the caller's own step.
    fn touch_local(
        &self,
        model: &str,
        member: &str,
        priority_class: &str,
        window_seconds: u64,
    ) -> Vec<PyActiveMember> {
        members(self.0.touch_local(
            model,
            member,
            priority_class,
            Duration::from_secs(window_seconds),
        ))
    }
}
