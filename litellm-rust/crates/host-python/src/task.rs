use futures_util::future::BoxFuture;
use litellm_coroutine::{Abandoned, Co, Reply};
use pyo3::prelude::*;

use crate::handle::Suspension;
use crate::{Awaiter, missing_state};

pub(crate) type Answer = PyResult<Py<PyAny>>;

pub(crate) struct Yielded {
    pub(crate) suspension: Suspension,
    pub(crate) reply: Reply<Answer>,
}

/// The caller's Python task as the running call sees it. Every suspension is handed to
/// that task and the call continues with what the task resumes it with.
#[derive(Clone)]
pub(crate) struct PythonTask {
    co: Co<Yielded>,
}

impl PythonTask {
    pub(crate) fn new(co: Co<Yielded>) -> Self {
        Self { co }
    }

    pub(crate) async fn suspend(&self, suspension: Suspension) -> Answer {
        match self.co.yield_(|reply| Yielded { suspension, reply }).await {
            Ok(answer) => answer,
            Err(Abandoned) => Err(missing_state()),
        }
    }
}

impl Awaiter for PythonTask {
    fn awaited(&self, awaitable: Py<PyAny>) -> BoxFuture<'_, Answer> {
        Box::pin(self.suspend(Suspension::Await(awaitable)))
    }
}
