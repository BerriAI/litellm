//! What a route tells the v1 contract about itself, and the one constructor callers get.

use pyo3::prelude::*;

use crate::adapter::V1PythonLifecycle;
use crate::subscribers::snapshot;

#[derive(Clone, Copy, Debug)]
pub struct V1PythonSurface {
    pub call_type: &'static str,
}

impl V1PythonLifecycle {
    /// The lifecycle for one call, over the subscribers registered right now; later
    /// registrations do not reach that call. `None` when nothing is subscribed, so an
    /// unobserved call builds no envelope at all.
    pub fn subscribed(
        py: Python<'_>,
        surface: V1PythonSurface,
        asynchronous: bool,
    ) -> PyResult<Option<Self>> {
        let subscribers = snapshot(py, asynchronous)?;
        Ok((!subscribers.is_empty()).then(|| Self::new(surface, subscribers, asynchronous)))
    }
}
