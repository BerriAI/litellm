use litellm_host::{hooks::ResponseHooks, lifecycle::Timing};
use litellm_host_python::{Awaiter, PythonSurface};
use pyo3::prelude::*;

use crate::{CallBoundary, LegacyHooks};

pub(crate) const CALLBACKS: &[(CallBoundary, &[&str])] = &[
    (
        CallBoundary::TransformResponse,
        &["async_post_call_success_deployment_hook"],
    ),
    (CallBoundary::StreamOpened, &[]),
    (CallBoundary::StreamChunk, &[]),
];

impl ResponseHooks<PythonSurface> for LegacyHooks {
    /// The deployment success hook, whose returned response is what the caller receives.
    async fn transform<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<Py<PyAny>> {
        let hook = self.with(|logging, py| logging.respond(py, response, timing))?;
        if let Some(hook) = hook {
            let hooked = task.awaited(hook).await;
            self.with(|logging, _| logging.adopt_hooked_response(hooked))?;
        }
        self.with(|logging, py| logging.finalize(py))
    }

    fn stream_opened(&self, head: &Py<PyAny>) -> PyResult<()> {
        self.with(|logging, py| logging.stream_opened(py, head))
    }

    fn stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()> {
        self.with(|logging, py| logging.stream_chunk(py, chunk))
    }
}
