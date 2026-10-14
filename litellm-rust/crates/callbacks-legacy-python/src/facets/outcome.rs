use litellm_host::{
    hooks::{Cancelled, OutcomeHooks},
    lifecycle::{FailureOrigin, Timing},
};
use litellm_host_python::{Awaiter, PythonSurface, report_unless_cancelled};
use pyo3::{exceptions::PyBaseException, prelude::*};

use crate::{CallBoundary, LegacyHooks, adapter::FailureStage, is_cancellation};

pub(crate) const CALLBACKS: &[(CallBoundary, &[&str])] = &[
    (
        CallBoundary::Succeeded,
        &[
            "log_success_event",
            "async_log_success_event",
            "logging_hook",
            "async_logging_hook",
            "redact_standard_logging_payload_from_model_call_details",
            "log_event",
            "async_log_event",
        ],
    ),
    (
        CallBoundary::Failed,
        &[
            "async_post_call_failure_deployment_hook",
            "log_failure_event",
            "async_log_failure_event",
            "log_model_group_rate_limit_error",
            "log_event",
            "async_log_event",
        ],
    ),
];

/// An awaited failure handler ends the call however it ends, unless it was cancelled.
fn async_failure_result(result: PyResult<Py<PyAny>>) -> PyResult<()> {
    match result {
        Err(error) if Python::attach(|py| is_cancellation(py, &error)) => Err(error),
        _ => Ok(()),
    }
}

impl LegacyHooks {
    async fn fail(
        &self,
        task: &dyn Awaiter,
        timing: Timing,
        origin: FailureOrigin,
        error: &Py<PyBaseException>,
    ) -> PyResult<()> {
        let stage = self.with(|logging, py| logging.failed(py, timing, origin, error))?;
        let handler = match stage {
            FailureStage::DeploymentHook(hook) => {
                drop(task.awaited(hook).await);
                self.with(|logging, py| logging.dispatch_failure(py))?
            }
            FailureStage::Handlers(handler) => handler,
        };
        match handler {
            Some(handler) => async_failure_result(task.awaited(handler).await),
            None => Ok(()),
        }
    }
}

impl OutcomeHooks<PythonSurface> for LegacyHooks {
    /// The sync and async success fan-out with the selected response.
    async fn succeeded<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        timing: Timing,
        response: &Py<PyAny>,
    ) -> Result<(), Cancelled<PyErr>> {
        report_unless_cancelled(self.with(|logging, py| logging.succeeded(py, timing, response)))
    }

    /// The deployment failure hook, then the sync and async failure fan-out with the
    /// selected exception.
    async fn failed<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        timing: Timing,
        origin: FailureOrigin,
        error: &Py<PyBaseException>,
    ) -> Result<(), Cancelled<PyErr>> {
        report_unless_cancelled(self.fail(task, timing, origin, error).await)
    }
}
