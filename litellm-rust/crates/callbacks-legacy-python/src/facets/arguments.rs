use litellm_host::hooks::ArgumentHooks;
use litellm_host_python::{Awaiter, PythonSurface};
use pyo3::{prelude::*, types::PyDict};

use crate::{CallBoundary, LegacyHooks};

pub(crate) const CALLBACKS: &[(CallBoundary, &[&str])] = &[(
    CallBoundary::PrepareArguments,
    &["async_pre_call_deployment_hook"],
)];

impl ArgumentHooks<PythonSurface> for LegacyHooks {
    /// `function_setup`, then the deployment pre-call hook, whose returned kwargs become
    /// the keyword view the rest of the call reads.
    async fn prepare<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<Py<PyDict>> {
        let hook = self.with(|logging, py| logging.begin_call(py, arguments, started_at))?;
        if let Some(hook) = hook {
            let hooked = task.awaited(hook).await;
            self.with(|logging, py| logging.adopt_hooked_arguments(py, hooked))?;
        }
        self.with(|logging, py| logging.prepare(py))
    }

    fn prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        self.with(|logging, py| logging.adopt_arguments(py, arguments));
        Ok(())
    }
}
