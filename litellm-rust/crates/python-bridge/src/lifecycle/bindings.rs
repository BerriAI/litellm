use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(FromPyObject)]
pub(crate) struct PythonLogger(Py<PyAny>);

impl PythonLogger {
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.0.bind(py)
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self(self.0.clone_ref(py))
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }

    pub(super) fn defers_async_logging(&self, py: Python<'_>) -> bool {
        self.object(py)
            .getattr("_defer_async_logging")
            .is_ok_and(|value| value.is_truthy().unwrap_or(false))
    }

    pub(super) fn defer_success(
        &self,
        py: Python<'_>,
        pending: Py<super::PendingLogging>,
    ) -> PyResult<()> {
        self.object(py).setattr("_native_pending_logging", pending)
    }

    pub(super) fn sync_success_for_async_call(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.object(py).call_method1(
            "handle_sync_success_callbacks_for_async_calls",
            (response, start, end),
        )?;
        Ok(())
    }

    pub(super) fn failure(
        &self,
        py: Python<'_>,
        error: &Py<PyBaseException>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        let trace = py
            .import("traceback")?
            .getattr("format_exception")?
            .call1((error,))?;
        let trace = pyo3::types::PyString::new(py, "").call_method1("join", (trace,))?;
        let value = self.object(py).call_method1(
            if asynchronous {
                "async_failure_handler"
            } else {
                "failure_handler"
            },
            (error, trace, start, end),
        )?;
        Ok(asynchronous.then(|| value.unbind()))
    }

    pub(super) fn restore_context(&self, py: Python<'_>) -> PyResult<()> {
        py.import("litellm.utils")?
            .getattr("_restore_correlation_context_if_supported")?
            .call1((self.object(py),))?;
        Ok(())
    }

    pub(super) fn submit_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let context = py.import("contextvars")?.call_method0("copy_context")?;
        py.import("litellm.litellm_core_utils.litellm_logging")?
            .getattr("executor")?
            .call_method1(
                "submit",
                (
                    context.getattr("run")?,
                    self.object(py).getattr("success_handler")?,
                    response,
                    start,
                    end,
                ),
            )?;
        Ok(())
    }

    pub(super) fn enqueue_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let context = py.import("contextvars")?.call_method0("copy_context")?;
        let worker = py
            .import("litellm.litellm_core_utils.logging_worker")?
            .getattr("GLOBAL_LOGGING_WORKER")?
            .getattr("ensure_initialized_and_enqueue")?;
        let coroutine = self
            .object(py)
            .call_method1("async_success_handler", (response, start, end))?;
        let enqueue = context.call_method1("run", (worker, &coroutine));
        if enqueue.is_err()
            && let Err(error) = coroutine.call_method0("close")
        {
            error.write_unraisable(py, Some(&coroutine));
        }
        enqueue.map(|_| ())
    }
}

pub(super) fn finalize(
    py: Python<'_>,
    response: &Option<Py<PyAny>>,
    logger: &PythonLogger,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    end: &Option<Py<PyAny>>,
) -> PyResult<()> {
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("finalize")?
        .call1((response, logger.object(py), kwargs, start, end))?;
    Ok(())
}

pub(super) fn is_internal_call(py: Python<'_>) -> PyResult<bool> {
    py.import("litellm._internal_context")?
        .getattr("is_internal_call")?
        .call_method0("get")?
        .extract()
}

pub(super) struct DeploymentHooks;

impl DeploymentHooks {
    pub(super) fn before_call(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_pre_call_deployment_hook")?
            .call1((kwargs, call_type))
            .map(Bound::unbind)
    }

    pub(super) fn after_success(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        response: &Option<Py<PyAny>>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_post_call_success_deployment_hook")?
            .call1((kwargs, response, call_type))
            .map(Bound::unbind)
    }

    pub(super) fn after_failure(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        error: &Py<PyBaseException>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_post_call_failure_deployment_hook")?
            .call1((kwargs, error, call_type))
            .map(Bound::unbind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logger_resolves_each_callback_at_invocation_and_preserves_arguments() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
calls = []
response, start, end = object(), object(), object()
class Logger:
    @property
    def handle_sync_success_callbacks_for_async_calls(self):
        generation = len(calls)
        def callback(*args):
            assert args == (response, start, end)
            calls.append(generation)
        return callback
logger = Logger()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let logger: PythonLogger = locals
                .get_item("logger")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let response = Some(locals.get_item("response").unwrap().unwrap().unbind());
            let start = locals.get_item("start").unwrap().unwrap().unbind();
            let end = Some(locals.get_item("end").unwrap().unwrap().unbind());
            for _ in 0..2 {
                logger
                    .sync_success_for_async_call(py, &response, &start, &end)
                    .unwrap();
            }
            assert_eq!(
                locals
                    .get_item("calls")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<usize>>()
                    .unwrap(),
                [0, 1]
            );
        });
    }
}
