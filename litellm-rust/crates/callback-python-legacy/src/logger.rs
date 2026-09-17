use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[derive(FromPyObject)]
pub struct LegacyPythonLogger(Py<PyAny>);

impl LegacyPythonLogger {
    pub fn object<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.0.bind(py)
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self(self.0.clone_ref(py))
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }

    pub fn callbacks_needed(&self, py: Python<'_>, phase: &str) -> PyResult<bool> {
        if !self
            .object(py)
            .getattr("_native_callback_fast_path")
            .is_ok_and(|value| value.is_truthy().unwrap_or(false))
        {
            return Ok(true);
        }
        py.import("litellm.rust_bridge.lifecycle")?
            .getattr("callbacks_needed")?
            .call1((self.object(py), phase))?
            .extract()
    }

    pub fn success_bookkeeping(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> PyResult<()> {
        py.import("litellm.rust_bridge.lifecycle")?
            .getattr("success_bookkeeping")?
            .call1((self.object(py), response, start, end, asynchronous))?;
        Ok(())
    }

    pub fn defers_async_logging(&self, py: Python<'_>) -> bool {
        self.object(py)
            .getattr("_defer_async_logging")
            .is_ok_and(|value| value.is_truthy().unwrap_or(false))
    }

    pub fn defer_success(&self, py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<()> {
        self.object(py).setattr("_native_pending_logging", pending)
    }

    pub fn sync_success_for_async_call(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if !self.callbacks_needed(py, "sync_success_async")? {
            return Ok(());
        }
        self.object(py).call_method1(
            "handle_sync_success_callbacks_for_async_calls",
            (response, start, end),
        )?;
        Ok(())
    }

    pub fn failure(
        &self,
        py: Python<'_>,
        error: &Py<PyBaseException>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        if !self.callbacks_needed(
            py,
            if asynchronous {
                "async_failure"
            } else {
                "sync_failure"
            },
        )? {
            py.import("litellm.rust_bridge.lifecycle")?
                .getattr("failure_bookkeeping")?
                .call1((self.object(py), error, start, end, asynchronous))?;
            return Ok(None);
        }
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

    pub fn restore_context(&self, py: Python<'_>) -> PyResult<()> {
        py.import("litellm.utils")?
            .getattr("_restore_correlation_context_if_supported")?
            .call1((self.object(py),))?;
        Ok(())
    }

    pub fn submit_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if !self.callbacks_needed(py, "sync_success")? {
            return self.success_bookkeeping(py, response, start, end, false);
        }
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

    pub fn enqueue_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if !self.callbacks_needed(py, "async_success")? {
            return self.success_bookkeeping(py, response, start, end, true);
        }
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

pub struct SetupResult<'py>(Bound<'py, PyAny>);

impl SetupResult<'_> {
    pub fn logger(&self) -> PyResult<LegacyPythonLogger> {
        self.0.getattr("logger")?.extract()
    }

    pub fn kwargs(&self) -> PyResult<Py<PyDict>> {
        Ok(self.0.getattr("kwargs")?.extract()?)
    }
}

pub fn setup<'py>(
    py: Python<'py>,
    call_type: &str,
    args: &Py<PyTuple>,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    asynchronous: bool,
) -> PyResult<SetupResult<'py>> {
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("setup")?
        .call1((call_type, args, kwargs, start, asynchronous))
        .map(SetupResult)
}

pub fn finalize(
    py: Python<'_>,
    response: &Option<Py<PyAny>>,
    logger: &LegacyPythonLogger,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    end: &Option<Py<PyAny>>,
) -> PyResult<()> {
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("finalize")?
        .call1((response, logger.object(py), kwargs, start, end))?;
    Ok(())
}

pub fn is_internal_call(py: Python<'_>) -> PyResult<bool> {
    py.import("litellm._internal_context")?
        .getattr("is_internal_call")?
        .call_method0("get")?
        .extract()
}

pub struct DeploymentHooks;

impl DeploymentHooks {
    pub fn needed(py: Python<'_>) -> PyResult<bool> {
        py.import("litellm.rust_bridge.lifecycle")?
            .getattr("deployment_callbacks_needed")?
            .call0()?
            .extract()
    }

    pub fn before_call(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_pre_call_deployment_hook")?
            .call1((kwargs, call_type))
            .map(Bound::unbind)
    }

    pub fn after_success(
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

    pub fn after_failure(
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
    use pyo3::exceptions::PyTypeError;

    #[test]
    fn setup_fields_are_checked_lazily() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
reads = []
class Logger:
    def __getattribute__(self, name):
        reads.append(name)
        raise AssertionError('logger methods must remain lazy')
logger = Logger()
class Setup:
    @property
    def logger(self):
        reads.append('logger')
        return logger
    @property
    def kwargs(self):
        reads.append('kwargs')
        return []
result = Setup()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let result = SetupResult(locals.get_item("result").unwrap().unwrap());
            let logger = result.logger().unwrap();
            assert!(
                logger
                    .object(py)
                    .is(locals.get_item("logger").unwrap().unwrap())
            );
            assert!(
                result
                    .kwargs()
                    .unwrap_err()
                    .is_instance_of::<PyTypeError>(py)
            );
            assert_eq!(
                locals
                    .get_item("reads")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["logger", "kwargs"]
            );
        });
    }
}
