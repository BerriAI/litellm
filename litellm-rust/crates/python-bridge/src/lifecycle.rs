use litellm_core::call_lifecycle::host::{CallHost, HostLifecycle, HostStep};
use litellm_python_interop::{CoroutineBody, CoroutineStep};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) struct PythonLifecycle<
    H: CallHost<Value = Py<PyAny>, Error = PyErr, Suspension = Py<PyAny>>,
>(pub HostLifecycle<H>);

impl<H> CoroutineBody for PythonLifecycle<H>
where
    H: CallHost<Value = Py<PyAny>, Error = PyErr, Suspension = Py<PyAny>> + Send,
{
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<CoroutineStep> {
        match self.0.resume(result)? {
            HostStep::Ready(value) => Ok(CoroutineStep::Return(value)),
            HostStep::Suspend(value) => Ok(CoroutineStep::Await(value)),
        }
    }
}

pub(crate) struct PythonCallState {
    pub kwargs: Py<PyDict>,
    pub logger: Option<Py<PyAny>>,
    pub start: Py<PyAny>,
    pub end: Option<Py<PyAny>>,
    pub response: Option<Py<PyAny>>,
    pub error: Option<PyErr>,
    pub asynchronous: bool,
    pub internal: bool,
    pub call_type: &'static str,
}

pub(crate) fn now(py: Python<'_>) -> PyResult<Py<PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call_method0("now")
        .map(Bound::unbind)
}

impl PythonCallState {
    pub fn new(
        py: Python<'_>,
        kwargs: Py<PyDict>,
        asynchronous: bool,
        call_type: &'static str,
    ) -> PyResult<Self> {
        let internal = py
            .import("litellm._internal_context")?
            .getattr("is_internal_call")?
            .call_method0("get")?
            .extract()?;
        Ok(Self {
            kwargs,
            logger: None,
            start: now(py)?,
            end: None,
            response: None,
            error: None,
            asynchronous,
            internal,
            call_type,
        })
    }

    pub fn logger<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.logger
            .as_ref()
            .map(|value| value.bind(py).clone())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("call logging is not initialized")
            })
    }

    pub fn setup(&mut self, py: Python<'_>) -> PyResult<()> {
        let result = py
            .import("litellm.rust_bridge.lifecycle")?
            .getattr("setup")?
            .call1((self.call_type, &self.kwargs, &self.start, self.asynchronous))?;
        self.logger = Some(result.getattr("logger")?.unbind());
        self.kwargs = result.getattr("kwargs")?.cast_into::<PyDict>()?.unbind();
        Ok(())
    }

    pub fn prepare(&mut self, py: Python<'_>) -> PyResult<()> {
        self.kwargs = py
            .import("litellm.rust_bridge.lifecycle")?
            .getattr("prepare")?
            .call1((&self.kwargs, self.logger(py)?))?
            .cast_into::<PyDict>()?
            .unbind();
        Ok(())
    }

    pub fn finalize(&self, py: Python<'_>) -> PyResult<()> {
        py.import("litellm.rust_bridge.lifecycle")?
            .getattr("finalize")?
            .call1((
                &self.response,
                self.logger(py)?,
                &self.kwargs,
                &self.start,
                &self.end,
            ))?;
        Ok(())
    }

    pub fn dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        let logger = self.logger(py)?;
        let pending = PendingSuccess {
            logger: logger.clone().unbind(),
            response: self.response.as_ref().map(|value| value.clone_ref(py)),
            start: self.start.clone_ref(py),
            end: self.end.as_ref().map(|value| value.clone_ref(py)),
            context: py
                .import("contextvars")?
                .call_method0("copy_context")?
                .unbind(),
        };
        if !self.asynchronous {
            return pending.sync(py);
        }
        if !self.internal
            && self
                .kwargs
                .bind(py)
                .get_item("fallbacks")?
                .is_none_or(|value| value.is_none())
        {
            if logger
                .getattr("_defer_async_logging")
                .is_ok_and(|value| value.is_truthy().unwrap_or(false))
            {
                logger.setattr(
                    "_native_pending_logging",
                    Py::new(
                        py,
                        PendingLogging {
                            pending: Some(pending),
                        },
                    )?,
                )?;
            } else {
                pending.asynchronous(py)?;
            }
        }
        logger.call_method1(
            "handle_sync_success_callbacks_for_async_calls",
            (&self.response, &self.start, &self.end),
        )?;
        Ok(())
    }

    pub fn dispatch_failure(
        &self,
        py: Python<'_>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        if self.logger.is_none() || (self.asynchronous && self.internal) {
            return Ok(None);
        }
        let Some(error) = &self.error else {
            return Ok(None);
        };
        let trace = py
            .import("traceback")?
            .getattr("format_exception")?
            .call1((error.value(py),))?;
        let trace = pyo3::types::PyString::new(py, "").call_method1("join", (trace,))?;
        let value = self.logger(py)?.call_method1(
            if asynchronous {
                "async_failure_handler"
            } else {
                "failure_handler"
            },
            (error.value(py), trace, &self.start, &self.end),
        )?;
        Ok(asynchronous.then(|| value.unbind()))
    }

    pub fn cleanup(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = py.import("litellm.utils").and_then(|utils| {
                utils
                    .getattr("_restore_correlation_context_if_supported")?
                    .call1((logger,))
            })
        {
            error.write_unraisable(py, None);
        }
    }
}

struct PendingSuccess {
    logger: Py<PyAny>,
    response: Option<Py<PyAny>>,
    start: Py<PyAny>,
    end: Option<Py<PyAny>>,
    context: Py<PyAny>,
}

impl PendingSuccess {
    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        py.import("litellm.litellm_core_utils.litellm_logging")?
            .getattr("executor")?
            .call_method1(
                "submit",
                (
                    self.context.getattr(py, "run")?,
                    self.logger.getattr(py, "success_handler")?,
                    &self.response,
                    &self.start,
                    &self.end,
                ),
            )?;
        Ok(())
    }

    fn asynchronous(&self, py: Python<'_>) -> PyResult<()> {
        let coroutine = self.logger.call_method1(
            py,
            "async_success_handler",
            (&self.response, &self.start, &self.end),
        )?;
        self.context.call_method1(
            py,
            "run",
            (
                py.import("litellm.litellm_core_utils.logging_worker")?
                    .getattr("GLOBAL_LOGGING_WORKER")?
                    .getattr("ensure_initialized_and_enqueue")?,
                coroutine,
            ),
        )?;
        Ok(())
    }
}

#[pyclass]
struct PendingLogging {
    pending: Option<PendingSuccess>,
}

#[pymethods]
impl PendingLogging {
    fn release(&mut self, py: Python<'_>, success: bool) -> PyResult<()> {
        if let Some(pending) = self.pending.take()
            && success
        {
            pending.asynchronous(py)?;
        }
        Ok(())
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        if let Some(pending) = &self.pending {
            visit.call(&pending.logger)?;
            visit.call(&pending.response)?;
            visit.call(&pending.start)?;
            visit.call(&pending.end)?;
            visit.call(&pending.context)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.pending = None;
    }
}
