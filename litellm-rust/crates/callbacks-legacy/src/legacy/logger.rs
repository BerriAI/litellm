//! Callback fan-out over litellm's `Logging` object: which callbacks are registered,
//! the deferred and worker-submitted success paths, and the sync-callbacks-for-async-calls
//! duplication. All of it expires with the legacy callback contract.

use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;

use crate::logger::PythonLogger;

pub trait LegacyCallbacks {
    fn callbacks_needed(&self, py: Python<'_>, phase: &str) -> PyResult<bool>;

    fn defers_async_logging(&self, py: Python<'_>) -> bool;

    fn defer_success(&self, py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<()>;

    fn sync_success_for_async_call(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()>;

    fn failure(
        &self,
        py: Python<'_>,
        error: &Py<PyBaseException>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>>;

    fn submit_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()>;

    fn enqueue_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()>;
}

impl LegacyCallbacks for PythonLogger {
    fn callbacks_needed(&self, py: Python<'_>, phase: &str) -> PyResult<bool> {
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
    fn defers_async_logging(&self, py: Python<'_>) -> bool {
        self.object(py)
            .getattr("_defer_async_logging")
            .is_ok_and(|value| value.is_truthy().unwrap_or(false))
    }

    fn defer_success(&self, py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<()> {
        self.object(py).setattr("_native_pending_logging", pending)
    }

    fn sync_success_for_async_call(
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

    fn failure(
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
    fn submit_success(
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

    fn enqueue_success(
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

/// Proxy-internal calls skip the legacy success fan-out.
pub fn is_internal_call(py: Python<'_>) -> PyResult<bool> {
    py.import("litellm._internal_context")?
        .getattr("is_internal_call")?
        .call_method0("get")?
        .extract()
}
