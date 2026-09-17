//! Per-call state the legacy callback contract keeps between host phases, and the
//! phase dispatch that fans out to litellm's `Logging` object. Expires with the contract.

use pyo3::exceptions::{PyBaseException, PyException};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_callbacks::protocol::{HostPhase, HostStep};

use crate::legacy::logger::{LegacyCallbacks, is_internal_call};
use crate::{DeploymentHooks, PythonLogger, finalize, prepare, setup};

pub fn missing_state() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("missing native call state")
}

pub struct PythonCallState {
    pub args: Py<PyTuple>,
    pub kwargs: Py<PyDict>,
    pub logger: Option<PythonLogger>,
    pub start: Py<PyAny>,
    pub end: Option<Py<PyAny>>,
    pub response: Option<Py<PyAny>>,
    pub error: Option<Py<PyBaseException>>,
    pub asynchronous: bool,
    pub internal: bool,
    pub call_type: &'static str,
}

pub fn now(py: Python<'_>) -> PyResult<Py<PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call_method0("now")
        .map(Bound::unbind)
}

impl PythonCallState {
    pub fn invoke(
        &mut self,
        py: Python<'_>,
        phase: HostPhase,
    ) -> PyResult<HostStep<Py<PyAny>, Py<PyAny>>> {
        match phase {
            HostPhase::Setup => self.setup(py)?,
            HostPhase::DeploymentPreCall => {
                if !DeploymentHooks::needed(py)? {
                    return Ok(HostStep::Ready(self.kwargs.clone_ref(py).into_any()));
                }
                return Ok(HostStep::Suspend(DeploymentHooks::before_call(
                    py,
                    &self.kwargs,
                    self.call_type,
                )?));
            }
            HostPhase::Prepare => self.prepare(py)?,
            HostPhase::DeploymentPostCall => {
                if !DeploymentHooks::needed(py)? {
                    return self
                        .response
                        .as_ref()
                        .map(|value| HostStep::Ready(value.clone_ref(py)))
                        .ok_or_else(missing_state);
                }
                return Ok(HostStep::Suspend(DeploymentHooks::after_success(
                    py,
                    &self.kwargs,
                    &self.response,
                    self.call_type,
                )?));
            }
            HostPhase::Finalize => self.finalize(py)?,
            HostPhase::Success => self.dispatch_success(py)?,
            HostPhase::DeploymentFailure => {
                if let Some(error) = &self.error
                    && DeploymentHooks::needed(py)?
                {
                    return Ok(HostStep::Suspend(DeploymentHooks::after_failure(
                        py,
                        &self.kwargs,
                        error,
                        self.call_type,
                    )?));
                }
            }
            HostPhase::Failure | HostPhase::AsyncFailure => {
                if let Some(awaitable) =
                    self.dispatch_failure(py, phase == HostPhase::AsyncFailure)?
                {
                    return Ok(HostStep::Suspend(awaitable));
                }
            }
            HostPhase::Execute
            | HostPhase::ConstructResponse
            | HostPhase::MapFailure
            | HostPhase::Complete => return Err(missing_state()),
        }
        Ok(HostStep::Ready(py.None()))
    }

    pub fn accept(&mut self, py: Python<'_>, phase: HostPhase, value: Py<PyAny>) -> PyResult<()> {
        match phase {
            HostPhase::DeploymentPreCall => {
                self.kwargs = value.into_bound(py).cast_into::<PyDict>()?.unbind()
            }
            HostPhase::DeploymentPostCall => self.response = Some(value),
            _ => {}
        }
        Ok(())
    }

    pub fn new(
        py: Python<'_>,
        args: Py<PyTuple>,
        kwargs: Py<PyDict>,
        asynchronous: bool,
        call_type: &'static str,
    ) -> PyResult<Self> {
        Ok(Self {
            args,
            kwargs,
            logger: None,
            start: py.None(),
            end: None,
            response: None,
            error: None,
            asynchronous,
            internal: false,
            call_type,
        })
    }

    pub fn logger(&self) -> PyResult<&PythonLogger> {
        self.logger.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("call logging is not initialized")
        })
    }

    pub fn setup(&mut self, py: Python<'_>) -> PyResult<()> {
        self.start = now(py)?;
        self.internal = is_internal_call(py)?;
        let result = setup(
            py,
            self.call_type,
            &self.args,
            &self.kwargs,
            &self.start,
            self.asynchronous,
        )?;
        self.logger = Some(result.logger()?);
        self.kwargs = result.kwargs()?;
        Ok(())
    }

    pub fn prepare(&mut self, py: Python<'_>) -> PyResult<()> {
        self.kwargs = prepare(py, self.kwargs.bind(py), self.logger()?)?.unbind();
        Ok(())
    }

    pub fn finalize(&self, py: Python<'_>) -> PyResult<()> {
        finalize(
            py,
            &self.response,
            self.logger()?,
            &self.kwargs,
            &self.start,
            &self.end,
        )
    }

    pub fn dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        match self.try_dispatch_success(py) {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, self.logger.as_ref().map(|logger| logger.object(py)));
                Ok(())
            }
            result => result,
        }
    }

    fn try_dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        let logger = self.logger()?;
        let pending = || PendingSuccess {
            logger: logger.clone_ref(py),
            response: self.response.as_ref().map(|value| value.clone_ref(py)),
            start: self.start.clone_ref(py),
            end: self.end.as_ref().map(|value| value.clone_ref(py)),
        };
        if !self.asynchronous {
            if !logger.callbacks_needed(py, "sync_success")? {
                return logger.success_bookkeeping(
                    py,
                    &self.response,
                    &self.start,
                    &self.end,
                    false,
                );
            }
            pending().sync(py)
        } else {
            if !self.internal
                && self
                    .kwargs
                    .bind(py)
                    .get_item("fallbacks")?
                    .is_none_or(|value| value.is_none())
            {
                if !logger.callbacks_needed(py, "async_success")? {
                    logger.success_bookkeeping(py, &self.response, &self.start, &self.end, true)?;
                } else if logger.defers_async_logging(py) {
                    let pending = Py::new(
                        py,
                        PendingLogging {
                            pending: Some(pending()),
                        },
                    )?;
                    logger.defer_success(py, pending.bind(py).as_any())?;
                } else {
                    pending().asynchronous(py)?;
                }
            }
            logger.sync_success_for_async_call(py, &self.response, &self.start, &self.end)
        }
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
        self.logger()?
            .failure(py, error, &self.start, &self.end, asynchronous)
    }

    pub fn cleanup(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = logger.restore_context(py)
        {
            error.write_unraisable(py, None);
        }
    }

    pub fn retain_error(&mut self, py: Python<'_>, error: PyErr) {
        self.error = Some(error.into_value(py));
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.args)?;
        visit.call(&self.kwargs)?;
        if let Some(logger) = &self.logger {
            logger.traverse(visit)?;
        }
        visit.call(&self.start)?;
        visit.call(&self.end)?;
        visit.call(&self.response)?;
        visit.call(&self.error)
    }
}

struct PendingSuccess {
    logger: PythonLogger,
    response: Option<Py<PyAny>>,
    start: Py<PyAny>,
    end: Option<Py<PyAny>>,
}

impl PendingSuccess {
    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        self.logger
            .submit_success(py, &self.response, &self.start, &self.end)
    }

    fn asynchronous(&self, py: Python<'_>) -> PyResult<()> {
        self.logger
            .enqueue_success(py, &self.response, &self.start, &self.end)
    }
}

#[pyclass]
struct PendingLogging {
    pending: Option<PendingSuccess>,
}

#[pymethods]
impl PendingLogging {
    fn release(slf: &Bound<'_, Self>, py: Python<'_>, success: bool) -> PyResult<()> {
        let pending = slf.borrow_mut().pending.take();
        if let Some(pending) = pending
            && success
        {
            match pending.asynchronous(py) {
                Err(error) if error.is_instance_of::<PyException>(py) => {
                    error.write_unraisable(py, Some(pending.logger.object(py)));
                }
                result => return result,
            }
        }
        Ok(())
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        if let Some(pending) = &self.pending {
            pending.logger.traverse(&visit)?;
            visit.call(&pending.response)?;
            visit.call(&pending.start)?;
            visit.call(&pending.end)?;
        }
        Ok(())
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let pending = slf.borrow_mut().pending.take();
        drop(pending);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static PYTHON_GLOBALS: Mutex<()> = Mutex::new(());

    fn install_logging_worker(py: Python<'_>, worker: &Bound<'_, PyAny>) -> PyResult<()> {
        py.import("litellm.litellm_core_utils.logging_worker")?
            .setattr("GLOBAL_LOGGING_WORKER", worker)
    }

    fn state(
        py: Python<'_>,
        logger: Py<PyAny>,
        response: Py<PyAny>,
        asynchronous: bool,
    ) -> PythonCallState {
        PythonCallState {
            args: PyTuple::empty(py).unbind(),
            kwargs: PyDict::new(py).unbind(),
            logger: Some(logger.extract(py).unwrap()),
            start: py.None(),
            end: Some(py.None()),
            response: Some(response),
            error: None,
            asynchronous,
            internal: false,
            call_type: "test",
        }
    }

    #[test]
    fn success_dispatch_reports_ordinary_failures_without_replacing_response() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import sys

response = object()
failure = ValueError('terminal diagnostic')
diagnostics = []
old_hook = sys.unraisablehook
sys.unraisablehook = lambda event: diagnostics.append(event.exc_value)

class Logger:
    def handle_sync_success_callbacks_for_async_calls(self, *args):
        raise failure

logger = Logger()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let response = locals.get_item("response").unwrap().unwrap().unbind();
            let mut lifecycle_state = state(
                py,
                locals.get_item("logger").unwrap().unwrap().unbind(),
                response.clone_ref(py),
                true,
            );
            lifecycle_state.internal = true;
            lifecycle_state.dispatch_success(py).unwrap();
            assert!(lifecycle_state.response.as_ref().unwrap().is(&response));
            py.run(
                pyo3::ffi::c_str!(
                    r#"
assert diagnostics == [failure]
sys.unraisablehook = old_hook
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }

    #[test]
    fn retained_failure_preserves_exception_identity() {
        Python::initialize();
        Python::attach(|py| {
            let logger = PyDict::new(py).into_any().unbind();
            let response = py.None();
            let failure = pyo3::exceptions::PyValueError::new_err("identity");
            let failure_value = failure.value(py).clone().unbind();
            let mut lifecycle_state = state(py, logger, response, false);
            lifecycle_state.retain_error(py, failure);
            let retained = lifecycle_state.error.take().unwrap();
            assert!(retained.is(&failure_value));
        });
    }

    #[test]
    fn deferred_release_uses_release_context_and_allows_reentry_once() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import sys
import types
from contextvars import ContextVar

litellm = types.ModuleType('litellm')
core_utils = types.ModuleType('litellm.litellm_core_utils')
logging_worker = types.ModuleType('litellm.litellm_core_utils.logging_worker')
litellm.litellm_core_utils = core_utils
core_utils.logging_worker = logging_worker
sys.modules['litellm'] = litellm
sys.modules['litellm.litellm_core_utils'] = core_utils
sys.modules['litellm.litellm_core_utils.logging_worker'] = logging_worker

marker = ContextVar('marker', default='unset')
observed = []

class Coroutine:
    def close(self):
        observed.append('closed')

class Worker:
    def ensure_initialized_and_enqueue(self, coroutine):
        observed.append(marker.get())
        pending.release(True)
        coroutine.close()

class Logger:
    def async_success_handler(self, *args):
        observed.append('created')
        return Coroutine()

worker = Worker()
logger = Logger()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            install_logging_worker(py, &locals.get_item("worker").unwrap().unwrap()).unwrap();
            let pending = Py::new(
                py,
                PendingLogging {
                    pending: Some(PendingSuccess {
                        logger: locals
                            .get_item("logger")
                            .unwrap()
                            .unwrap()
                            .extract()
                            .unwrap(),
                        response: Some(py.None()),
                        start: py.None(),
                        end: Some(py.None()),
                    }),
                },
            )
            .unwrap();
            locals.set_item("pending", &pending).unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
marker.set('release')
pending.release(True)
pending.release(True)
assert observed == ['created', 'release', 'closed']
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }

    #[test]
    fn deferred_logging_collects_cycles_through_typed_logger() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!("class Logger: pass\nlogger = Logger()"),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let pending = Py::new(
                py,
                PendingLogging {
                    pending: Some(PendingSuccess {
                        logger: locals
                            .get_item("logger")
                            .unwrap()
                            .unwrap()
                            .extract()
                            .unwrap(),
                        response: None,
                        start: py.None(),
                        end: None,
                    }),
                },
            )
            .unwrap();
            locals.set_item("pending", pending).unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import gc
import weakref
logger.pending = pending
reference = weakref.ref(logger)
del logger, pending
gc.collect()
assert reference() is None
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }
}
