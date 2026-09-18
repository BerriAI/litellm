//! The proxy's deferred success release: the async success handler is queued only once
//! the proxy accepts the response, and at most once.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::{LegacyCallbacks, PythonLogger};

pub(crate) struct PendingSuccess {
    pub(crate) logger: PythonLogger,
    pub(crate) response: Option<Py<PyAny>>,
    pub(crate) start: Py<PyAny>,
    pub(crate) end: Option<Py<PyAny>>,
}

impl PendingSuccess {
    pub(crate) fn sync(&self, py: Python<'_>) -> PyResult<()> {
        self.logger
            .submit_success(py, &self.response, &self.start, &self.end)
    }

    pub(crate) fn asynchronous(&self, py: Python<'_>) -> PyResult<()> {
        self.logger
            .enqueue_success(py, &self.response, &self.start, &self.end)
    }
}

#[pyclass]
pub(crate) struct PendingLogging {
    pub(crate) pending: Option<PendingSuccess>,
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
    use pyo3::types::PyDict;

    use super::*;

    fn install_logging_worker(py: Python<'_>, worker: &Bound<'_, PyAny>) -> PyResult<()> {
        py.import("litellm.litellm_core_utils.logging_worker")?
            .setattr("GLOBAL_LOGGING_WORKER", worker)
    }

    #[test]
    fn deferred_release_uses_release_context_and_allows_reentry_once() {
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
