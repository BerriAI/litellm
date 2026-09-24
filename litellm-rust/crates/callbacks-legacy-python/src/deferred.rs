//! The proxy's deferred success release: the async success handler is queued only once
//! the proxy accepts the response, and at most once.

use pyo3::{exceptions::PyException, prelude::*};

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
#[path = "../tests/legacy/deferred.rs"]
mod tests;
