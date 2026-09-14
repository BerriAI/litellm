use litellm_core::call_lifecycle::dispatch::{
    AsyncSuccessDelivery, DeferredSuccess, SuccessDispatch, SuccessFacts, failure_dispatch,
    success_dispatch,
};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use super::bindings::PythonLogger;
use super::state::PythonCallState;

impl PythonCallState {
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
        let facts = PythonSuccessFacts {
            py,
            state: self,
            logger,
        };
        match success_dispatch(self.asynchronous, self.internal, &facts)? {
            SuccessDispatch::SyncBookkeeping => {
                logger.success_bookkeeping(py, &self.response, &self.start, &self.end, false)
            }
            SuccessDispatch::SyncWorker => pending().sync(py),
            SuccessDispatch::Async(delivery) => {
                match delivery {
                    AsyncSuccessDelivery::Skip => {}
                    AsyncSuccessDelivery::Bookkeeping => logger.success_bookkeeping(
                        py,
                        &self.response,
                        &self.start,
                        &self.end,
                        true,
                    )?,
                    AsyncSuccessDelivery::Background => pending().asynchronous(py)?,
                    AsyncSuccessDelivery::Deferred => {
                        logger.defer_success(py, Py::new(py, PendingLogging::new(pending()))?)?
                    }
                }
                logger.sync_success_for_async_call(py, &self.response, &self.start, &self.end)
            }
        }
    }

    pub fn dispatch_failure(
        &self,
        py: Python<'_>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        if !failure_dispatch(self.asynchronous, self.internal, self.logger.is_some()) {
            return Ok(None);
        }
        let Some(error) = &self.error else {
            return Ok(None);
        };
        self.logger()?
            .failure(py, error, &self.start, &self.end, asynchronous)
    }
}

struct PythonSuccessFacts<'a, 'py> {
    py: Python<'py>,
    state: &'a PythonCallState,
    logger: &'a PythonLogger,
}

impl SuccessFacts for PythonSuccessFacts<'_, '_> {
    type Error = PyErr;

    fn has_fallbacks(&self) -> PyResult<bool> {
        Ok(self
            .state
            .kwargs
            .bind(self.py)
            .get_item("fallbacks")?
            .is_some_and(|value| !value.is_none()))
    }

    fn callbacks_needed(&self, asynchronous: bool) -> PyResult<bool> {
        self.logger.callbacks_needed(
            self.py,
            if asynchronous {
                "async_success"
            } else {
                "sync_success"
            },
        )
    }

    fn defers_async_logging(&self) -> PyResult<bool> {
        Ok(self.logger.defers_async_logging(self.py))
    }
}

pub(super) struct PendingSuccess {
    pub(super) logger: PythonLogger,
    pub(super) response: Option<Py<PyAny>>,
    pub(super) start: Py<PyAny>,
    pub(super) end: Option<Py<PyAny>>,
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
pub(super) struct PendingLogging {
    pending: Option<PendingSuccess>,
    gate: DeferredSuccess,
}

impl PendingLogging {
    pub(super) fn new(pending: PendingSuccess) -> Self {
        Self {
            pending: Some(pending),
            gate: DeferredSuccess::default(),
        }
    }

    fn take(&mut self, accepted: bool) -> (bool, Option<PendingSuccess>) {
        (self.gate.resolve(accepted), self.pending.take())
    }
}

#[pymethods]
impl PendingLogging {
    fn release(slf: &Bound<'_, Self>, py: Python<'_>, success: bool) -> PyResult<()> {
        let (dispatch, pending) = slf.borrow_mut().take(success);
        if let Some(pending) = pending
            && dispatch
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
        let (_, pending) = slf.borrow_mut().take(false);
        drop(pending);
    }
}
