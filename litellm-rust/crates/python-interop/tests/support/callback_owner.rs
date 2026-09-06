use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyclass]
#[derive(Default)]
pub struct OwnerFactory {
    live: Arc<AtomicUsize>,
}

#[pymethods]
impl OwnerFactory {
    #[getter]
    fn live(&self) -> usize {
        self.live.load(Ordering::SeqCst)
    }

    #[pyo3(signature = (callable, positional, keywords=None, awaited=false))]
    fn prepare(
        &self,
        callable: Py<PyAny>,
        positional: Py<PyTuple>,
        keywords: Option<Py<PyDict>>,
        awaited: bool,
    ) -> Owner {
        self.live.fetch_add(1, Ordering::SeqCst);
        Owner {
            call: Some(PreparedCall::new(
                if awaited {
                    InvocationMode::Await
                } else {
                    InvocationMode::Direct
                },
                callable,
                positional,
                keywords,
            )),
            live: self.live.clone(),
        }
    }
}

#[pyclass(weakref)]
struct Owner {
    call: Option<PreparedCall>,
    live: Arc<AtomicUsize>,
}

impl Owner {
    fn release(&mut self) -> Option<PreparedCall> {
        let call = self.call.take();
        if call.is_some() {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
        call
    }
}

#[pymethods]
impl Owner {
    fn invoke(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let call = slf
            .borrow()
            .call
            .as_ref()
            .map(|call| call.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("invocation owner released"))?;
        match call.invoke(py)? {
            InvocationOutcome::Returned(value) | InvocationOutcome::Awaitable(value) => Ok(value),
        }
    }

    fn clone_owner(&self, py: Python<'_>) -> PyResult<Self> {
        let call = self
            .call
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("invocation owner released"))?;
        self.live.fetch_add(1, Ordering::SeqCst);
        Ok(Self {
            call: Some(call.clone_ref(py)),
            live: self.live.clone(),
        })
    }

    fn close(slf: &Bound<'_, Self>) {
        let released = slf.borrow_mut().release();
        drop(released);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(call) = &self.call {
            call.traverse(visit)?;
        }
        Ok(())
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        Self::close(slf);
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        drop(self.release());
    }
}
