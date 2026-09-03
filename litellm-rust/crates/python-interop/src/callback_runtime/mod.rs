use std::ffi::CStr;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::thread::{self, ThreadId};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{Pythonized, from_py};

const PYTHON_RUNTIME_SOURCE: &CStr = pyo3::ffi::c_str!(include_str!("invoke.py"));

pub struct Direct;
pub struct Awaitable;

pub trait ReturnMode: Send {
    const AWAITABLE: bool;
}

impl ReturnMode for Direct {
    const AWAITABLE: bool = false;
}

impl ReturnMode for Awaitable {
    const AWAITABLE: bool = true;
}

pub trait CallbackContext<M: ReturnMode>: Send {
    fn invoke(
        &mut self,
        callable: Py<PyAny>,
        payload: Py<PyAny>,
    ) -> impl Future<Output = PyResult<Py<PyAny>>> + Send;
}

pub struct Callback<I, O, M> {
    callable: Py<PyAny>,
    signature: PhantomData<fn(I) -> (O, M)>,
}

impl<I, O, M> Callback<I, O, M>
where
    I: Serialize + Sync,
    O: DeserializeOwned + Send,
    M: ReturnMode,
{
    pub fn new(callable: Bound<'_, PyAny>) -> PyResult<Self> {
        if !callable.is_callable() {
            return Err(PyTypeError::new_err("hook binding must be callable"));
        }
        Ok(Self {
            callable: callable.unbind(),
            signature: PhantomData,
        })
    }

    pub async fn call<C: CallbackContext<M>>(&mut self, context: &mut C, input: &I) -> PyResult<O> {
        let (callable, payload) = Python::attach(|py| {
            Ok::<_, PyErr>((
                self.callable.clone_ref(py),
                Pythonized(input).into_pyobject(py)?.unbind(),
            ))
        })?;
        let result = context.invoke(callable, payload).await?;
        Python::attach(|py| {
            from_py(result.bind(py))
                .map_err(|_| PyTypeError::new_err("hook result does not match its typed contract"))
        })
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let shim = PyModule::from_code(
        module.py(),
        PYTHON_RUNTIME_SOURCE,
        c"litellm_callbacks.py",
        c"_litellm_callbacks",
    )?;
    module.add("__callback_runtime__", shim)
}

struct RuntimeState {
    invocation: Py<PyAny>,
    direct: Py<PyAny>,
    capacity: Arc<Semaphore>,
}

#[derive(Clone)]
pub struct CallbackRuntime(Arc<RuntimeState>);

impl CallbackRuntime {
    pub fn new(module: &Bound<'_, PyModule>, max_in_flight: NonZeroUsize) -> PyResult<Self> {
        if max_in_flight.get() > Semaphore::MAX_PERMITS {
            return Err(PyValueError::new_err(
                "callback capacity exceeds the runtime limit",
            ));
        }
        let shim = module.getattr("__callback_runtime__")?;
        Ok(Self(Arc::new(RuntimeState {
            invocation: shim.getattr("Invocation")?.unbind(),
            direct: shim.getattr("invoke_direct")?.unbind(),
            capacity: Arc::new(Semaphore::new(max_in_flight.get())),
        })))
    }

    pub fn async_context(&self, py: Python<'_>) -> PyResult<AsyncContext> {
        Ok(AsyncContext {
            runtime: self.clone(),
            locals: pyo3_async_runtimes::tokio::get_current_locals(py)?,
            interrupted: false,
        })
    }

    pub fn sync_context(&self, py: Python<'_>) -> PyResult<SyncContext> {
        Ok(SyncContext {
            runtime: self.clone(),
            context: py
                .import("contextvars")?
                .call_method0("copy_context")?
                .unbind(),
            caller: thread::current().id(),
        })
    }

    fn admit(&self) -> PyResult<OwnedSemaphorePermit> {
        Arc::clone(&self.0.capacity)
            .try_acquire_owned()
            .map_err(|_| PyRuntimeError::new_err("callback capacity exhausted"))
    }
}

pub struct SyncContext {
    runtime: CallbackRuntime,
    context: Py<PyAny>,
    caller: ThreadId,
}

impl CallbackContext<Direct> for SyncContext {
    async fn invoke(&mut self, callable: Py<PyAny>, payload: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if thread::current().id() != self.caller {
            return Err(PyRuntimeError::new_err(
                "synchronous callbacks must run on the caller thread",
            ));
        }
        let _permit = self.runtime.admit()?;
        Python::attach(|py| {
            self.context
                .call_method1(py, "run", (&self.runtime.0.direct, callable, payload))
        })
    }
}

pub struct AsyncContext {
    runtime: CallbackRuntime,
    locals: TaskLocals,
    interrupted: bool,
}

#[pyclass(frozen)]
struct Admission {
    _permit: OwnedSemaphorePermit,
}

impl<M: ReturnMode> CallbackContext<M> for AsyncContext {
    async fn invoke(&mut self, callable: Py<PyAny>, payload: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if self.interrupted {
            return Err(PyRuntimeError::new_err("callback session was cancelled"));
        }
        let permit = self.runtime.admit()?;
        let (mut cancellation, future) = Python::attach(|py| {
            let invocation = self.runtime.0.invocation.call1(
                py,
                (
                    callable,
                    payload,
                    M::AWAITABLE,
                    Admission { _permit: permit },
                ),
            )?;
            let cancellation = CancelOnDrop {
                event_loop: self.locals.event_loop(py).unbind(),
                invocation: Some(invocation.clone_ref(py)),
            };
            let coroutine = invocation.call_method0(py, "run")?;
            let future = pyo3_async_runtimes::into_future_with_locals(
                &self.locals,
                coroutine.clone_ref(py).into_bound(py),
            );
            match future {
                Ok(future) => Ok((cancellation, future)),
                Err(error) => {
                    coroutine.call_method0(py, "close")?;
                    Err(error)
                }
            }
        })?;
        self.interrupted = true;
        let result = future.await;
        cancellation.invocation = None;
        self.interrupted = false;
        result
    }
}

struct CancelOnDrop {
    event_loop: Py<PyAny>,
    invocation: Option<Py<PyAny>>,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let Some(invocation) = self.invocation.take() else {
            return;
        };
        Python::attach(|py| {
            let result = invocation.getattr(py, "cancel").and_then(|cancel| {
                self.event_loop
                    .call_method1(py, "call_soon_threadsafe", (cancel,))
            });
            if let Err(error) = result {
                error.write_unraisable(py, Some(invocation.bind(py)));
            }
        });
    }
}
