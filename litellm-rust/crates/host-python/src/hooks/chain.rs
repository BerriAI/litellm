use std::future::{Future, pending, poll_fn};
use std::pin::pin;
use std::task::Poll;

use futures_util::future::BoxFuture;
use litellm_host::{
    hooks::{CallHooks, ChainRuntime, DynCallHooks, HookGroupAdapter},
    interceptors::{RequestContext, WireRequest},
    lifecycle::{CallEvent, Timing},
};
use parking_lot::Mutex;
use pyo3::{
    exceptions::PyException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{Awaiter, PythonCallEvent, PythonCallHooks, PythonOwned, PythonRuntime};

pub type HookChain = litellm_host::hooks::HookChain<PythonRuntime>;

pub trait PythonHookGroup: DynCallHooks<PythonRuntime> + PythonOwned {}
impl<H: DynCallHooks<PythonRuntime> + PythonOwned> PythonHookGroup for H {}

impl ChainRuntime for PythonRuntime {
    type Group = dyn PythonHookGroup;
}

impl<H: PythonCallHooks + 'static> HookGroupAdapter<H> for PythonRuntime {
    fn adapt_group(hooks: H) -> Box<dyn PythonHookGroup> {
        Box::new(PythonHookAdapter(hooks))
    }
}

struct PythonHookAdapter<H>(H);

impl<H: PythonOwned> PythonOwned for PythonHookAdapter<H> {
    fn close(&mut self, py: Python<'_>) {
        self.0.close(py);
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.0.traverse(visit)
    }
}

fn is_cancellation(error: &PyErr) -> bool {
    Python::attach(|py| !error.is_instance_of::<PyException>(py))
}

/// The caller's task as one chain stage hands it to its hooks. A cancellation the task
/// resumes with ends the stage there: the hook that awaited it never runs again.
struct Stage<'a> {
    task: &'a dyn Awaiter,
    cancelled: Mutex<Option<PyErr>>,
}

impl<'a> Stage<'a> {
    fn new(task: &'a dyn Awaiter) -> Self {
        Self {
            task,
            cancelled: Mutex::new(None),
        }
    }

    async fn guard<T>(&self, hook: impl Future<Output = PyResult<T>>) -> PyResult<T> {
        let mut hook = pin!(hook);
        poll_fn(|context| match hook.as_mut().poll(context) {
            Poll::Ready(result) => Poll::Ready(result),
            Poll::Pending => self
                .cancelled
                .lock()
                .take()
                .map_or(Poll::Pending, |error| Poll::Ready(Err(error))),
        })
        .await
    }
}

impl Awaiter for Stage<'_> {
    fn awaited(&self, awaitable: Py<PyAny>) -> BoxFuture<'_, PyResult<Py<PyAny>>> {
        Box::pin(async move {
            match self.task.awaited(awaitable).await {
                Err(error) if is_cancellation(&error) => {
                    *self.cancelled.lock() = Some(error);
                    pending().await
                }
                answer => answer,
            }
        })
    }
}

fn is_terminal<Response, Error, Raw>(event: &CallEvent<Response, Error, Raw>) -> bool {
    matches!(
        event,
        CallEvent::Succeeded { .. } | CallEvent::Failed { .. }
    )
}

/// A terminal notification reaches every hook, so an ordinary error from one is reported
/// and the next hook still runs.
fn notification_result(terminal: bool, result: PyResult<()>) -> PyResult<()> {
    match result {
        Err(error) if terminal && !is_cancellation(&error) => {
            Python::attach(|py| error.write_unraisable(py, None));
            Ok(())
        }
        result => result,
    }
}

impl<H: PythonCallHooks> CallHooks<PythonRuntime> for PythonHookAdapter<H> {
    async fn prepare_arguments<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<Py<PyDict>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.prepare_arguments(&stage, arguments, started_at))
            .await
    }

    fn arguments_prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        self.0.arguments_prepared(arguments)
    }

    async fn before_provider_request<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.before_provider_request(&stage, wire, context))
            .await
    }

    async fn transform_response<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<Py<PyAny>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.transform_response(&stage, response, timing))
            .await
    }

    async fn on_event<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        event: PythonCallEvent<'_>,
    ) -> PyResult<()> {
        let stage = Stage::new(task);
        let terminal = is_terminal(&event);
        let result = stage.guard(self.0.on_event(&stage, event)).await;
        notification_result(terminal, result)
    }

    fn on_stream_open(&self, head: &Py<PyAny>) -> PyResult<()> {
        self.0.on_stream_open(head)
    }

    fn on_stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()> {
        self.0.on_stream_chunk(chunk)
    }
}

impl PythonOwned for HookChain {
    fn close(&mut self, py: Python<'_>) {
        for hooks in self.groups_mut() {
            hooks.close(py);
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.groups().try_for_each(|hooks| hooks.traverse(visit))
    }
}
