use std::future::{Future, pending, poll_fn};
use std::pin::pin;
use std::task::Poll;

use futures_util::future::BoxFuture;
use litellm_host::{
    hooks::CallHooks,
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

trait ChainHooks: PythonOwned {
    fn prepare_arguments<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> BoxFuture<'a, PyResult<Py<PyDict>>>;
    fn arguments_prepared(&self, arguments: &Py<PyDict>) -> PyResult<()>;
    fn before_provider_request<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &'a RequestContext,
    ) -> BoxFuture<'a, PyResult<Box<WireRequest>>>;
    fn transform_response<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        timing: Timing,
    ) -> BoxFuture<'a, PyResult<Py<PyAny>>>;
    fn on_event<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        event: PythonCallEvent<'a>,
    ) -> BoxFuture<'a, PyResult<()>>;
    fn on_stream_open(&self, head: &Py<PyAny>) -> PyResult<()>;
    fn on_stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()>;
}

impl<H: PythonCallHooks> ChainHooks for H {
    fn prepare_arguments<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> BoxFuture<'a, PyResult<Py<PyDict>>> {
        Box::pin(CallHooks::prepare_arguments(
            self, task, arguments, started_at,
        ))
    }

    fn arguments_prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        CallHooks::arguments_prepared(self, arguments)
    }

    fn before_provider_request<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &'a RequestContext,
    ) -> BoxFuture<'a, PyResult<Box<WireRequest>>> {
        Box::pin(CallHooks::before_provider_request(
            self, task, wire, context,
        ))
    }

    fn transform_response<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        timing: Timing,
    ) -> BoxFuture<'a, PyResult<Py<PyAny>>> {
        Box::pin(CallHooks::transform_response(self, task, response, timing))
    }

    fn on_event<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        event: PythonCallEvent<'a>,
    ) -> BoxFuture<'a, PyResult<()>> {
        Box::pin(CallHooks::on_event(self, task, event))
    }

    fn on_stream_open(&self, head: &Py<PyAny>) -> PyResult<()> {
        CallHooks::on_stream_open(self, head)
    }

    fn on_stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()> {
        CallHooks::on_stream_chunk(self, chunk)
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

#[derive(Default)]
pub struct HookChain {
    hooks: Vec<Box<dyn ChainHooks>>,
}

impl HookChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, hooks: impl PythonCallHooks + 'static) -> Self {
        self.hooks.push(Box::new(hooks));
        self
    }
}

impl CallHooks<PythonRuntime> for HookChain {
    async fn prepare_arguments<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<Py<PyDict>> {
        let stage = Stage::new(task);
        let mut arguments = arguments;
        for hooks in &self.hooks {
            arguments = stage
                .guard(hooks.prepare_arguments(&stage, arguments, started_at))
                .await?;
        }
        Ok(arguments)
    }

    fn arguments_prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        self.hooks
            .iter()
            .try_for_each(|hooks| hooks.arguments_prepared(arguments))
    }

    async fn before_provider_request<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        let stage = Stage::new(task);
        let mut wire = wire;
        for hooks in &self.hooks {
            wire = stage
                .guard(hooks.before_provider_request(&stage, wire, context))
                .await?;
        }
        Ok(wire)
    }

    async fn transform_response<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<Py<PyAny>> {
        let stage = Stage::new(task);
        let mut response = response;
        for hooks in &self.hooks {
            response = stage
                .guard(hooks.transform_response(&stage, response, timing))
                .await?;
        }
        Ok(response)
    }

    async fn on_event<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        event: PythonCallEvent<'_>,
    ) -> PyResult<()> {
        let stage = Stage::new(task);
        let terminal = is_terminal(&event);
        for hooks in &self.hooks {
            let result = stage.guard(hooks.on_event(&stage, event.clone())).await;
            notification_result(terminal, result)?;
        }
        Ok(())
    }

    fn on_stream_open(&self, head: &Py<PyAny>) -> PyResult<()> {
        self.hooks
            .iter()
            .try_for_each(|hooks| hooks.on_stream_open(head))
    }

    fn on_stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()> {
        self.hooks
            .iter()
            .try_for_each(|hooks| hooks.on_stream_chunk(chunk))
    }
}

impl PythonOwned for HookChain {
    fn close(&mut self, py: Python<'_>) {
        for hooks in &mut self.hooks {
            hooks.close(py);
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.hooks
            .iter()
            .try_for_each(|hooks| hooks.traverse(visit))
    }
}
