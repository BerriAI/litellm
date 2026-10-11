use futures_util::future::BoxFuture;
use litellm_host::{
    hooks::{CallHooks, StreamPolicy},
    interceptors::{RequestContext, WireRequest},
    lifecycle::{CallEvent, Timing},
};
use pyo3::{
    exceptions::PyException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{PythonCallEvent, PythonCallHooks, PythonOwned, PythonRuntime, PythonTask};

trait ChainHooks: PythonOwned {
    fn prepare_arguments<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> BoxFuture<'a, PyResult<Py<PyDict>>>;
    fn prepare_request<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: Py<PyDict>,
    ) -> BoxFuture<'a, PyResult<Py<PyDict>>>;
    fn arguments_prepared<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: &'a Py<PyDict>,
    ) -> BoxFuture<'a, PyResult<()>>;
    fn stream_policy<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: &'a Py<PyDict>,
    ) -> BoxFuture<'a, PyResult<StreamPolicy>>;
    fn before_provider_request<'a>(
        &'a mut self,
        task: &'a PythonTask,
        wire: Box<WireRequest>,
        context: &'a RequestContext,
    ) -> BoxFuture<'a, PyResult<Box<WireRequest>>>;
    fn transform_response<'a>(
        &'a mut self,
        task: &'a PythonTask,
        response: Py<PyAny>,
        timing: Timing,
    ) -> BoxFuture<'a, PyResult<Py<PyAny>>>;
    fn on_event<'a>(
        &'a mut self,
        task: &'a PythonTask,
        event: PythonCallEvent<'a>,
    ) -> BoxFuture<'a, PyResult<()>>;
    fn on_stream_open<'a>(
        &'a mut self,
        task: &'a PythonTask,
        head: &'a Py<PyAny>,
    ) -> BoxFuture<'a, PyResult<()>>;
    fn on_stream_chunk<'a>(
        &'a mut self,
        task: &'a PythonTask,
        chunk: &'a Py<PyAny>,
    ) -> BoxFuture<'a, PyResult<()>>;
}

impl<H: PythonCallHooks> ChainHooks for H {
    fn prepare_arguments<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> BoxFuture<'a, PyResult<Py<PyDict>>> {
        Box::pin(CallHooks::prepare_arguments(
            self, task, arguments, started_at,
        ))
    }

    fn prepare_request<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: Py<PyDict>,
    ) -> BoxFuture<'a, PyResult<Py<PyDict>>> {
        Box::pin(CallHooks::prepare_request(self, task, arguments))
    }

    fn arguments_prepared<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: &'a Py<PyDict>,
    ) -> BoxFuture<'a, PyResult<()>> {
        Box::pin(CallHooks::arguments_prepared(self, task, arguments))
    }

    fn stream_policy<'a>(
        &'a mut self,
        task: &'a PythonTask,
        arguments: &'a Py<PyDict>,
    ) -> BoxFuture<'a, PyResult<StreamPolicy>> {
        Box::pin(CallHooks::stream_policy(self, task, arguments))
    }

    fn before_provider_request<'a>(
        &'a mut self,
        task: &'a PythonTask,
        wire: Box<WireRequest>,
        context: &'a RequestContext,
    ) -> BoxFuture<'a, PyResult<Box<WireRequest>>> {
        Box::pin(CallHooks::before_provider_request(
            self, task, wire, context,
        ))
    }

    fn transform_response<'a>(
        &'a mut self,
        task: &'a PythonTask,
        response: Py<PyAny>,
        timing: Timing,
    ) -> BoxFuture<'a, PyResult<Py<PyAny>>> {
        Box::pin(CallHooks::transform_response(self, task, response, timing))
    }

    fn on_event<'a>(
        &'a mut self,
        task: &'a PythonTask,
        event: PythonCallEvent<'a>,
    ) -> BoxFuture<'a, PyResult<()>> {
        Box::pin(CallHooks::on_event(self, task, event))
    }

    fn on_stream_open<'a>(
        &'a mut self,
        task: &'a PythonTask,
        head: &'a Py<PyAny>,
    ) -> BoxFuture<'a, PyResult<()>> {
        Box::pin(CallHooks::on_stream_open(self, task, head))
    }

    fn on_stream_chunk<'a>(
        &'a mut self,
        task: &'a PythonTask,
        chunk: &'a Py<PyAny>,
    ) -> BoxFuture<'a, PyResult<()>> {
        Box::pin(CallHooks::on_stream_chunk(self, task, chunk))
    }
}

/// Hooks composed as layers: the argument and request transformations walk inbound in
/// append order and the response transformation walks back outbound, so the last hook
/// appended is the innermost. Every notification reaches every hook, and a terminal
/// notification that raises an ordinary exception is reported without stopping the later
/// hooks.
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

fn is_terminal(event: &PythonCallEvent<'_>) -> bool {
    matches!(
        event,
        CallEvent::Succeeded { .. } | CallEvent::Failed { .. }
    )
}

fn notification_result(terminal: bool, result: PyResult<()>) -> PyResult<()> {
    match result {
        Err(error) if terminal => Python::attach(|py| {
            if !error.is_instance_of::<PyException>(py) {
                return Err(error);
            }
            error.write_unraisable(py, None);
            Ok(())
        }),
        result => result,
    }
}

impl CallHooks<PythonRuntime> for HookChain {
    async fn prepare_arguments(
        &mut self,
        task: &PythonTask,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<Py<PyDict>> {
        let mut prepared = arguments;
        for hooks in &mut self.hooks {
            prepared = hooks.prepare_arguments(task, prepared, started_at).await?;
        }
        Ok(prepared)
    }

    async fn prepare_request(
        &mut self,
        task: &PythonTask,
        arguments: Py<PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let mut prepared = arguments;
        for hooks in &mut self.hooks {
            prepared = hooks.prepare_request(task, prepared).await?;
        }
        Ok(prepared)
    }

    async fn arguments_prepared(
        &mut self,
        task: &PythonTask,
        arguments: &Py<PyDict>,
    ) -> PyResult<()> {
        for hooks in &mut self.hooks {
            hooks.arguments_prepared(task, arguments).await?;
        }
        Ok(())
    }

    async fn stream_policy(
        &mut self,
        task: &PythonTask,
        arguments: &Py<PyDict>,
    ) -> PyResult<StreamPolicy> {
        let mut policy = StreamPolicy::Relay;
        for hooks in &mut self.hooks {
            policy = policy.merge(hooks.stream_policy(task, arguments).await?);
        }
        Ok(policy)
    }

    async fn before_provider_request(
        &mut self,
        task: &PythonTask,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        let mut rewritten = wire;
        for hooks in &mut self.hooks {
            rewritten = hooks
                .before_provider_request(task, rewritten, context)
                .await?;
        }
        Ok(rewritten)
    }

    async fn transform_response(
        &mut self,
        task: &PythonTask,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<Py<PyAny>> {
        let mut transformed = response;
        for hooks in self.hooks.iter_mut().rev() {
            transformed = hooks.transform_response(task, transformed, timing).await?;
        }
        Ok(transformed)
    }

    async fn on_event(&mut self, task: &PythonTask, event: PythonCallEvent<'_>) -> PyResult<()> {
        let terminal = is_terminal(&event);
        for hooks in &mut self.hooks {
            let result = hooks.on_event(task, event.clone()).await;
            notification_result(terminal, result)?;
        }
        Ok(())
    }

    async fn on_stream_open(&mut self, task: &PythonTask, head: &Py<PyAny>) -> PyResult<()> {
        for hooks in &mut self.hooks {
            hooks.on_stream_open(task, head).await?;
        }
        Ok(())
    }

    async fn on_stream_chunk(&mut self, task: &PythonTask, chunk: &Py<PyAny>) -> PyResult<()> {
        for hooks in &mut self.hooks {
            hooks.on_stream_chunk(task, chunk).await?;
        }
        Ok(())
    }
}

impl PythonOwned for HookChain {
    fn close(&mut self, py: Python<'_>) {
        for hooks in &mut self.hooks {
            hooks.close(py);
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for hooks in &self.hooks {
            hooks.traverse(visit)?;
        }
        Ok(())
    }
}
