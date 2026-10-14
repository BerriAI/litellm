use std::future::{Future, pending, poll_fn};
use std::pin::pin;
use std::task::Poll;

use futures_util::future::BoxFuture;
use litellm_host::{
    hooks::{ArgumentHooks, Cancelled, OutcomeHooks, ProviderHooks, ResponseHooks},
    interceptors::{ExecutionFacts, RawResponse, RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
};
use parking_lot::Mutex;
use pyo3::{
    exceptions::PyBaseException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::hooks::is_cancellation;
use crate::{Awaiter, Hooks, PythonGroup, PythonOwned, PythonSurface};

impl PythonSurface {
    /// Mounts a group on the Python surface. A cancellation the caller's task resumes a
    /// hook with ends that hook's stage there, whatever the hook would have done with it.
    pub fn group(hooks: impl PythonGroup + 'static) -> Box<dyn PythonGroup> {
        Box::new(Guarded(hooks))
    }
}

struct Guarded<H>(H);

impl<H: PythonOwned> PythonOwned for Guarded<H> {
    fn close(&mut self, py: Python<'_>) {
        self.0.close(py);
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.0.traverse(visit)
    }
}

/// The caller's task as one stage hands it to its hook.
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

    async fn guard<T, E>(
        &self,
        hook: impl Future<Output = Result<T, E>>,
        cancelled: impl FnOnce(PyErr) -> E,
    ) -> Result<T, E> {
        let mut hook = pin!(hook);
        let mut cancelled = Some(cancelled);
        poll_fn(|context| match hook.as_mut().poll(context) {
            Poll::Ready(result) => Poll::Ready(result),
            Poll::Pending => match self.cancelled.lock().take() {
                Some(error) => Poll::Ready(Err(cancelled.take().expect("settled once")(error))),
                None => Poll::Pending,
            },
        })
        .await
    }
}

impl Awaiter for Stage<'_> {
    fn awaited(&self, awaitable: Py<PyAny>) -> BoxFuture<'_, PyResult<Py<PyAny>>> {
        Box::pin(async move {
            match self.task.awaited(awaitable).await {
                Err(error) if Python::attach(|py| is_cancellation(py, &error)) => {
                    *self.cancelled.lock() = Some(error);
                    pending().await
                }
                answer => answer,
            }
        })
    }
}

impl<H: PythonGroup> ArgumentHooks<PythonSurface> for Guarded<H> {
    async fn prepare<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<Py<PyDict>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.prepare(&stage, arguments, started_at), |error| error)
            .await
    }

    fn prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        self.0.prepared(arguments)
    }
}

impl<H: PythonGroup> ProviderHooks<PythonSurface> for Guarded<H> {
    async fn before_provider_request<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        let stage = Stage::new(task);
        stage
            .guard(
                self.0.before_provider_request(&stage, wire, context),
                |error| error,
            )
            .await
    }

    async fn after_provider_response<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        raw: &RawResponse,
    ) -> PyResult<()> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.after_provider_response(&stage, raw), |error| error)
            .await
    }

    async fn result_ready<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        facts: &ExecutionFacts,
    ) -> PyResult<()> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.result_ready(&stage, facts), |error| error)
            .await
    }
}

impl<H: PythonGroup> ResponseHooks<PythonSurface> for Guarded<H> {
    async fn transform<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<Py<PyAny>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.transform(&stage, response, timing), |error| error)
            .await
    }

    fn stream_opened(&self, head: &Py<PyAny>) -> PyResult<()> {
        self.0.stream_opened(head)
    }

    fn stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()> {
        self.0.stream_chunk(chunk)
    }
}

impl<H: PythonGroup> OutcomeHooks<PythonSurface> for Guarded<H> {
    async fn started<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        start_time: f64,
    ) -> Result<(), Cancelled<PyErr>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.started(&stage, start_time), Cancelled)
            .await
    }

    async fn succeeded<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        timing: Timing,
        response: &Py<PyAny>,
    ) -> Result<(), Cancelled<PyErr>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.succeeded(&stage, timing, response), Cancelled)
            .await
    }

    async fn failed<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        timing: Timing,
        origin: FailureOrigin,
        error: &Py<PyBaseException>,
    ) -> Result<(), Cancelled<PyErr>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.failed(&stage, timing, origin, error), Cancelled)
            .await
    }

    async fn cancelled<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        timing: Timing,
    ) -> Result<(), Cancelled<PyErr>> {
        let stage = Stage::new(task);
        stage
            .guard(self.0.cancelled(&stage, timing), Cancelled)
            .await
    }
}

impl PythonOwned for Hooks {
    fn close(&mut self, py: Python<'_>) {
        for group in self.groups_mut() {
            group.close(py);
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.groups().try_for_each(|group| group.traverse(visit))
    }
}
