use litellm_host::interceptors::{RequestContext, WireRequest};
use litellm_host::lifecycle::Timing;
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{HookResume, HookStep, PythonCallEvent, PythonCallHooks, PythonOwned, missing_state};

pub(super) enum ChainStep<T> {
    Ready(T),
    Await(Py<PyAny>),
}

enum Continuation<H> {
    Arguments(HookResume<H, Py<PyDict>>),
    Wire(HookResume<H, Box<WireRequest>>),
    Response(HookResume<H, Py<PyAny>>),
    Event(HookResume<H, ()>),
}

pub(super) struct HookAdapter<H> {
    hooks: H,
    continuation: Option<Continuation<H>>,
}

impl<H> HookAdapter<H> {
    pub(super) fn new(hooks: H) -> Self {
        Self {
            hooks,
            continuation: None,
        }
    }

    fn step<T>(
        &mut self,
        step: HookStep<H, T>,
        continuation: impl FnOnce(HookResume<H, T>) -> Continuation<H>,
    ) -> ChainStep<T> {
        match step {
            HookStep::Ready(value) => ChainStep::Ready(value),
            HookStep::Await(awaitable, resume) => {
                self.continuation = Some(continuation(resume));
                ChainStep::Await(awaitable)
            }
        }
    }
}

pub(super) trait ChainHooks: PythonOwned {
    fn prepare_arguments(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<ChainStep<Py<PyDict>>>;
    fn resume_arguments(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyDict>>>;
    fn prepare_request(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
    ) -> PyResult<ChainStep<Py<PyDict>>>;
    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<ChainStep<Box<WireRequest>>>;
    fn resume_wire(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Box<WireRequest>>>;
    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<ChainStep<Py<PyAny>>>;
    fn resume_response(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyAny>>>;
    fn on_event(&mut self, py: Python<'_>, event: PythonCallEvent<'_>) -> PyResult<ChainStep<()>>;
    fn resume_event(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<()>>;
    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()>;
    fn on_stream_open(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()>;
    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()>;
    fn on_cancelled(&mut self, py: Python<'_>, timing: Timing);
}

impl<H: PythonCallHooks> ChainHooks for HookAdapter<H> {
    fn prepare_request(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        let step = self.hooks.prepare_request(py, arguments)?;
        Ok(self.step(step, Continuation::Arguments))
    }

    fn prepare_arguments(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        let step = self.hooks.prepare_arguments(py, arguments, started_at)?;
        Ok(self.step(step, Continuation::Arguments))
    }

    fn resume_arguments(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        let Some(Continuation::Arguments(resume)) = self.continuation.take() else {
            return Err(missing_state());
        };
        let step = resume(&mut self.hooks, py, result)?;
        Ok(self.step(step, Continuation::Arguments))
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<ChainStep<Box<WireRequest>>> {
        let step = self.hooks.before_provider_request(py, wire, context)?;
        Ok(self.step(step, Continuation::Wire))
    }

    fn resume_wire(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Box<WireRequest>>> {
        let Some(Continuation::Wire(resume)) = self.continuation.take() else {
            return Err(missing_state());
        };
        let step = resume(&mut self.hooks, py, result)?;
        Ok(self.step(step, Continuation::Wire))
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        let step = self.hooks.transform_response(py, response, timing)?;
        Ok(self.step(step, Continuation::Response))
    }

    fn resume_response(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        let Some(Continuation::Response(resume)) = self.continuation.take() else {
            return Err(missing_state());
        };
        let step = resume(&mut self.hooks, py, result)?;
        Ok(self.step(step, Continuation::Response))
    }

    fn on_event(&mut self, py: Python<'_>, event: PythonCallEvent<'_>) -> PyResult<ChainStep<()>> {
        let step = self.hooks.on_event(py, event)?;
        Ok(self.step(step, Continuation::Event))
    }

    fn resume_event(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<()>> {
        let Some(Continuation::Event(resume)) = self.continuation.take() else {
            return Err(missing_state());
        };
        let step = resume(&mut self.hooks, py, result)?;
        Ok(self.step(step, Continuation::Event))
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        self.hooks.arguments_prepared(py, arguments)
    }

    fn on_stream_open(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()> {
        self.hooks.on_stream_open(py, head)
    }

    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        self.hooks.on_stream_chunk(py, chunk)
    }

    fn on_cancelled(&mut self, py: Python<'_>, timing: Timing) {
        self.hooks.on_cancelled(py, timing)
    }
}

impl<H: PythonOwned> PythonOwned for HookAdapter<H> {
    fn close(&mut self, py: Python<'_>) {
        self.continuation = None;
        self.hooks.close(py);
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.hooks.traverse(visit)
    }
}
