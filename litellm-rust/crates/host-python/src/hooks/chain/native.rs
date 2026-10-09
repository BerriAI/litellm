use litellm_host::{
    error::HookError,
    hooks::NativeHooks,
    interceptors::{RequestContext, WireRequest},
    lifecycle::{CallEvent, ExecutionEvent, Timing},
};
use pyo3::{
    exceptions::PyRuntimeError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{PythonCallEvent, PythonOwned};

use super::adapter::{ChainHooks, ChainStep};

pub(super) struct NativeAdapter<H>(pub(super) H);

fn rejection(error: HookError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

impl<H: NativeHooks> ChainHooks for NativeAdapter<H> {
    fn prepare_arguments(
        &mut self,
        _py: Python<'_>,
        arguments: Py<PyDict>,
        _started_at: f64,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        Ok(ChainStep::Ready(arguments))
    }

    fn resume_arguments(
        &mut self,
        _py: Python<'_>,
        _result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        Err(crate::missing_state())
    }

    fn prepare_request(
        &mut self,
        _py: Python<'_>,
        arguments: Py<PyDict>,
    ) -> PyResult<ChainStep<Py<PyDict>>> {
        Ok(ChainStep::Ready(arguments))
    }

    fn before_provider_request(
        &mut self,
        _py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<ChainStep<Box<WireRequest>>> {
        self.0
            .before_provider_request(wire, context)
            .map(ChainStep::Ready)
            .map_err(rejection)
    }

    fn resume_wire(
        &mut self,
        _py: Python<'_>,
        _result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Box<WireRequest>>> {
        Err(crate::missing_state())
    }

    fn transform_response(
        &mut self,
        _py: Python<'_>,
        response: Py<PyAny>,
        _timing: Timing,
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        Ok(ChainStep::Ready(response))
    }

    fn resume_response(
        &mut self,
        _py: Python<'_>,
        _result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Py<PyAny>>> {
        Err(crate::missing_state())
    }

    fn on_event(&mut self, _py: Python<'_>, event: PythonCallEvent<'_>) -> PyResult<ChainStep<()>> {
        let snapshot = event.snapshot();
        self.0.on_event(&snapshot);
        if let CallEvent::Execution(ExecutionEvent::ResultReady { facts }) = &snapshot {
            self.0.result_ready(facts).map_err(rejection)?;
        }
        Ok(ChainStep::Ready(()))
    }

    fn resume_event(
        &mut self,
        _py: Python<'_>,
        _result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<()>> {
        Err(crate::missing_state())
    }

    fn arguments_prepared(&mut self, _py: Python<'_>, _arguments: &Py<PyDict>) -> PyResult<()> {
        Ok(())
    }

    fn on_stream_open(&mut self, _py: Python<'_>, _head: &Py<PyAny>) -> PyResult<()> {
        Ok(())
    }

    fn on_stream_chunk(&mut self, _py: Python<'_>, _chunk: &Py<PyAny>) -> PyResult<()> {
        Ok(())
    }

    fn on_cancelled(&mut self, _py: Python<'_>, timing: Timing) {
        self.0.on_event(&CallEvent::Cancelled { timing });
    }
}

impl<H: NativeHooks> PythonOwned for NativeAdapter<H> {
    fn close(&mut self, _py: Python<'_>) {}

    fn traverse(&self, _visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}
