use std::ops::ControlFlow;

use litellm_coroutine::{Abandoned, Co, Reply};
use pyo3::prelude::*;

use crate::Error;
use crate::handle::Suspension;

pub(crate) type Answer = PyResult<Py<PyAny>>;

pub(crate) struct Yielded {
    pub(crate) suspension: Suspension,
    pub(crate) reply: Reply<Answer>,
}

/// Hands suspensions to the Python caller and receives its answers.
#[derive(Clone)]
pub(crate) struct PythonCaller {
    co: Co<Yielded>,
}

impl PythonCaller {
    /// Connects the call body to its coroutine's yield handle.
    pub(crate) fn new(co: Co<Yielded>) -> Self {
        Self { co }
    }

    /// Asks the caller to await Python work and return its value or exception.
    pub(crate) async fn await_python(&self, awaitable: Py<PyAny>) -> Answer {
        self.suspend(Suspension::Await(awaitable)).await
    }

    /// Hands the caller a stream head and waits for demand or closure.
    pub(crate) async fn open_stream(&self, head: Py<PyAny>) -> ControlFlow<()> {
        self.stream_boundary(Suspension::StreamOpen(head)).await
    }

    /// Delivers a chunk and waits for the caller to request another or close.
    pub(crate) async fn yield_chunk(&self, chunk: Py<PyAny>) -> ControlFlow<()> {
        self.stream_boundary(Suspension::StreamChunk(chunk)).await
    }

    /// Maps stream demand to continuation and any error to detachment.
    async fn stream_boundary(&self, suspension: Suspension) -> ControlFlow<()> {
        match self.suspend(suspension).await {
            Ok(_) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        }
    }

    /// Yields a suspension with its reply handle and waits for the answer.
    async fn suspend(&self, suspension: Suspension) -> Answer {
        match self.co.yield_(|reply| Yielded { suspension, reply }).await {
            Ok(answer) => answer,
            Err(Abandoned) => Err(Error::MissingState.into()),
        }
    }
}
