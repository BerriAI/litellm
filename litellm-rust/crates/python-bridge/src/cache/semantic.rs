use std::{collections::VecDeque, time::Duration};

use litellm_cache::Error;
use litellm_host_python::{Execution, ExecutionBody, ExecutionStep, run_async};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyException, PyRuntimeError},
    prelude::*,
};
use serde_json::Value;

use super::{
    cache_error,
    embedder::{PythonEmbedder, with_prepared_embedding},
    native::{NativeResponseCache, SemanticReply},
    request::{NativeRequest, now},
};

pub(super) enum SemanticOperation {
    Lookup(NativeRequest),
    /// A lookup that also reports the similarity, as `SemanticReply`.
    LookupSemantic(NativeRequest),
    Store(NativeRequest, Value),
    StoreBatch(VecDeque<(NativeRequest, Value)>),
}

/// What an exception from the Python embedder means for the operation.
#[derive(Clone, Copy)]
pub(super) enum EmbeddingFailure {
    /// Raise the Python exception unchanged.
    Propagate,
    /// Treat the embedding as unavailable and let the backend report that.
    Unavailable,
}

enum Phase {
    Start,
    AwaitingEmbedding,
    AwaitingBackend,
}

/// Runs a semantic cache operation whose embedding comes from Python: await the Python
/// embedder in the caller's event loop, seed the native backend with the vector, await the
/// backend, and repeat for each entry of a batch.
pub(super) struct SemanticExecution {
    service: NativeResponseCache,
    embedder: PythonEmbedder,
    failure: EmbeddingFailure,
    operation: SemanticOperation,
    pending: Option<(NativeRequest, Option<Value>)>,
    phase: Phase,
    now: Duration,
}

impl SemanticExecution {
    pub(super) fn new(
        service: NativeResponseCache,
        embedder: PythonEmbedder,
        failure: EmbeddingFailure,
        operation: SemanticOperation,
    ) -> Self {
        Self {
            service,
            embedder,
            failure,
            operation,
            pending: None,
            phase: Phase::Start,
            now: now(),
        }
    }

    /// Takes the next entry of the operation; `None` once a batch is exhausted.
    fn next_pending(&mut self) -> Option<(NativeRequest, Option<Value>)> {
        match &mut self.operation {
            SemanticOperation::Lookup(request) | SemanticOperation::LookupSemantic(request) => {
                Some((request.clone(), None))
            }
            SemanticOperation::Store(request, response) => {
                Some((request.clone(), Some(std::mem::take(response))))
            }
            SemanticOperation::StoreBatch(queue) => queue
                .pop_front()
                .map(|(request, response)| (request, Some(response))),
        }
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<ExecutionStep> {
        let Some(pending) = self.next_pending() else {
            return Ok(ExecutionStep::Return(py.None()));
        };
        let (request, response) = &pending;
        let enabled = match response {
            None => request.controls.reads(),
            Some(_) => request.controls.writes(),
        };
        let input = enabled
            .then(|| self.service.embedding_input(request))
            .flatten();
        self.pending = Some(pending);
        let Some(input) = input else {
            return self.backend_step(py, Err(Error::Unavailable));
        };
        let awaitable =
            self.embedder
                .async_embedding(py, &input.prompt, input.metadata.as_ref())?;
        self.phase = Phase::AwaitingEmbedding;
        Ok(ExecutionStep::Await(awaitable))
    }

    fn embedded(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<ExecutionStep> {
        let seed = match result {
            Ok(vector) => {
                PythonEmbedder::extract(vector.into_bound(py)).map_err(|_| Error::Unavailable)
            }
            Err(error) => match self.embedding_failure() {
                EmbeddingFailure::Propagate => return Err(error),
                EmbeddingFailure::Unavailable if error.is_instance_of::<PyException>(py) => {
                    Err(Error::Unavailable)
                }
                EmbeddingFailure::Unavailable => return Err(error),
            },
        };
        self.backend_step(py, seed)
    }

    /// Python's semantic lookups catch embedding errors and stamp a similarity of `0.0`.
    fn embedding_failure(&self) -> EmbeddingFailure {
        match self.operation {
            SemanticOperation::LookupSemantic(_) => EmbeddingFailure::Unavailable,
            _ => self.failure,
        }
    }

    fn backend_step(
        &mut self,
        py: Python<'_>,
        seed: Result<Vec<f32>, Error>,
    ) -> PyResult<ExecutionStep> {
        self.phase = Phase::AwaitingBackend;
        let (request, response) = self.pending.take().ok_or_else(|| {
            PyRuntimeError::new_err("semantic execution resumed without a pending operation")
        })?;
        let service = self.service.clone();
        let now = self.now;
        let with_similarity = matches!(self.operation, SemanticOperation::LookupSemantic(_));
        let future = async move {
            match response {
                None if with_similarity => service
                    .async_lookup_semantic(&request, now)
                    .await
                    .map(|lookup| Reply::Semantic(lookup.into())),
                None => service.async_lookup(&request, now).await.map(Reply::Plain),
                Some(response) => service
                    .async_store(&request, response, now)
                    .await
                    .map(|_| Reply::Plain(None)),
            }
        };
        let awaitable = run_async(py, with_prepared_embedding(seed, future), cache_error)?;
        Ok(ExecutionStep::Await(awaitable.unbind()))
    }

    fn resume_py(
        &mut self,
        py: Python<'_>,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<ExecutionStep> {
        match (&self.phase, result) {
            (Phase::Start, None) => self.start(py),
            (Phase::AwaitingEmbedding, Some(result)) => self.embedded(py, result),
            (Phase::AwaitingBackend, Some(Err(error))) => Err(error),
            (Phase::AwaitingBackend, Some(Ok(value))) => {
                let more = matches!(
                    &self.operation,
                    SemanticOperation::StoreBatch(queue) if !queue.is_empty()
                );
                if more {
                    self.phase = Phase::Start;
                    return self.start(py);
                }
                Ok(ExecutionStep::Return(value))
            }
            _ => Err(PyRuntimeError::new_err(
                "invalid semantic cache execution state",
            )),
        }
    }
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum Reply {
    Plain(Option<Value>),
    Semantic(SemanticReply),
}

impl ExecutionBody for SemanticExecution {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| self.resume_py(py, result))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.embedder.traverse(visit)
    }
}

pub(super) fn drive(py: Python<'_>, body: SemanticExecution) -> PyResult<Bound<'_, PyAny>> {
    let execution = Py::new(py, Execution::new(body))?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive")?
        .call1((execution,))
}
