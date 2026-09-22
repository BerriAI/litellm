use std::collections::VecDeque;

use litellm_cache::Error;
use litellm_cache_redis_semantic::prompt_from_context;
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
    native::NativeResponseCache,
    request::{NativeRequest, now},
};

pub(super) enum SemanticOperation {
    Lookup(NativeRequest),
    Store(NativeRequest, Value),
    StoreBatch(VecDeque<(NativeRequest, Value)>),
}

enum Phase {
    Start,
    AwaitingEmbedding,
    AwaitingBackend,
}

pub(super) struct SemanticBody {
    service: NativeResponseCache,
    operation: SemanticOperation,
    pending: Option<(NativeRequest, Option<Value>)>,
    phase: Phase,
}

impl SemanticBody {
    pub(super) fn new(service: NativeResponseCache, operation: SemanticOperation) -> Self {
        Self {
            service,
            operation,
            pending: None,
            phase: Phase::Start,
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
        let future = async move {
            match response {
                None => service.async_lookup(&request, now()).await,
                Some(response) => service
                    .async_store(&request, response, now())
                    .await
                    .map(|_| None),
            }
        };
        let awaitable = run_async(py, with_prepared_embedding(seed, future), cache_error)?;
        Ok(ExecutionStep::Await(awaitable.unbind()))
    }
}

impl ExecutionBody for SemanticBody {
    fn resume(&mut self, mut result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| {
            loop {
                match self.phase {
                    Phase::Start => {
                        if result.is_some() {
                            return Err(PyRuntimeError::new_err(
                                "semantic execution received a result before starting",
                            ));
                        }
                        if self.pending.is_none() {
                            match &mut self.operation {
                                SemanticOperation::Lookup(request) => {
                                    self.pending = Some((request.clone(), None));
                                }
                                SemanticOperation::Store(request, response) => {
                                    let response = std::mem::replace(response, Value::Null);
                                    self.pending = Some((request.clone(), Some(response)));
                                }
                                SemanticOperation::StoreBatch(queue) => {
                                    let Some((request, response)) = queue.pop_front() else {
                                        return Ok(ExecutionStep::Return(py.None()));
                                    };
                                    self.pending = Some((request, Some(response)));
                                }
                            }
                        }
                        let (request, _) = self.pending.as_ref().ok_or_else(|| {
                            PyRuntimeError::new_err("semantic execution has no pending operation")
                        })?;
                        let semantic = NativeResponseCache::semantic_request(request);
                        let Some(prompt) = prompt_from_context(&semantic.context) else {
                            return self.backend_step(py, Err(Error::Unavailable));
                        };
                        let embedder = self.service.semantic_embedder().ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "semantic execution requires a redis-semantic backend",
                            )
                        })?;
                        let coroutine = embedder.async_embedding_coroutine(
                            py,
                            &prompt,
                            semantic.context.metadata.as_ref(),
                        )?;
                        self.phase = Phase::AwaitingEmbedding;
                        return Ok(ExecutionStep::Await(coroutine));
                    }
                    Phase::AwaitingEmbedding => {
                        let result = result.take().ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "semantic execution expected an embedding result",
                            )
                        })?;
                        let seed = match result {
                            Ok(value) => PythonEmbedder::extract(value.into_bound(py))
                                .map_err(|_| Error::Unavailable),
                            Err(error) => {
                                if !error.is_instance_of::<PyException>(py) {
                                    return Err(error);
                                }
                                Err(Error::Unavailable)
                            }
                        };
                        return self.backend_step(py, seed);
                    }
                    Phase::AwaitingBackend => {
                        let result = result.take().ok_or_else(|| {
                            PyRuntimeError::new_err("semantic execution expected a backend result")
                        })?;
                        let value = match result {
                            Ok(value) => value,
                            Err(error) => return Err(error),
                        };
                        let more = matches!(
                            &self.operation,
                            SemanticOperation::StoreBatch(queue) if !queue.is_empty()
                        );
                        if more {
                            self.phase = Phase::Start;
                            continue;
                        }
                        return Ok(ExecutionStep::Return(value));
                    }
                }
            }
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(embedder) = self.service.semantic_embedder() {
            embedder.traverse(visit)?;
        }
        Ok(())
    }
}

pub(super) fn drive(py: Python<'_>, body: SemanticBody) -> PyResult<Bound<'_, PyAny>> {
    let execution = Py::new(py, Execution::new(body))?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive")?
        .call1((execution,))
}
