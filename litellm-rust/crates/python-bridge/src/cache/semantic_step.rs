use std::{sync::Arc, time::Duration};

use litellm_cache::SemanticCacheContext;
use litellm_cache_response::{ResponseCache, ResponseCacheCodec, ResponseCacheRequest};
use litellm_cache_valkey_semantic::{PreparedEmbedding, ValkeySemanticCache, prompt_from_context};
use litellm_host_python::{Execution, ExecutionBody, ExecutionStep, run_async};
use pyo3::{PyTraverseError, PyVisit, exceptions::PyRuntimeError, prelude::*};
use serde_json::Value;

use super::{cache_error, embedder::PythonEmbedder};

pub(super) enum Op {
    Lookup,
    Store(Value),
    StoreBatch(Vec<Value>),
}

#[derive(Clone, Copy)]
enum State {
    Start,
    AwaitingEmbedding,
    AwaitingStorage,
    Done,
}

pub(super) struct SemanticEmbedExecution {
    backend: Arc<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>,
    embedder: PythonEmbedder,
    requests: Vec<ResponseCacheRequest<SemanticCacheContext>>,
    op: Op,
    now: Option<Duration>,
    prepared: Vec<Option<Vec<f32>>>,
    index: usize,
    state: State,
}

impl SemanticEmbedExecution {
    pub(super) fn lookup(
        backend: Arc<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>,
        embedder: PythonEmbedder,
        request: ResponseCacheRequest<SemanticCacheContext>,
    ) -> Self {
        Self {
            backend,
            embedder,
            requests: vec![request],
            op: Op::Lookup,
            now: None,
            prepared: vec![None],
            index: 0,
            state: State::Start,
        }
    }

    pub(super) fn store(
        backend: Arc<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>,
        embedder: PythonEmbedder,
        request: ResponseCacheRequest<SemanticCacheContext>,
        response: Value,
    ) -> Self {
        Self {
            backend,
            embedder,
            requests: vec![request],
            op: Op::Store(response),
            now: None,
            prepared: vec![None],
            index: 0,
            state: State::Start,
        }
    }

    pub(super) fn store_batch(
        backend: Arc<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>,
        embedder: PythonEmbedder,
        requests: Vec<ResponseCacheRequest<SemanticCacheContext>>,
        responses: Vec<Value>,
    ) -> Self {
        Self {
            backend,
            embedder,
            prepared: vec![None; requests.len()],
            requests,
            op: Op::StoreBatch(responses),
            now: None,
            index: 0,
            state: State::Start,
        }
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<ExecutionStep> {
        if self.now.is_none() {
            self.now = Some(super::request::now());
        }
        while self.index < self.requests.len() {
            let request = &self.requests[self.index];
            let enabled = match &self.op {
                Op::Lookup => request.controls.reads(),
                Op::Store(_) | Op::StoreBatch(_) => request.controls.writes(),
            };
            if !enabled {
                self.index += 1;
                continue;
            }
            let Some(prompt) = prompt_from_context(&request.context) else {
                self.index += 1;
                continue;
            };
            let metadata = request.context.metadata.clone();
            let awaitable = self
                .embedder
                .async_embed_awaitable(py, &prompt, &metadata)?;
            self.state = State::AwaitingEmbedding;
            return Ok(ExecutionStep::Await(awaitable.unbind()));
        }
        self.state = State::AwaitingStorage;
        self.storage_step(py)
    }

    fn storage_step(&self, py: Python<'_>) -> PyResult<ExecutionStep> {
        let requests = self.requests.clone();
        let prepared = self.prepared.clone();
        let backend = Arc::clone(&self.backend);
        let now = self
            .now
            .ok_or_else(|| PyRuntimeError::new_err("semantic cache timestamp is unavailable"))?;
        let awaitable = match &self.op {
            Op::Lookup => {
                let Some(request) = requests.into_iter().next() else {
                    return Err(PyRuntimeError::new_err(
                        "semantic lookup requires one request",
                    ));
                };
                match prepared.into_iter().next().flatten() {
                    Some(values) => {
                        let backend = backend.with_embedder(PreparedEmbedding(values));
                        let cache = Arc::new(ResponseCache::new(Arc::new(backend)));
                        run_async(
                            py,
                            async move { cache.async_lookup(&request, now).await },
                            cache_error,
                        )?
                    }
                    None => {
                        let cache = Arc::new(ResponseCache::new(backend));
                        run_async(
                            py,
                            async move { cache.async_lookup(&request, now).await },
                            cache_error,
                        )?
                    }
                }
            }
            Op::Store(response) => {
                let Some(request) = requests.into_iter().next() else {
                    return Err(PyRuntimeError::new_err(
                        "semantic store requires one request",
                    ));
                };
                let response = response.clone();
                match prepared.into_iter().next().flatten() {
                    Some(values) => {
                        let backend = backend.with_embedder(PreparedEmbedding(values));
                        let cache = Arc::new(ResponseCache::new(Arc::new(backend)));
                        run_async(
                            py,
                            async move { cache.async_store(&request, response, now).await },
                            cache_error,
                        )?
                    }
                    None => {
                        let cache = Arc::new(ResponseCache::new(backend));
                        run_async(
                            py,
                            async move { cache.async_store(&request, response, now).await },
                            cache_error,
                        )?
                    }
                }
            }
            Op::StoreBatch(responses) => {
                let responses = responses.clone();
                run_async(
                    py,
                    async move {
                        for ((request, response), prepared) in
                            requests.into_iter().zip(responses).zip(prepared)
                        {
                            let Some(values) = prepared else {
                                continue;
                            };
                            let backend = backend.with_embedder(PreparedEmbedding(values));
                            let cache = ResponseCache::new(Arc::new(backend));
                            cache.async_store(&request, response, now).await?;
                        }
                        Ok(())
                    },
                    cache_error,
                )?
            }
        };
        Ok(ExecutionStep::Await(awaitable.unbind()))
    }

    fn resume_py(
        &mut self,
        py: Python<'_>,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<ExecutionStep> {
        match (self.state, result) {
            (State::Start, None) => self.start(py),
            (State::AwaitingEmbedding, Some(Ok(value))) => {
                let values = value.bind(py).extract::<Vec<f64>>()?;
                self.prepared[self.index] =
                    Some(values.into_iter().map(|value| value as f32).collect());
                self.index += 1;
                self.start(py)
            }
            (State::AwaitingStorage, Some(Ok(value))) => {
                self.state = State::Done;
                Ok(ExecutionStep::Return(value))
            }
            (_, Some(Err(error))) => Err(error),
            _ => Err(PyRuntimeError::new_err(
                "invalid semantic cache execution state",
            )),
        }
    }
}

impl ExecutionBody for SemanticEmbedExecution {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| self.resume_py(py, result))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.embedder.traverse(visit)
    }
}

pub(super) fn drive_semantic<'py>(
    py: Python<'py>,
    body: SemanticEmbedExecution,
) -> PyResult<Bound<'py, PyAny>> {
    let execution = Py::new(py, Execution::new(body))?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive")?
        .call1((execution,))
}
