use std::{sync::Arc, time::Duration};

use litellm_cache::SemanticCacheContext;
use litellm_cache_response::{ResponseCache, ResponseCacheCodec, ResponseCacheRequest};
use litellm_cache_valkey_semantic::{
    Embedder, PreparedEmbedding, ValkeySemanticCache, prompt_from_context,
};
use litellm_host_python::{Execution, ExecutionBody, ExecutionStep, run_async};
use pyo3::{PyTraverseError, PyVisit, exceptions::PyRuntimeError, prelude::*};
use serde_json::Value;

use super::{cache_error, embedder::PythonEmbedder};

pub(super) enum Op {
    Lookup,
    Store(Value),
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
    request: ResponseCacheRequest<SemanticCacheContext>,
    op: Op,
    now: Duration,
    state: State,
}

impl SemanticEmbedExecution {
    pub(super) fn lookup(
        backend: Arc<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>,
        embedder: PythonEmbedder,
        request: ResponseCacheRequest<SemanticCacheContext>,
        now: Duration,
    ) -> Self {
        Self {
            backend,
            embedder,
            request,
            op: Op::Lookup,
            now,
            state: State::Start,
        }
    }

    pub(super) fn store(
        backend: Arc<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>,
        embedder: PythonEmbedder,
        request: ResponseCacheRequest<SemanticCacheContext>,
        response: Value,
        now: Duration,
    ) -> Self {
        Self {
            backend,
            embedder,
            request,
            op: Op::Store(response),
            now,
            state: State::Start,
        }
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<ExecutionStep> {
        let Some(prompt) = prompt_from_context(&self.request.context) else {
            let cache = Arc::new(ResponseCache::new(Arc::clone(&self.backend)));
            self.state = State::AwaitingStorage;
            return storage_step(py, cache, self.request.clone(), &self.op, self.now);
        };
        let awaitable =
            self.embedder
                .async_embed_awaitable(py, &prompt, &self.request.context.metadata)?;
        self.state = State::AwaitingEmbedding;
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
                let backend = self.backend.with_embedder(PreparedEmbedding(
                    values.into_iter().map(|value| value as f32).collect(),
                ));
                let cache = Arc::new(ResponseCache::new(Arc::new(backend)));
                self.state = State::AwaitingStorage;
                storage_step(py, cache, self.request.clone(), &self.op, self.now)
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

fn storage_step<E: Embedder>(
    py: Python<'_>,
    cache: Arc<ResponseCache<ValkeySemanticCache<E, ResponseCacheCodec>>>,
    request: ResponseCacheRequest<SemanticCacheContext>,
    op: &Op,
    now: Duration,
) -> PyResult<ExecutionStep> {
    let awaitable = match op {
        Op::Lookup => run_async(
            py,
            async move { cache.async_lookup(&request, now).await },
            cache_error,
        )?,
        Op::Store(response) => {
            let response = response.clone();
            run_async(
                py,
                async move { cache.async_store(&request, response, now).await },
                cache_error,
            )?
        }
    };
    Ok(ExecutionStep::Await(awaitable.unbind()))
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
