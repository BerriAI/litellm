use std::{future::Future, sync::Arc};

use litellm_cache::Error;
use litellm_cache_valkey_semantic::Embedder;
use litellm_host_python::to_py;
use pyo3::{PyTraverseError, PyVisit, prelude::*};
use serde_json::Value;

#[derive(Clone)]
pub(super) struct PythonEmbedder {
    sync_embed: Arc<Py<PyAny>>,
    async_embed_callable: Arc<Py<PyAny>>,
}

impl PythonEmbedder {
    pub(super) fn from_backend(backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            sync_embed: Arc::new(backend.getattr("_get_embedding")?.unbind()),
            async_embed_callable: Arc::new(backend.getattr("_get_async_embedding")?.unbind()),
        })
    }

    pub(super) fn async_embed_awaitable<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        metadata: &Option<Value>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let metadata = to_py(py, metadata)?;
        self.async_embed_callable.bind(py).call1((prompt, metadata))
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&*self.sync_embed)?;
        visit.call(&*self.async_embed_callable)
    }
}

impl Embedder for PythonEmbedder {
    fn embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        let result = Python::attach(|py| -> PyResult<Vec<f64>> {
            let metadata = to_py(py, &metadata)?;
            self.sync_embed
                .bind(py)
                .call1((prompt, metadata))?
                .extract()
        })
        .map_err(|_| Error::Unavailable)?;
        Ok(result.into_iter().map(|value| value as f32).collect())
    }

    #[expect(
        clippy::manual_async_fn,
        reason = "the shared Embedder trait uses an impl Future return"
    )]
    fn async_embed(
        &self,
        _prompt: &str,
        _metadata: Option<&Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send {
        async { Err(Error::Unavailable) }
    }
}
