use std::{future::Future, sync::Arc};

use litellm_cache::Error;
use litellm_cache_valkey_semantic::Embedder;
use litellm_host_python::to_py;
use pyo3::prelude::*;
use serde_json::Value;

#[derive(Clone)]
pub(super) struct PythonEmbedder {
    sync_embed: Arc<Py<PyAny>>,
    async_embed: Arc<Py<PyAny>>,
}

impl PythonEmbedder {
    pub(super) fn from_backend(backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            sync_embed: Arc::new(backend.getattr("_get_embedding")?.unbind()),
            async_embed: Arc::new(backend.getattr("_get_async_embedding")?.unbind()),
        })
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

    fn async_embed(
        &self,
        prompt: &str,
        metadata: Option<&Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send {
        let callable = Arc::clone(&self.async_embed);
        let prompt = prompt.to_owned();
        let metadata = metadata.cloned();
        async move {
            let future = Python::attach(|py| -> PyResult<_> {
                let metadata = to_py(py, &metadata)?;
                let awaitable = callable.bind(py).call1((prompt, metadata))?;
                pyo3_async_runtimes::tokio::into_future(awaitable)
            })
            .map_err(|_| Error::Unavailable)?;
            let result = future.await.map_err(|_| Error::Unavailable)?;
            let result = Python::attach(|py| result.bind(py).extract::<Vec<f64>>())
                .map_err(|_| Error::Unavailable)?;
            Ok(result.into_iter().map(|value| value as f32).collect())
        }
    }
}
