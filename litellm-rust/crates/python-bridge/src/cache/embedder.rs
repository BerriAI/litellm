use std::future::Future;

use litellm_cache::Error;
use litellm_cache_redis_semantic::Embedder;
use litellm_host_python::to_py;
use pyo3::{PyTraverseError, PyVisit, prelude::*, types::PyDict};
use serde_json::{Map, Value};

tokio::task_local! {
    static PREPARED_EMBEDDING: Result<Vec<f32>, Error>;
}

pub(super) fn with_prepared_embedding<F: Future>(
    vector: Result<Vec<f32>, Error>,
    future: F,
) -> impl Future<Output = F::Output> {
    PREPARED_EMBEDDING.scope(vector, future)
}

pub(super) struct PythonEmbedder(Py<PyAny>);

impl PythonEmbedder {
    pub(super) fn new(object: Py<PyAny>) -> Self {
        Self(object)
    }

    pub(super) fn object(&self) -> &Py<PyAny> {
        &self.0
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }

    fn metadata_kwargs<'py>(
        py: Python<'py>,
        metadata: &Map<String, Value>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kwargs = PyDict::new(py);
        if metadata.is_empty() {
            kwargs.set_item("metadata", py.None())?;
        } else {
            kwargs.set_item("metadata", to_py(py, metadata)?)?;
        }
        Ok(kwargs)
    }

    pub(super) fn async_embedding_coroutine(
        &self,
        py: Python<'_>,
        prompt: &str,
        metadata: &Map<String, Value>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = Self::metadata_kwargs(py, metadata)?;
        self.0
            .bind(py)
            .call_method("_get_async_embedding", (prompt,), Some(&kwargs))
            .map(Bound::unbind)
    }

    pub(super) fn extract(vector: Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        Ok(vector
            .extract::<Vec<f64>>()?
            .into_iter()
            .map(|value| value as f32)
            .collect())
    }
}

impl Embedder for PythonEmbedder {
    fn embed(&self, prompt: &str, metadata: &Map<String, Value>) -> Result<Vec<f32>, Error> {
        Python::attach(|py| {
            let kwargs = Self::metadata_kwargs(py, metadata)?;
            Self::extract(self.0.bind(py).call_method(
                "_get_embedding",
                (prompt,),
                Some(&kwargs),
            )?)
        })
        .map_err(|_| Error::Unavailable)
    }

    fn async_embed(
        &self,
        _prompt: &str,
        _metadata: &Map<String, Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send {
        let seeded = PREPARED_EMBEDDING
            .try_with(Clone::clone)
            .unwrap_or(Err(Error::Unavailable));
        std::future::ready(seeded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn async_embed_returns_the_seeded_vector_or_unavailable() {
        Python::initialize();
        let embedder = Python::attach(|py| PythonEmbedder::new(py.None()));
        let metadata = Map::new();
        let embedder_ref = &embedder;
        let metadata_ref = &metadata;
        assert_eq!(
            with_prepared_embedding(Ok(vec![0.5f32, 0.25]), async move {
                embedder_ref.async_embed("prompt", metadata_ref).await
            })
            .await,
            Ok(vec![0.5, 0.25])
        );
        assert_eq!(
            embedder.async_embed("prompt", &metadata).await,
            Err(Error::Unavailable)
        );
    }
}
