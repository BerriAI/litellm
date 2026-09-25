use std::future::Future;

use litellm_cache::Error;
use litellm_host_python::to_py;
use pyo3::{PyTraverseError, PyVisit, prelude::*, types::PyDict};
use serde_json::Value;

tokio::task_local! {
    static PREPARED_EMBEDDING: Result<Vec<f32>, Error>;
}

/// Runs `future` with the vector the Python embedder already produced, so the backend's
/// `async_embed` never has to call back into Python from the runtime.
pub(super) fn with_prepared_embedding<F: Future>(
    vector: Result<Vec<f32>, Error>,
    future: F,
) -> impl Future<Output = F::Output> {
    PREPARED_EMBEDDING.scope(vector, future)
}

/// The Python object that owns embedding for a semantic backend.
pub(super) struct PythonEmbedder(Py<PyAny>);

impl Clone for PythonEmbedder {
    fn clone(&self) -> Self {
        Python::attach(|py| Self(self.0.clone_ref(py)))
    }
}

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
        metadata: Option<&Value>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("metadata", to_py(py, &metadata)?)?;
        Ok(kwargs)
    }

    /// The awaitable of `_get_async_embedding(prompt, metadata=...)`, to run in the caller's loop.
    pub(super) fn async_embedding(
        &self,
        py: Python<'_>,
        prompt: &str,
        metadata: Option<&Value>,
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

    fn embed_sync(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
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

    fn seeded_embedding() -> Result<Vec<f32>, Error> {
        PREPARED_EMBEDDING
            .try_with(Clone::clone)
            .unwrap_or(Err(Error::Unavailable))
    }
}

impl litellm_cache::semantic::Embedder for PythonEmbedder {
    fn embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        self.embed_sync(prompt, metadata)
    }

    fn async_embed(
        &self,
        _prompt: &str,
        _metadata: Option<&Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send {
        std::future::ready(Self::seeded_embedding())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn async_embed_returns_the_seeded_vector_or_unavailable() {
        Python::initialize();
        let object = Python::attach(|py| py.None());
        let embedder = PythonEmbedder::new(object);
        let scoped_embedder = embedder.clone();
        let scoped = with_prepared_embedding(Ok(vec![0.25]), async move {
            litellm_cache::semantic::Embedder::async_embed(&scoped_embedder, "prompt", None).await
        });
        assert_eq!(scoped.await, Ok(vec![0.25]));
        let unscoped =
            litellm_cache::semantic::Embedder::async_embed(&embedder, "prompt", None).await;
        assert_eq!(unscoped, Err(Error::Unavailable));
        let valkey = with_prepared_embedding(Ok(vec![0.5]), async move {
            litellm_cache::semantic::Embedder::async_embed(&embedder, "prompt", None).await
        });
        assert_eq!(valkey.await, Ok(vec![0.5]));
    }
}
