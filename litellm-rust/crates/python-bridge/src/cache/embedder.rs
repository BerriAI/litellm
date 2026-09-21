use std::future::Future;

use litellm_cache::Error;
use litellm_cache_redis_semantic::Embedder;
use litellm_host_python::to_py;
use pyo3::{PyTraverseError, PyVisit, prelude::*, types::PyDict};
use serde_json::{Map, Value};

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

    fn extract(vector: Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
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
        prompt: &str,
        metadata: &Map<String, Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send {
        let coroutine = Python::attach(|py| {
            let kwargs = Self::metadata_kwargs(py, metadata)?;
            self.0
                .bind(py)
                .call_method("_get_async_embedding", (prompt,), Some(&kwargs))
                .map(Bound::unbind)
        })
        .map_err(|_| Error::Unavailable);
        async move {
            let coroutine = coroutine?;
            let awaited = Python::attach(|py| {
                pyo3_async_runtimes::tokio::into_future(coroutine.into_bound(py))
            })
            .map_err(|_| Error::Unavailable)?
            .await
            .map_err(|_| Error::Unavailable)?;
            let vector = Python::attach(|py| awaited.extract::<Vec<f64>>(py))
                .map_err(|_| Error::Unavailable)?;
            Ok(vector.into_iter().map(|value| value as f32).collect())
        }
    }
}
