use pyo3::{PyTraverseError, PyVisit, prelude::*};

use super::binding::ResolvedCache;

#[pyclass(frozen, name = "_CacheResolver")]
pub(crate) struct CacheResolver {
    namespace: Py<PyAny>,
}

#[pymethods]
impl CacheResolver {
    #[new]
    fn new(namespace: Py<PyAny>) -> Self {
        Self { namespace }
    }

    pub(crate) fn resolve(&self, py: Python<'_>) -> PyResult<ResolvedCache> {
        let object = self.namespace.bind(py).getattr("cache")?;
        ResolvedCache::from_selected(&object)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.namespace)
    }
}
