use pyo3::{PyTraverseError, PyVisit, prelude::*};

use super::{
    binding::{CacheBinding, ResolvedCache},
    callback::PythonCallback,
    facade,
    handle::CacheTestHandle,
};

#[pyclass(frozen, name = "_CacheTestResolver")]
pub(crate) struct CacheTestResolver {
    namespace: Py<PyAny>,
}

#[pymethods]
impl CacheTestResolver {
    #[new]
    fn new(namespace: Py<PyAny>) -> Self {
        Self { namespace }
    }

    pub(crate) fn resolve(&self, py: Python<'_>) -> PyResult<ResolvedCache> {
        let object = self.namespace.bind(py).getattr("cache")?;
        let binding = if object.is_none() {
            CacheBinding::Disabled
        } else if let Ok(handle) = object.extract::<PyRef<'_, CacheTestHandle>>() {
            CacheBinding::Native(handle.service()?)
        } else if let Some(service) = facade::resolve(py, &object)? {
            CacheBinding::Native(service)
        } else {
            CacheBinding::PythonCallback(PythonCallback::new(object.unbind()))
        };
        Ok(ResolvedCache::new(binding))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.namespace)
    }
}
