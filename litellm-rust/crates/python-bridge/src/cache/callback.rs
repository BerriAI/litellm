use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};

use super::future::ready_none;

pub(super) struct PythonCallback(Py<PyAny>);

impl PythonCallback {
    pub(super) fn new(object: Py<PyAny>) -> Self {
        Self(object)
    }

    pub(super) fn lookup<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.0
            .bind(py)
            .call_method("get_cache", (), Some(callback_kwargs(kwargs)?))
    }

    pub(super) fn async_lookup<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.0
            .bind(py)
            .call_method("async_get_cache", (), Some(callback_kwargs(kwargs)?))
    }

    pub(super) fn store(
        &self,
        py: Python<'_>,
        response: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.0
            .bind(py)
            .call_method("add_cache", (response,), Some(callback_kwargs(kwargs)?))
            .map(|_| ())
    }

    pub(super) fn async_store<'py>(
        &self,
        py: Python<'py>,
        response: &Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.0.bind(py).call_method(
            "async_add_cache",
            (response,),
            Some(callback_kwargs(kwargs)?),
        )
    }

    pub(super) fn lookup_batch<'py>(
        &self,
        py: Python<'py>,
        requests: &Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = PyList::empty(py);
        for kwargs in batch_callback_kwargs(requests, kwargs)? {
            results.append(
                self.0
                    .bind(py)
                    .call_method("get_cache", (), Some(&kwargs))?,
            )?;
        }
        Ok(results.into_any())
    }

    pub(super) fn async_lookup_batch<'py>(
        &self,
        py: Python<'py>,
        requests: &Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let awaitables = batch_callback_kwargs(requests, kwargs)?
            .iter()
            .map(|kwargs| {
                self.0
                    .bind(py)
                    .call_method("async_get_cache", (), Some(kwargs))
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.import("asyncio")?
            .call_method1("gather", PyTuple::new(py, awaitables)?)
    }

    pub(super) fn async_store_batch<'py>(
        &self,
        py: Python<'py>,
        result: Option<&Bound<'py, PyAny>>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let result = result.ok_or_else(|| {
            PyTypeError::new_err("Python cache callbacks require their original callback_result")
        })?;
        self.0.bind(py).call_method(
            "async_add_cache_pipeline",
            (result,),
            Some(callback_kwargs(kwargs)?),
        )
    }

    pub(super) fn async_flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let object = self.0.bind(py);
        let backend = match object.getattr_opt("cache")? {
            Some(backend) if !backend.is_none() => backend,
            _ => object.clone(),
        };
        if backend.hasattr("async_flush_cache")? {
            return backend.call_method0("async_flush_cache");
        }
        backend.call_method0("flush_cache")?;
        ready_none(py)
    }

    pub(super) fn ping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.0.bind(py).call_method0("ping")
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}

fn callback_kwargs<'a, 'py>(
    kwargs: Option<&'a Bound<'py, PyDict>>,
) -> PyResult<&'a Bound<'py, PyDict>> {
    kwargs.ok_or_else(|| {
        PyTypeError::new_err("Python cache callbacks require their original callback_kwargs")
    })
}

fn batch_callback_kwargs<'py>(
    requests: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyAny>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let kwargs = kwargs
        .ok_or_else(|| {
            PyTypeError::new_err(
                "Python cache callbacks require one original callback_kwargs mapping per request",
            )
        })?
        .try_iter()?
        .map(|item| Ok(item?.cast_into::<PyDict>()?))
        .collect::<PyResult<Vec<_>>>()?;
    if kwargs.len() != requests.len()? {
        return Err(PyValueError::new_err(
            "batch cache requests and callback_kwargs must have equal lengths",
        ));
    }
    Ok(kwargs)
}
