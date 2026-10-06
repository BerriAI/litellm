mod native;
mod python;
mod selection;

use litellm_cache::Error;
pub(crate) use native::NativeCacheHandle;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
};
pub(crate) use python::{CacheCall, PythonCache};
pub(crate) use selection::{PythonCacheSelection, PythonCached, select_python_cache};

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        Error::UnsupportedOperation => PyNotImplementedError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
