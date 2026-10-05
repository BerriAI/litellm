mod future;
mod native;
mod python;
mod runtime;
mod selection;

use litellm_cache::Error;
pub(crate) use native::NativeCacheHandle;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
};
pub(crate) use python::{CacheCall, PythonCache};
pub(crate) use runtime::ResolvedCache;
pub(crate) use selection::{Cached, Selection, admit_native, configure, configured_native};

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        Error::UnsupportedOperation => PyNotImplementedError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
