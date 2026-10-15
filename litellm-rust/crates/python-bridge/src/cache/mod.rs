mod binding;
mod native;
mod python;

pub(crate) use binding::{Cached, CachedHost};
pub(crate) use native::NativeCacheHandle;
pub(crate) use python::scoped as python_cache;

use litellm_cache::Error;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
};

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        Error::UnsupportedOperation => PyNotImplementedError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
