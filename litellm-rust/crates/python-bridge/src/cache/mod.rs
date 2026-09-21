mod binding;
mod callback;
mod config;
mod facade;
mod future;
mod handle;
mod native;
mod request;
mod resolver;

use litellm_cache::Error;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

pub(crate) use self::{
    binding::ResolvedCache, handle::CacheTestHandle, resolver::CacheTestResolver,
};

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
