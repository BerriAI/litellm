mod activation;
mod binding;
mod callback;
mod config;
mod embedder;
mod facade;
mod future;
mod handle;
mod identity;
mod native;
mod request;
mod resolver;
mod semantic;

use litellm_cache::Error;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
};

pub(crate) use self::{binding::ResolvedCache, handle::CacheTestHandle, resolver::CacheResolver};

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        Error::UnsupportedOperation => PyNotImplementedError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
