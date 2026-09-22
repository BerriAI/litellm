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
use litellm_http::ClientVariant;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

pub(crate) use self::{
    binding::ResolvedCache, handle::CacheTestHandle, resolver::CacheTestResolver,
};

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        Error::UnsupportedOperation => PyNotImplementedError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// The host's pooled HTTP client, configured from the proxy's HTTP settings.
fn host_client(py: Python<'_>, variant: ClientVariant) -> PyResult<reqwest::Client> {
    let http_config = crate::http::call_config(py, &PyDict::new(py), true)?;
    crate::http::pool()
        .client(&http_config, variant)
        .map_err(crate::http::client_error)
}
