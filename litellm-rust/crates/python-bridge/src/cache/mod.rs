mod python;
mod selection;

pub(crate) use python::{CacheCall, PythonCache};
pub(crate) use selection::{Cached, Selection, configure, ensure_cache_disabled};

use pyo3::{exceptions::PyRuntimeError, prelude::*};

const CACHE_HOST_MODULE: &str = "litellm.rust_bridge.host.cache";

#[derive(Debug, thiserror::Error)]
pub(crate) enum HandleError {
    #[error("this inference route does not support configured caches")]
    Unsupported,
    #[error("cache operation without configured cache")]
    Unconfigured,
    #[error("cache arguments unavailable")]
    MissingArguments,
    #[error("cache reply without pending operation")]
    NothingPending,
}

impl From<HandleError> for PyErr {
    fn from(error: HandleError) -> Self {
        let message = litellm_error::Report::new(&error).to_string();
        match error {
            HandleError::Unsupported
            | HandleError::Unconfigured
            | HandleError::MissingArguments
            | HandleError::NothingPending => PyRuntimeError::new_err(message),
        }
    }
}
