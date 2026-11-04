mod native;
mod python;
mod selection;

pub(crate) use native::NativeCacheHandle;
pub(crate) use python::{CacheCall, PythonCache};
pub(crate) use selection::{Cached, Selection, configure, configured_native};

use litellm_cache::Error;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
};

/// A cache call the bridge rejects itself, before or after the cache ran.
#[derive(Debug, thiserror::Error)]
pub(crate) enum HandleError {
    #[error("recreate the v2 cache after fork")]
    Forked,
    #[error("cache limits must be positive")]
    Limits,
    #[error("namespace and a positive cache limit are required")]
    RedisLimits,
    #[error("cache durations must be finite and positive")]
    Duration(#[source] Option<std::time::TryFromFloatSecsError>),
    #[error("unsupported v2 cache control: {0}")]
    Control(String),
    #[error("native inference needs a v2 cache, the catalog keeps Python caches on Python")]
    NotNative,
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
            HandleError::Limits
            | HandleError::RedisLimits
            | HandleError::Duration(_)
            | HandleError::Control(_) => PyValueError::new_err(message),
            HandleError::Forked
            | HandleError::NotNative
            | HandleError::Unconfigured
            | HandleError::MissingArguments
            | HandleError::NothingPending => PyRuntimeError::new_err(message),
        }
    }
}

fn cache_error(error: Error) -> PyErr {
    let message = litellm_error::Report::new(&error).to_string();
    match error {
        Error::InvalidEntry(_) => PyValueError::new_err(message),
        Error::UnsupportedOperation => PyNotImplementedError::new_err(message),
        Error::Unavailable(_) | Error::UnscopedFlush | Error::MissingPrompt => {
            PyRuntimeError::new_err(message)
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{
        exceptions::{PyRuntimeError, PyValueError},
        prelude::*,
    };
    use rstest::rstest;

    use super::{Error, cache_error};

    fn refused() -> std::io::Error {
        std::io::Error::other("connection refused")
    }

    #[rstest]
    #[case::unavailable(
        Error::unavailable(refused()),
        false,
        "cache is unavailable: connection refused"
    )]
    #[case::invalid_entry(
        Error::invalid_entry(refused()),
        true,
        "invalid cache entry: connection refused"
    )]
    fn python_sees_the_class_and_the_whole_chain(
        #[case] error: Error,
        #[case] caller_fault: bool,
        #[case] message: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let failure = cache_error(error);
            assert_eq!(failure.is_instance_of::<PyValueError>(py), caller_fault);
            assert_eq!(failure.is_instance_of::<PyRuntimeError>(py), !caller_fault);
            assert_eq!(failure.value(py).to_string(), message);
        });
    }
}
