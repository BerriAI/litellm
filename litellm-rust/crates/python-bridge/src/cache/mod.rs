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
