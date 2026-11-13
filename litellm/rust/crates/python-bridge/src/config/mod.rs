mod document;

use std::path::PathBuf;

use litellm_config_load::{HostCall, Includes, Location, Reply, Step};
use litellm_error::Report;
use litellm_host_python::{Execution, ExecutionBody, ExecutionStep, Suspension};
use parking_lot::Mutex;
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyRuntimeError, PyValueError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
};
use serde_yaml_ng::Value;

use document::Opaque;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("config include execution was resumed out of order")]
    OutOfOrder,
    #[error("config document lost a Python value it carried")]
    LostValue,
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        PyRuntimeError::new_err(Report::new(&error).to_string())
    }
}

fn load_failure(error: litellm_config_load::Error<PyErr>) -> PyErr {
    match error {
        litellm_config_load::Error::Host(raised) => raised,
        refused @ (litellm_config_load::Error::IncludeNotPaths
        | litellm_config_load::Error::NotMapping(_)) => {
            PyValueError::new_err(Report::new(&refused).to_string())
        }
        abandoned @ litellm_config_load::Error::Abandoned => {
            PyRuntimeError::new_err(Report::new(&abandoned).to_string())
        }
    }
}

/// Merges the configs `config` includes, reading each one by awaiting `read(path)`.
#[pyfunction]
pub(crate) fn resolve_file_includes(
    py: Python<'_>,
    config: &Bound<'_, PyAny>,
    config_file_path: PathBuf,
    read: Py<PyAny>,
    warn: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    execution(py, Location::File(config_file_path), config, read, warn)
}

/// Merges the bucket objects `config` includes, reading each one by awaiting `read(object_key)`.
#[pyfunction]
pub(crate) fn resolve_object_includes(
    py: Python<'_>,
    config: &Bound<'_, PyAny>,
    object_key: String,
    read: Py<PyAny>,
    warn: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    execution(py, Location::Object(object_key), config, read, warn)
}

fn execution(
    py: Python<'_>,
    root: Location,
    config: &Bound<'_, PyAny>,
    read: Py<PyAny>,
    warn: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let mut opaque = Opaque::default();
    let document = document::from_python(config, &mut opaque)?;
    let driver = IncludesDriver {
        includes: Mutex::new(Includes::new(root, document)),
        read,
        warn,
        opaque,
        reading: None,
    };
    Py::new(py, Execution::new(driver, crate::routes::execution_steps)).map(Py::into_any)
}

struct IncludesDriver {
    includes: Mutex<Includes<PyErr>>,
    read: Py<PyAny>,
    warn: Py<PyAny>,
    opaque: Opaque,
    reading: Option<Reply<Result<Value, PyErr>>>,
}

impl IncludesDriver {
    fn advance(
        &mut self,
        py: Python<'_>,
        answer: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<ExecutionStep> {
        match (self.reading.take(), answer) {
            (Some(reply), Some(answer)) => reply.send(
                answer.and_then(|read| document::from_python(read.bind(py), &mut self.opaque)),
            ),
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => return Err(Error::OutOfOrder.into()),
        }
        loop {
            match self.includes.get_mut().step() {
                Step::Call(HostCall::Exists(path, reply)) => reply.send(path.exists()),
                Step::Call(HostCall::Warn(warning, reply)) => {
                    self.warn.call1(py, (warning.to_string(),))?;
                    reply.send(());
                }
                Step::Call(HostCall::Read(location, reply)) => {
                    let awaitable = self.read.call1(py, (location_text(py, &location)?,))?;
                    self.reading = Some(reply);
                    return Ok(ExecutionStep::Suspended(Suspension::Await(awaitable)));
                }
                Step::Done(Ok(merged)) => {
                    return document::to_python(py, &Value::Mapping(merged), &self.opaque)
                        .map(|merged| ExecutionStep::Return(merged.unbind()));
                }
                Step::Done(Err(error)) => return Err(load_failure(error)),
            }
        }
    }
}

impl ExecutionBody for IncludesDriver {
    fn resume(&mut self, answer: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| self.advance(py, answer))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.read)?;
        visit.call(&self.warn)?;
        self.opaque.traverse(visit)
    }
}

fn location_text<'py>(py: Python<'py>, location: &Location) -> PyResult<Bound<'py, PyAny>> {
    match location {
        Location::File(path) => path.as_os_str().into_bound_py_any(py),
        Location::Object(key) => key.as_str().into_bound_py_any(py),
    }
}
