use pyo3::{
    exceptions::PyBaseException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::python::{self, Wrapper};

/// The `Logging` instance one call fans out through.
pub struct PythonLogger {
    object: Py<PyAny>,
}

impl PythonLogger {
    pub(crate) fn new(object: Py<PyAny>) -> Self {
        Self { object }
    }

    pub(crate) fn object<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.object.bind(py)
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            object: self.object.clone_ref(py),
        }
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.object)
    }

    pub fn restore_context(&self, py: Python<'_>) -> PyResult<()> {
        Wrapper::RestoreContext.call(py, (self.object(py),))?;
        Ok(())
    }
}

impl FromPyObject<'_, '_> for PythonLogger {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(object.to_owned().unbind()))
    }
}

pub struct SetupResult<'py>(Bound<'py, PyAny>);

impl SetupResult<'_> {
    pub fn logger(&self) -> PyResult<PythonLogger> {
        Ok(PythonLogger::new(self.0.getattr("logger")?.unbind()))
    }

    pub fn kwargs(&self) -> PyResult<Py<PyDict>> {
        Ok(self.0.getattr("kwargs")?.extract()?)
    }
}

pub fn setup<'py>(
    py: Python<'py>,
    call_type: &str,
    args: &Py<PyTuple>,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    asynchronous: bool,
) -> PyResult<SetupResult<'py>> {
    Wrapper::Setup
        .call(py, (call_type, args, kwargs, start, asynchronous))
        .map(SetupResult)
}

pub fn finalize(
    py: Python<'_>,
    response: &Option<Py<PyAny>>,
    logger: &PythonLogger,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    end: &Option<Py<PyAny>>,
) -> PyResult<()> {
    Wrapper::Finalize.call(py, (response, logger.object(py), kwargs, start, end))?;
    Ok(())
}

/// The awaitables of the `litellm.utils` deployment hook fan-outs.
pub struct DeploymentHooks;

impl DeploymentHooks {
    pub fn pre_call(
        py: Python<'_>,
        logger: &PythonLogger,
        kwargs: &Py<PyDict>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        python::DeploymentHooks::PreCall
            .call(py, (logger.object(py), kwargs, call_type))
            .map(Bound::unbind)
    }

    pub fn post_call_success(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        response: &Option<Py<PyAny>>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        python::DeploymentHooks::PostCallSuccess
            .call(py, (kwargs, response, call_type))
            .map(Bound::unbind)
    }

    pub fn post_call_failure(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        error: &Py<PyBaseException>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        python::DeploymentHooks::PostCallFailure
            .call(py, (kwargs, error, call_type))
            .map(Bound::unbind)
    }
}

/// The awaitable of the Messages handler's `async_pre_request_hook` fan-out.
pub struct MessagesHandler;

impl MessagesHandler {
    pub fn execute_pre_request_hooks(
        py: Python<'_>,
        model: &str,
        messages: &Bound<'_, PyAny>,
        kwargs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        python::MessagesHandler::ExecutePreRequestHooks
            .call(py, (model, messages, kwargs))
            .map(Bound::unbind)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::PyTypeError;

    use super::*;

    #[test]
    fn setup_fields_are_checked_lazily() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
reads = []
class Logger:
    def __getattribute__(self, name):
        reads.append(name)
        raise AssertionError('logger methods must remain lazy')
logger = Logger()
class Setup:
    @property
    def logger(self):
        reads.append('logger')
        return logger
    @property
    def kwargs(self):
        reads.append('kwargs')
        return []
result = Setup()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let result = SetupResult(locals.get_item("result").unwrap().unwrap());
            let logger = result.logger().unwrap();
            assert!(
                logger
                    .object(py)
                    .is(locals.get_item("logger").unwrap().unwrap())
            );
            assert!(
                result
                    .kwargs()
                    .unwrap_err()
                    .is_instance_of::<PyTypeError>(py)
            );
            assert_eq!(
                locals
                    .get_item("reads")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["logger", "kwargs"]
            );
        });
    }
}
