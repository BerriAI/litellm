use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// The `Logging` instance one call fans out through, and who owns it. A logger the caller
/// handed in is observed in full, because the caller reads it after the call; one this
/// crate built through `function_setup` is elided wherever no registry needs it.
pub struct PythonLogger {
    object: Py<PyAny>,
    bridge_owned: bool,
}

impl PythonLogger {
    pub(crate) fn new(object: Py<PyAny>, bridge_owned: bool) -> Self {
        Self {
            object,
            bridge_owned,
        }
    }

    pub(crate) fn object<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.object.bind(py)
    }

    pub(crate) fn bridge_owned(&self) -> bool {
        self.bridge_owned
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            object: self.object.clone_ref(py),
            bridge_owned: self.bridge_owned,
        }
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.object)
    }

    pub fn success_bookkeeping(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> PyResult<()> {
        py.import("litellm.rust_bridge.legacy_callbacks")?
            .getattr("success_bookkeeping")?
            .call1((self.object(py), response, start, end, asynchronous))?;
        Ok(())
    }

    pub fn restore_context(&self, py: Python<'_>) -> PyResult<()> {
        py.import("litellm.utils")?
            .getattr("_restore_correlation_context_if_supported")?
            .call1((self.object(py),))?;
        Ok(())
    }
}

/// A bare Python object was not obtained from `setup`, so it is caller-owned.
impl FromPyObject<'_, '_> for PythonLogger {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(object.to_owned().unbind(), false))
    }
}

pub struct SetupResult<'py>(Bound<'py, PyAny>);

impl SetupResult<'_> {
    pub fn logger(&self) -> PyResult<PythonLogger> {
        let object = self.0.getattr("logger")?.unbind();
        let bridge_owned = self.0.getattr("bridge_owned")?.extract()?;
        Ok(PythonLogger::new(object, bridge_owned))
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
    py.import("litellm.rust_bridge.legacy_callbacks")?
        .getattr("setup")?
        .call1((call_type, args, kwargs, start, asynchronous))
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
    py.import("litellm.rust_bridge.legacy_callbacks")?
        .getattr("finalize")?
        .call1((response, logger.object(py), kwargs, start, end))?;
    Ok(())
}

pub struct DeploymentHooks;

impl DeploymentHooks {
    pub fn needed(py: Python<'_>) -> PyResult<bool> {
        py.import("litellm.rust_bridge.legacy_callbacks")?
            .getattr("deployment_callbacks_needed")?
            .call0()?
            .extract()
    }

    pub fn before_call(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_pre_call_deployment_hook")?
            .call1((kwargs, call_type))
            .map(Bound::unbind)
    }

    pub fn after_success(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        response: &Option<Py<PyAny>>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_post_call_success_deployment_hook")?
            .call1((kwargs, response, call_type))
            .map(Bound::unbind)
    }

    pub fn after_failure(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        error: &Py<PyBaseException>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_post_call_failure_deployment_hook")?
            .call1((kwargs, error, call_type))
            .map(Bound::unbind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyTypeError;

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
    def bridge_owned(self):
        reads.append('bridge_owned')
        return True
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
            assert!(logger.bridge_owned());
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
                ["logger", "bridge_owned", "kwargs"]
            );
        });
    }

    #[test]
    fn a_logger_extracted_from_a_bare_object_is_caller_owned() {
        Python::initialize();
        Python::attach(|py| {
            let logger: PythonLogger = py.None().into_bound(py).extract().unwrap();
            assert!(!logger.bridge_owned());
        });
    }
}
