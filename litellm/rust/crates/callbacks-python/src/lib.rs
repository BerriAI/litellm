use std::{any::Any, panic::AssertUnwindSafe, sync::Arc};

use futures_util::FutureExt;
use litellm_callbacks::{
    CallFacts, ErasedCallback, Field, FieldValues, Hook, HookInput, ManifestError, RequestState,
    Verdict,
};
use litellm_host_python::{Execution, from_py, json_from_py, json_to_py, run_async_value};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

mod hosted;
pub mod payload;

#[pyclass(frozen, name = "RustCallback")]
pub struct RustCallback {
    callback: Arc<dyn ErasedCallback>,
    manifest: String,
}

#[pyclass(frozen, name = "RustRequestState")]
pub struct RustRequestState(RequestState);

#[derive(Debug, thiserror::Error)]
pub enum RegisterError {
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("the manifest does not serialise: {0}")]
    Serialise(#[from] serde_json::Error),
}

impl From<RegisterError> for PyErr {
    fn from(error: RegisterError) -> Self {
        PyValueError::new_err(error.to_string())
    }
}

impl RustCallback {
    pub fn new(callback: Arc<dyn ErasedCallback>) -> Result<Self, RegisterError> {
        let manifest = callback.manifest();
        manifest.validate()?;
        let manifest = serde_json::to_string(manifest)?;
        Ok(Self { callback, manifest })
    }

    pub fn callback(&self) -> &Arc<dyn ErasedCallback> {
        &self.callback
    }
}

struct Jsonized(Value);

impl<'py> IntoPyObject<'py> for Jsonized {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        json_to_py(py, &self.0)
    }
}

fn verdict_object(py: Python<'_>, verdict: Option<Verdict>) -> PyResult<Py<PyAny>> {
    let verdict = serde_json::to_value(verdict)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    json_to_py(py, &verdict).map(Bound::unbind)
}

fn execution_steps(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("litellm.rust_bridge.execution")
}

fn panic_error(payload: Box<dyn Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("panic from Rust code");
    PyRuntimeError::new_err(format!("the rust callback panicked: {message}"))
}

fn contained<T>(run: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    std::panic::catch_unwind(AssertUnwindSafe(run))
        .unwrap_or_else(|payload| Err(panic_error(payload)))
}

fn named<T: serde::de::DeserializeOwned>(name: &str) -> PyResult<T> {
    serde_json::from_value(Value::String(name.to_owned()))
        .map_err(|_| PyValueError::new_err(format!("{name} is not a known name")))
}

#[pymethods]
impl RustCallback {
    fn manifest(&self) -> &str {
        &self.manifest
    }

    fn request_state(&self) -> PyResult<Option<RustRequestState>> {
        contained(|| Ok(self.callback.request_state().map(RustRequestState)))
    }

    #[pyo3(signature = (hook, fields, call, state, /))]
    fn handle<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: Option<PyRef<'py, RustRequestState>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = contained(|| self.input(hook, fields, call))?;
        let state = state.map(|state| Arc::clone(&state.0));
        let render_loop = matches!(input.fields.payload(), Ok(Some(_)))
            .then(|| payload::running_loop(py))
            .flatten();
        let callback = Arc::clone(&self.callback);
        run_async_value(py, async move {
            let handled = payload::on_render_loop(
                render_loop,
                AssertUnwindSafe(async move { callback.handle(&input, state.as_ref()).await })
                    .catch_unwind(),
            )
            .await
            .map_err(panic_error)?;
            let verdict = handled.map_err(hosted::callback_error)?;
            serde_json::to_value(verdict)
                .map(Jsonized)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (hook, fields, call, state, host, /))]
    fn start<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: Option<PyRef<'py, RustRequestState>>,
        host: Py<PyAny>,
    ) -> PyResult<Execution> {
        let input = contained(|| self.input(hook, fields, call))?;
        let state = state.map(|state| Arc::clone(&state.0));
        let render_loop = matches!(input.fields.payload(), Ok(Some(_)))
            .then(|| payload::running_loop(py))
            .flatten();
        Ok(Execution::new(
            hosted::HostedExecution::new(
                Arc::clone(&self.callback),
                input,
                state,
                render_loop,
                host,
            ),
            execution_steps,
        ))
    }
}

impl RustCallback {
    fn input<'py>(
        &self,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<HookInput> {
        let hook: Hook = named(hook)?;
        let subscription = self
            .callback
            .manifest()
            .subscription(hook)
            .ok_or_else(|| PyValueError::new_err(format!("{hook:?} has no subscription")))?;
        let mut values = FieldValues::new(subscription.reads);
        for (name, value) in fields {
            let field: Field = named(&name.extract::<String>()?)?;
            if field.by_reference() {
                values.set_payload(payload::stash(value.unbind()));
                continue;
            }
            values.insert(field, json_from_py(&value)?);
        }
        Ok(HookInput {
            hook,
            fields: values,
            call: call.map(from_py::<CallFacts>).transpose()?,
        })
    }
}
