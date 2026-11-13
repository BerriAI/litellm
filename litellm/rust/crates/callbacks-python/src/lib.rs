use std::sync::Arc;

use litellm_callbacks::{
    CallFacts, Callback, Field, FieldValues, Hook, HookInput, ManifestError, RequestState,
};
use litellm_host_python::{Pythonized, from_py, json_from_py, run_async_value};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

#[pyclass(frozen, name = "NativeCallback")]
pub struct NativeCallback {
    callback: Arc<dyn Callback>,
    manifest: String,
}

#[pyclass(frozen, name = "NativeRequestState")]
pub struct NativeRequestState(RequestState);

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

impl NativeCallback {
    pub fn new(callback: Arc<dyn Callback>) -> Result<Self, RegisterError> {
        let manifest = callback.manifest();
        manifest.validate()?;
        let manifest = serde_json::to_string(manifest)?;
        Ok(Self { callback, manifest })
    }

    pub fn callback(&self) -> &Arc<dyn Callback> {
        &self.callback
    }
}

fn named<T: serde::de::DeserializeOwned>(name: &str) -> PyResult<T> {
    serde_json::from_value(Value::String(name.to_owned()))
        .map_err(|_| PyValueError::new_err(format!("{name} is not a known name")))
}

#[pymethods]
impl NativeCallback {
    fn manifest(&self) -> &str {
        &self.manifest
    }

    fn request_state(&self) -> Option<NativeRequestState> {
        self.callback.request_state().map(NativeRequestState)
    }

    #[pyo3(signature = (hook, fields, call, state, /))]
    fn handle<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: Option<PyRef<'py, NativeRequestState>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hook: Hook = named(hook)?;
        let subscription = self
            .callback
            .manifest()
            .subscription(hook)
            .ok_or_else(|| PyValueError::new_err(format!("{hook:?} has no subscription")))?;
        let mut values = FieldValues::new(subscription.reads);
        for (name, value) in fields {
            let field: Field = named(&name.extract::<String>()?)?;
            values.insert(field, json_from_py(&value)?);
        }
        let input = HookInput {
            hook,
            fields: values,
            call: call.map(from_py::<CallFacts>).transpose()?,
            state: state.map(|state| Arc::clone(&state.0)),
        };
        let callback = Arc::clone(&self.callback);
        run_async_value(py, async move {
            let verdict = callback
                .handle(&input)
                .await
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(Pythonized(verdict))
        })
    }
}
