use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{CallFacts, Callback, ErasedCallback, FieldValues, HookInput};
use litellm_callbacks_model_capacity::{
    CapacityError, Delegate, Delegated, Frame, KeyFacts, KeyInputs, ModelCapacity, SharedStash,
};
use litellm_callbacks_python::RustCallback;
use litellm_host_python::{Pythonized, from_py, json_from_py, run_async_value, to_py};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

struct PyDelegate(Py<PyAny>);

impl Delegate for PyDelegate {
    fn call(&self, method: &'static str, args: Value) -> BoxFuture<'static, Result<Value, String>> {
        let target = Python::attach(|py| self.0.clone_ref(py));
        Box::pin(async move {
            let pending = Python::attach(|py| {
                let args = to_py(py, &args)?;
                let awaitable = target.bind(py).call_method1(method, (args,))?;
                pyo3_async_runtimes::tokio::into_future(awaitable)
            })
            .map_err(|error| error.to_string())?;
            let result = pending.await.map_err(|error| error.to_string())?;
            Python::attach(|py| json_from_py(result.bind(py))).map_err(|error| error.to_string())
        })
    }
}

#[pyclass(frozen, name = "NativeModelCapacity")]
pub struct NativeModelCapacity {
    inner: Arc<ModelCapacity>,
}

#[pyclass(frozen, name = "NativeModelCapacityState")]
pub struct NativeModelCapacityState(SharedStash);

fn key_facts(key: &Bound<'_, PyAny>) -> PyResult<KeyFacts> {
    let inputs: KeyInputs = serde_json::from_value(json_from_py(key)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(KeyFacts::from_inputs(inputs))
}

fn named<T: DeserializeOwned>(name: &str) -> PyResult<T> {
    serde_json::from_value(Value::String(name.to_owned()))
        .map_err(|_| PyValueError::new_err(format!("{name} is not a known name")))
}

#[pymethods]
impl NativeModelCapacity {
    #[new]
    fn new(delegate: Py<PyAny>) -> Self {
        let delegated = Arc::new(Delegated(PyDelegate(delegate)));
        Self {
            inner: Arc::new(ModelCapacity::new(delegated.clone(), delegated)),
        }
    }

    fn callback(&self) -> PyResult<RustCallback> {
        let callback: Arc<dyn ErasedCallback> = self.inner.clone();
        Ok(RustCallback::new(callback)?)
    }

    fn request_state(&self) -> NativeModelCapacityState {
        NativeModelCapacityState(SharedStash::default())
    }

    #[pyo3(signature = (hook, fields, call, state, /))]
    fn handle<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: Option<PyRef<'py, NativeModelCapacityState>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hook = named(hook)?;
        let subscription = Callback::manifest(&*self.inner)
            .subscription(hook)
            .ok_or_else(|| PyValueError::new_err(format!("{hook:?} has no subscription")))?;
        let mut values = FieldValues::new(subscription.reads);
        for (name, value) in fields {
            values.insert(named(&name.extract::<String>()?)?, json_from_py(&value)?);
        }
        let input = HookInput {
            hook,
            fields: values,
            call: call.map(from_py::<CallFacts>).transpose()?,
        };
        let stash = state.map(|state| Arc::clone(&state.0));
        let inner = Arc::clone(&self.inner);
        run_async_value(py, async move {
            let verdict = Callback::handle(&*inner, &input, stash.as_ref())
                .await
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(Pythonized(verdict))
        })
    }

    fn post_call_headers<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = key_facts(key)?;
        Ok(to_py(py, &json!(self.inner.post_call_headers(&key)))?.into_bound(py))
    }

    #[pyo3(signature = (state, key, model, commit))]
    fn fallback_target_allowed<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeModelCapacityState>,
        key: &Bound<'py, PyAny>,
        model: String,
        commit: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = key_facts(key)?;
        let inner = Arc::clone(&self.inner);
        let stash = Arc::clone(&state.0);
        run_async_value(py, async move {
            let answer = match inner
                .fallback_target_allowed(&stash, &key, &model, commit)
                .await
            {
                Ok(allowed) => json!({"allowed": allowed}),
                Err(error) => failure(&error),
            };
            Ok(Pythonized(answer))
        })
    }

    #[pyo3(signature = (connection, key, model))]
    fn enforce_frame<'py>(
        &self,
        py: Python<'py>,
        connection: Option<PyRef<'py, NativeModelCapacityState>>,
        key: &Bound<'py, PyAny>,
        model: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = key_facts(key)?;
        let inner = Arc::clone(&self.inner);
        let connection = connection.map(|state| Arc::clone(&state.0));
        run_async_value(py, async move {
            let answer = match inner.enforce_frame(connection.as_ref(), &key, &model).await {
                Ok(Frame::Allow) => json!({"allowed": true}),
                Ok(Frame::Refuse(refusal)) => json!({"refused": {
                    "detail": refusal.detail,
                    "headers": refusal.headers,
                    "rate_limit_type": refusal.rate_limit_type,
                    "model": refusal.model,
                }}),
                Ok(Frame::Unverifiable(error)) => {
                    let mut answer = failure(&error);
                    answer["unverifiable"] = json!(true);
                    answer
                }
                Err(error) => failure(&error),
            };
            Ok(Pythonized(answer))
        })
    }
}

fn failure(error: &CapacityError) -> Value {
    match error.raised() {
        Some(raised) => json!({"raised": raised.token}),
        None if error.is_type_error() => json!({"type_error": error.to_string()}),
        None => json!({"failed": error.to_string()}),
    }
}
