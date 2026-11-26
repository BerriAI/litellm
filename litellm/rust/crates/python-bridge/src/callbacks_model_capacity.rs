use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{CallFacts, Callback, ErasedCallback, FieldValues, HookInput};
use litellm_callbacks_model_capacity::{
    CapacityError, Facts, Frame, HostTransport, JsonHost, KeyFacts, KeyInputs, ModelCapacity,
    SharedStash, StoreLimiter, UsageTokens,
};
use litellm_callbacks_parallel_request_limiter::{Engine, Settings};
use litellm_callbacks_python::RustCallback;
use litellm_host_python::{Pythonized, from_py, json_from_py, to_py};
use litellm_redis_store::rate_limits::RateLimits;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::{
    parallel_request_limiter::{limiter_admission, python_clock, system_now},
    redis::store::{rate_limits::NativeRateLimits, relabel, request::NativeRequestRedis},
};

struct PyHostTransport(Py<PyAny>);

impl HostTransport for PyHostTransport {
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

    fn call_now(&self, method: &'static str, args: Value) -> Result<Value, String> {
        Python::attach(|py| {
            let args = to_py(py, &args)?;
            let result = self.0.bind(py).call_method1(method, (args,))?;
            json_from_py(&result)
        })
        .map_err(|error| error.to_string())
    }
}

#[pyclass(frozen, name = "NativeModelCapacity")]
pub struct NativeModelCapacity {
    inner: Arc<ModelCapacity>,
    rate_limits: Arc<RateLimits>,
}

#[pyclass(frozen, name = "NativeModelCapacityState")]
pub struct NativeModelCapacityState(SharedStash);

#[pymethods]
impl NativeModelCapacityState {
    fn take_output_cap(&self) -> Option<(i64, Option<String>)> {
        self.0.lock().output_cap.take()
    }
}

fn facts(facts: &Bound<'_, PyAny>) -> PyResult<Facts> {
    serde_json::from_value(json_from_py(facts)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

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
    #[pyo3(signature = (delegate, rate_limits, window_size, tpm_reservation_enabled, now = None))]
    fn new(
        delegate: Py<PyAny>,
        rate_limits: PyRef<'_, NativeRateLimits>,
        window_size: i64,
        tpm_reservation_enabled: bool,
        now: Option<Py<PyAny>>,
    ) -> Self {
        let clock: litellm_callbacks_parallel_request_limiter::Now = match now {
            Some(now) => python_clock(now),
            None => Arc::new(system_now),
        };
        let engine = Engine::new(
            Arc::clone(&rate_limits.0),
            Settings {
                window_size,
                tpm_reservation_enabled,
            },
            clock,
        )
        .with_relabel(Arc::new(relabel));
        let limiter = Arc::new(StoreLimiter::new(engine, Some(Arc::new(relabel))));
        let host = Arc::new(JsonHost(PyHostTransport(delegate)));
        Self {
            inner: Arc::new(ModelCapacity::new(limiter, host)),
            rate_limits: Arc::clone(&rate_limits.0),
        }
    }

    fn callback(&self) -> PyResult<RustCallback> {
        let callback: Arc<dyn ErasedCallback> = self.inner.clone();
        Ok(RustCallback::new(callback)?)
    }

    fn request_state(&self) -> NativeModelCapacityState {
        NativeModelCapacityState(SharedStash::default())
    }

    #[pyo3(signature = (hook, fields, call, state, facts, /, request = None))]
    #[allow(clippy::too_many_arguments)]
    fn handle<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: Option<PyRef<'py, NativeModelCapacityState>>,
        facts: &Bound<'py, PyAny>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let facts = self::facts(facts)?;
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
        let redis = request.map(|request| request.0.clone());
        let inner = Arc::clone(&self.inner);
        let in_memory = !self.rate_limits.has_redis();
        limiter_admission(py, in_memory, async move {
            let verdict = inner
                .handle_with(&input, stash.as_ref(), redis.as_ref(), &facts)
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

    #[pyo3(signature = (state, key, model, commit, facts, request = None))]
    #[allow(clippy::too_many_arguments)]
    fn fallback_target_allowed<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeModelCapacityState>,
        key: &Bound<'py, PyAny>,
        model: String,
        commit: bool,
        facts: &Bound<'py, PyAny>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = key_facts(key)?;
        let facts = self::facts(facts)?;
        let inner = Arc::clone(&self.inner);
        let stash = Arc::clone(&state.0);
        let redis = request.map(|request| request.0.clone());
        let in_memory = !self.rate_limits.has_redis();
        limiter_admission(py, in_memory, async move {
            let answer = match inner
                .fallback_target_allowed(&stash, &key, &model, commit, redis.as_ref(), &facts)
                .await
            {
                Ok(allowed) => json!({"allowed": allowed}),
                Err(error) => failure(&error),
            };
            Ok(Pythonized(answer))
        })
    }

    #[pyo3(signature = (connection, key, model, facts, request = None))]
    fn enforce_frame<'py>(
        &self,
        py: Python<'py>,
        connection: Option<PyRef<'py, NativeModelCapacityState>>,
        key: &Bound<'py, PyAny>,
        model: String,
        facts: &Bound<'py, PyAny>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = key_facts(key)?;
        let facts = self::facts(facts)?;
        let inner = Arc::clone(&self.inner);
        let connection = connection.map(|state| Arc::clone(&state.0));
        let redis = request.map(|request| request.0.clone());
        let in_memory = !self.rate_limits.has_redis();
        limiter_admission(py, in_memory, async move {
            let answer = match inner
                .enforce_frame(connection.as_ref(), &key, &model, redis.as_ref(), &facts)
                .await
            {
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

    #[pyo3(signature = (connection, key, model, usage, facts, request = None))]
    #[allow(clippy::too_many_arguments)]
    fn charge_turn<'py>(
        &self,
        py: Python<'py>,
        connection: Option<PyRef<'py, NativeModelCapacityState>>,
        key: &Bound<'py, PyAny>,
        model: String,
        usage: (i64, i64, i64, i64),
        facts: &Bound<'py, PyAny>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = key_facts(key)?;
        let facts = self::facts(facts)?;
        let inner = Arc::clone(&self.inner);
        let connection = connection.map(|state| Arc::clone(&state.0));
        let redis = request.map(|request| request.0.clone());
        let in_memory = !self.rate_limits.has_redis();
        let (prompt_tokens, completion_tokens, total_tokens, cached_tokens) = usage;
        let used = UsageTokens {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens,
        };
        limiter_admission(py, in_memory, async move {
            let answer = match inner
                .charge_turn(
                    connection.as_ref(),
                    &key,
                    &model,
                    &used,
                    redis.as_ref(),
                    &facts,
                )
                .await
            {
                Ok(()) => json!({}),
                Err(error) => failure(&error),
            };
            Ok(Pythonized(answer))
        })
    }
}

fn failure(error: &CapacityError) -> Value {
    match error {
        CapacityError::Raised(raised) => json!({"raised": raised.token}),
        CapacityError::Unverifiable => json!({"unverifiable": true}),
        error if error.is_type_error() => json!({"type_error": error.to_string()}),
        error => json!({"failed": error.to_string()}),
    }
}
