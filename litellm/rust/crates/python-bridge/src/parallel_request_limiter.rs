use std::sync::{Arc, Mutex, PoisonError};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    CallFacts, Callback, Field, FieldValues, Hook, HookInput, HookOutput, Host, Manifest, Verdict,
};
use litellm_callbacks_parallel_request_limiter::{
    Backend, CapSettings, Ctx, Descriptor, Engine, EnqueuedReservation, EnqueuedScope,
    EstimateRequest, FailureFacts, IoReservation, LimiterState, Op, ParallelRequestLimiter,
    PolicyFacts, PostCallFailureFacts, Response, Settings, SlotAcquisition, Stash, SuccessFacts,
    Unverifiable, WindowIdentity, input_and_output_tokens, process_monotonic, tokens_for_request,
};
use litellm_callbacks_python::RustCallback;
use litellm_host_python::{Pythonized, from_py, json_from_py, json_loads, run_async_value, to_py};
use litellm_redis_store::{Number, rate_limits::Scope};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString},
};
use serde_json::{Map, Value, json};

use crate::redis::store::{
    rate_limits::NativeRateLimits, relabel, request::NativeRequestRedis, targeted_call,
};

#[pyclass(frozen, name = "NativeParallelRequestLimiter")]
pub(crate) struct NativeParallelRequestLimiter {
    limiter: Arc<ParallelRequestLimiter>,
    manifest: String,
}

#[pyclass(frozen, name = "NativeParallelRequestLimiterState")]
pub(crate) struct NativeParallelRequestLimiterState(Arc<LimiterState>);

fn limiter_async<'py, T, F>(py: Python<'py>, operation: F) -> PyResult<Bound<'py, PyAny>>
where
    T: for<'a> IntoPyObject<'a> + Send + 'static,
    F: Future<Output = PyResult<T>> + Send + 'static,
{
    let target = py
        .import("litellm.redis.stores.rate_limits.types")?
        .getattr("RATE_LIMITS_TARGET")?;
    let call = targeted_call(py, &target, "eval")?;
    run_async_value(py, call.scope(operation))
}

fn named<T: serde::de::DeserializeOwned>(name: &str) -> PyResult<T> {
    serde_json::from_value(Value::String(name.to_owned()))
        .map_err(|_| PyValueError::new_err(format!("{name} is not a known name")))
}

fn decoded<T: serde::de::DeserializeOwned>(py: Python<'_>, text: &str) -> PyResult<T> {
    let parsed = json_loads(py, text.as_bytes())?;
    serde_json::from_value(json_from_py(parsed.bind(py))?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn encoded<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let value =
        serde_json::to_value(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    to_py(py, &value)
}

fn python_clock(now: Py<PyAny>) -> litellm_callbacks_parallel_request_limiter::Now {
    Arc::new(move || {
        Python::attach(|py| {
            now.call0(py)
                .and_then(|value| value.extract::<f64>(py))
                .unwrap_or_else(|_| system_now())
        })
    })
}

fn system_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or_default()
}

fn windows_json(windows: &[WindowIdentity]) -> Vec<Value> {
    windows
        .iter()
        .map(|identity| {
            let kind: &str = identity.counter.kind.into();
            let backend = match identity.backend {
                Backend::Redis => "redis",
                Backend::Local => "local",
            };
            json!([
                identity.counter.scope.descriptor,
                identity.counter.scope.value,
                kind,
                identity.window_start,
                backend,
            ])
        })
        .collect()
}

fn io_json(reservation: &IoReservation) -> Value {
    json!({
        "tokens": reservation.tokens,
        "scopes": reservation.scopes,
        "windows": windows_json(&reservation.windows),
    })
}

fn op_json(op: &Op) -> Value {
    let (increment, guard) = match op {
        Op::Plain(increment) => (increment, None),
        Op::Guarded(guarded) => (
            &guarded.increment,
            Some((
                guarded.expected_window_start.as_str(),
                match guarded.backend {
                    Backend::Redis => "redis",
                    Backend::Local => "local",
                },
            )),
        ),
    };
    let kind: &str = increment.counter.kind.into();
    let amount = match increment.amount {
        Number::Int(value) => json!(value),
        Number::Float(value) => json!(value),
    };
    json!([
        increment.counter.scope.descriptor,
        increment.counter.scope.value,
        kind,
        amount,
        increment.ttl.map(|ttl| ttl.as_secs()),
        guard.map(|(window_start, _)| window_start),
        guard.map(|(_, backend)| backend),
    ])
}

fn reservation_json(reservation: &EnqueuedReservation) -> Value {
    serde_json::to_value(reservation).unwrap_or(Value::Null)
}

fn reply(response: Result<Response, Unverifiable>) -> Value {
    match response {
        Err(Unverifiable) => json!({"unverifiable": true}),
        Ok(response) => {
            let windows: Option<Vec<Value>> = response
                .windows
                .as_ref()
                .map(|windows| windows_json(windows));
            json!({
                "overall_code": response.overall,
                "statuses": response.statuses,
                "windows": windows,
            })
        }
    }
}

struct Hosted {
    limiter: Arc<ParallelRequestLimiter>,
    call: Mutex<Option<crate::redis::events::RedisCall>>,
    state: Arc<LimiterState>,
    run: Mutex<Option<(Value, Option<Value>)>>,
}

impl Callback for Hosted {
    type State = LimiterState;

    fn manifest(&self) -> &Manifest {
        self.limiter.manifest()
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a LimiterState>,
    ) -> BoxFuture<'a, HookOutput> {
        self.handle_hosted(input, state, Host::detached())
    }

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a LimiterState>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        let (inputs, body) = self
            .run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .unwrap_or_default();
        let state = state.unwrap_or(&self.state);
        let call = self
            .call
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        Box::pin(async move {
            let handled = self
                .limiter
                .handle_hosted_with(input, Some(state), inputs, body, host);
            match call {
                Some(call) => call.scope(handled).await,
                None => handled.await,
            }
        })
    }
}

fn exported(stash: &mut Stash) -> Value {
    json!({
        "rate_limit_response": stash.rate_limit_response.as_ref().map(|response| json!({
            "overall_code": response.overall,
            "statuses": response.statuses,
        })),
        "parallel_slot": stash.parallel_slot.as_ref().map(|slot| json!({
            "slot_id": slot.slot_id,
            "gauges": slot.gauges.iter().map(|scope| (&scope.descriptor, &scope.value)).collect::<Vec<_>>(),
        })),
        "reserved_tokens": stash.reserved_tokens,
        "reserved_model": stash.reserved_model.as_ref().map(|model| (&model.requested, &model.group)),
        "reserved_scopes": stash.reserved_scopes,
        "fallback_models": stash.fallback_models,
        "reservation_released": stash.reservation_released,
        "io_reservation_released": stash.io_reservation_released,
        "tpm_limited_tags": stash.tpm_limited_tags,
        "ptu_reserved_tokens": stash.ptu_reserved_tokens,
        "ptu_windows": windows_json(&stash.ptu_windows),
        "itpm": io_json(&stash.itpm),
        "otpm": io_json(&stash.otpm),
        "batch_enqueued_reservation": stash.batch_enqueued.take().map(|reservation| reservation_json(&reservation)),
        "batch_tpd_refunds": std::mem::take(&mut stash.batch_tpd_refunds).iter().map(op_json).collect::<Vec<_>>(),
    })
}

fn estimable(value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    if value.is_none() {
        return Ok(Some(Value::Null));
    }
    if let Ok(flag) = value.cast_exact::<PyBool>() {
        return Ok(Some(Value::Bool(flag.is_true())));
    }
    if let Ok(int) = value.cast_exact::<PyInt>() {
        if let Ok(number) = int.extract::<i64>() {
            return Ok(Some(Value::from(number)));
        }
        if let Ok(number) = int.extract::<u64>() {
            return Ok(Some(Value::from(number)));
        }
        return Ok(Some(Value::String(litellm_host::raw_json::encode(
            &int.str()?.to_cow()?,
        ))));
    }
    if let Ok(float) = value.cast_exact::<PyFloat>() {
        let float = float.value();
        return Ok(Some(match serde_json::Number::from_f64(float) {
            Some(number) => Value::Number(number),
            None if float.is_nan() => Value::String(litellm_host::raw_json::encode("NaN")),
            None if float > 0.0 => Value::String(litellm_host::raw_json::encode("Infinity")),
            None => Value::String(litellm_host::raw_json::encode("-Infinity")),
        }));
    }
    if let Ok(text) = value.cast_exact::<PyString>() {
        return Ok(text
            .to_str()
            .ok()
            .filter(|text| !litellm_host::raw_json::is_marked(text))
            .map(|text| Value::String(text.to_owned())));
    }
    if let Ok(list) = value.cast_exact::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            let Some(item) = estimable(&item)? else {
                return Ok(None);
            };
            items.push(item);
        }
        return Ok(Some(Value::Array(items)));
    }
    if let Ok(dict) = value.cast_exact::<PyDict>() {
        let mut fields = Map::new();
        for (key, item) in dict.iter() {
            let Ok(key) = key.cast_exact::<PyString>() else {
                return Ok(None);
            };
            let Some(key) = key
                .to_str()
                .ok()
                .filter(|key| !litellm_host::raw_json::is_marked(key))
            else {
                return Ok(None);
            };
            let Some(item) = estimable(&item)? else {
                return Ok(None);
            };
            fields.insert(key.to_owned(), item);
        }
        return Ok(Some(Value::Object(fields)));
    }
    Ok(None)
}

#[pymethods]
impl NativeParallelRequestLimiter {
    #[pyo3(signature = (data, call_type, min_configured_tpm_limit, configured_output_tokens, include_media = true))]
    fn estimate_input_and_output_tokens(
        &self,
        data: &Bound<'_, PyAny>,
        call_type: &str,
        min_configured_tpm_limit: Option<i64>,
        configured_output_tokens: Option<i128>,
        include_media: bool,
    ) -> PyResult<Option<(i128, i128)>> {
        let Some(data) = estimable(data)? else {
            return Ok(None);
        };
        Ok(input_and_output_tokens(
            EstimateRequest {
                data: &data,
                call_type,
                min_configured_tpm_limit,
                configured_output_tokens,
                include_media,
            },
            &self.limiter.cap_settings(),
        )
        .ok())
    }

    #[pyo3(signature = (data, call_type, min_configured_tpm_limit, configured_output_tokens))]
    fn estimate_tokens_for_request(
        &self,
        data: &Bound<'_, PyAny>,
        call_type: &str,
        min_configured_tpm_limit: Option<i64>,
        configured_output_tokens: Option<i128>,
    ) -> PyResult<Option<i128>> {
        let Some(data) = estimable(data)? else {
            return Ok(None);
        };
        Ok(tokens_for_request(
            EstimateRequest {
                data: &data,
                call_type,
                min_configured_tpm_limit,
                configured_output_tokens,
                include_media: true,
            },
            &self.limiter.cap_settings(),
        )
        .ok())
    }

    #[new]
    #[pyo3(signature = (rate_limits, window_size, tpm_reservation_enabled, now = None, monotonic = None, output_cap = None))]
    fn new(
        py: Python<'_>,
        rate_limits: PyRef<'_, NativeRateLimits>,
        window_size: i64,
        tpm_reservation_enabled: bool,
        now: Option<Py<PyAny>>,
        monotonic: Option<Py<PyAny>>,
        output_cap: Option<(f64, i64, i64)>,
    ) -> PyResult<Self> {
        let clock = match now {
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
        let limiter = ParallelRequestLimiter::with_monotonic(
            engine,
            monotonic.map_or_else(process_monotonic, python_clock),
        )
        .with_cap_settings(output_cap.map_or_else(
            CapSettings::default,
            |(fraction, default_max_tokens_estimate, responses_min_output_tokens)| CapSettings {
                fraction,
                default_max_tokens_estimate,
                responses_min_output_tokens,
            },
        ));
        let limiter = Arc::new(limiter);
        let manifest = Bound::new(
            py,
            RustCallback::new(Arc::clone(&limiter) as Arc<dyn litellm_callbacks::ErasedCallback>)?,
        )?
        .call_method0("manifest")?
        .extract::<String>()?;
        Ok(Self { limiter, manifest })
    }

    fn manifest(&self) -> &str {
        &self.manifest
    }

    fn request_state(&self) -> NativeParallelRequestLimiterState {
        NativeParallelRequestLimiterState(Arc::new(LimiterState::default()))
    }

    #[pyo3(signature = (hook, fields, call, state, /, request = None, inputs = None))]
    #[allow(clippy::too_many_arguments)]
    fn handle<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: Option<PyRef<'py, NativeParallelRequestLimiterState>>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
        inputs: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inputs: Value = inputs.map(json_from_py).transpose()?.unwrap_or(Value::Null);
        let hook: Hook = named(hook)?;
        let subscription = self
            .limiter
            .manifest()
            .subscription(hook)
            .ok_or_else(|| PyValueError::new_err(format!("{hook:?} has no subscription")))?;
        let mut values = FieldValues::new(subscription.reads);
        for (name, value) in fields {
            let field: Field = named(&name.extract::<String>()?)?;
            values.insert(field, json_from_py(&value)?);
        }
        let state = state.map(|state| Arc::clone(&state.0));
        if let Some(state) = &state {
            state.bind_request(request.map(|request| request.0.clone()));
        }
        let input = HookInput {
            hook,
            fields: values,
            call: call.map(from_py::<CallFacts>).transpose()?,
        };
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let verdict: Option<Verdict> = limiter
                .handle_with(&input, state.as_deref(), inputs)
                .await
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(Pythonized(verdict))
        })
    }

    #[pyo3(signature = (hook, fields, call, state, host, /, request = None, inputs = None, body = None))]
    #[allow(clippy::too_many_arguments)]
    fn start<'py>(
        &self,
        py: Python<'py>,
        hook: &str,
        fields: &Bound<'py, PyDict>,
        call: Option<&Bound<'py, PyAny>>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        host: Py<PyAny>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
        inputs: Option<&Bound<'py, PyAny>>,
        body: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inputs: Value = inputs.map(json_from_py).transpose()?.unwrap_or(Value::Null);
        let body = body.map(estimable).transpose()?.flatten();
        let state = Arc::clone(&state.0);
        state.bind_request(request.map(|request| request.0.clone()));
        let target = py
            .import("litellm.redis.stores.rate_limits.types")?
            .getattr("RATE_LIMITS_TARGET")?;
        let callback = RustCallback::new(Arc::new(Hosted {
            limiter: Arc::clone(&self.limiter),
            call: Mutex::new(Some(targeted_call(py, &target, "eval")?)),
            state,
            run: Mutex::new(Some((inputs, body))),
        }))?;
        Bound::new(py, callback)?.call_method1("start", (hook, fields, call, py.None(), host))
    }

    fn lock_memory(&self, py: Python<'_>) {
        let limiter = Arc::clone(&self.limiter);
        py.detach(|| limiter.engine().memory_updates().acquire());
    }

    fn unlock_memory(&self) {
        self.limiter.engine().memory_updates().release();
    }

    #[pyo3(signature = (descriptors, read_only, skip_tpm_check, parallel_slot_id, skip_parallel_check, policy, request = None))]
    #[allow(clippy::too_many_arguments)]
    fn should_rate_limit<'py>(
        &self,
        py: Python<'py>,
        descriptors: &str,
        read_only: bool,
        skip_tpm_check: bool,
        parallel_slot_id: Option<String>,
        skip_parallel_check: bool,
        policy: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptors: Vec<Descriptor> = decoded(py, descriptors)?;
        let policy: PolicyFacts = decoded(py, policy)?;
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let ctx = Ctx {
                policy: policy.into(),
                request: request.as_ref(),
            };
            let response = limiter
                .engine()
                .should_rate_limit(
                    &descriptors,
                    read_only,
                    skip_tpm_check,
                    parallel_slot_id.as_deref(),
                    skip_parallel_check,
                    ctx,
                )
                .await;
            Ok(Pythonized(reply(response)))
        })
    }

    #[pyo3(signature = (descriptors, increments, policy, request = None))]
    fn atomic_check_and_increment<'py>(
        &self,
        py: Python<'py>,
        descriptors: &str,
        increments: Vec<(i64, i64)>,
        policy: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptors: Vec<Descriptor> = decoded(py, descriptors)?;
        let policy: PolicyFacts = decoded(py, policy)?;
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let ctx = Ctx {
                policy: policy.into(),
                request: request.as_ref(),
            };
            let response = limiter
                .engine()
                .atomic_check_and_increment(&descriptors, &increments, ctx)
                .await;
            Ok(Pythonized(reply(response)))
        })
    }

    fn release_slots<'py>(
        &self,
        py: Python<'py>,
        slot_id: String,
        gauges: Vec<(String, String)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let acquisition = SlotAcquisition {
            slot_id,
            gauges: gauges
                .into_iter()
                .map(|(descriptor, value)| Scope { descriptor, value })
                .collect(),
        };
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.engine().release_slots(&acquisition).await;
            Ok(())
        })
    }

    fn owner_call_id(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> Option<String> {
        state.0.stash().owner_call_id.clone()
    }

    fn identity(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> Option<String> {
        state.0.stash().identity.clone()
    }

    fn hand_over(
        &self,
        py: Python<'_>,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
    ) -> PyResult<Py<PyAny>> {
        let mut stash = state.0.stash();
        stash.owned = false;
        to_py(py, &exported(&mut stash))
    }

    fn view(
        &self,
        py: Python<'_>,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
    ) -> PyResult<Py<PyAny>> {
        to_py(py, &exported(&mut state.0.snapshot()))
    }

    fn enqueued_reserve<'py>(
        &self,
        py: Python<'py>,
        tokens: i64,
        scopes: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scopes: Vec<EnqueuedScope> = decoded(py, scopes)?;
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let outcome = limiter.enqueued().reserve(tokens, scopes).await;
            serde_json::to_value(&outcome)
                .map(Pythonized)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
    }

    fn enqueued_refund<'py>(
        &self,
        py: Python<'py>,
        reservation: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reservation: EnqueuedReservation = decoded(py, reservation)?;
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.enqueued().refund(&reservation).await;
            Ok(())
        })
    }

    fn enqueued_save<'py>(
        &self,
        py: Python<'py>,
        batch_id: String,
        reservation: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reservation: EnqueuedReservation = decoded(py, reservation)?;
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.enqueued().save(&batch_id, &reservation).await;
            Ok(())
        })
    }

    fn enqueued_pop<'py>(&self, py: Python<'py>, batch_id: String) -> PyResult<Bound<'py, PyAny>> {
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            Ok(limiter
                .enqueued()
                .pop(&batch_id)
                .await
                .map(|reservation| Pythonized(reservation_json(&reservation))))
        })
    }

    fn owned(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> bool {
        state.0.stash().owned
    }

    fn rate_limit_response(
        &self,
        py: Python<'_>,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let stash = state.0.stash();
        stash
            .rate_limit_response
            .as_ref()
            .map(|response| {
                to_py(
                    py,
                    &json!({
                        "overall_code": response.overall,
                        "statuses": response.statuses,
                    }),
                )
            })
            .transpose()
    }

    fn reserved_model(
        &self,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
    ) -> Option<(String, String)> {
        state
            .0
            .stash()
            .reserved_model
            .as_ref()
            .map(|model| (model.requested.clone(), model.group.clone()))
    }

    fn reserved_tokens(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> i64 {
        state.0.stash().reserved_tokens
    }

    fn holds_parallel_slot(&self, state: PyRef<'_, NativeParallelRequestLimiterState>) -> bool {
        state.0.stash().parallel_slot.is_some()
    }

    #[pyo3(signature = (state, min_configured_limit, configured_output_tokens = None, cap_applied = false))]
    fn implicit_output_cap(
        &self,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
        min_configured_limit: Option<i64>,
        configured_output_tokens: Option<i64>,
        cap_applied: bool,
    ) -> Option<i64> {
        self.limiter.implicit_output_cap(
            &state.0,
            min_configured_limit,
            configured_output_tokens,
            cap_applied,
        )
    }

    fn descriptors(
        &self,
        py: Python<'_>,
        state: PyRef<'_, NativeParallelRequestLimiterState>,
    ) -> PyResult<Py<PyAny>> {
        encoded(py, &self.limiter.descriptors(&state.0))
    }

    fn take_parallel_slot<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&state.0);
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            Ok(limiter.take_parallel_slot(&state).await.map(|slot| {
                (
                    slot.slot_id,
                    slot.gauges
                        .into_iter()
                        .map(|scope| (scope.descriptor, scope.value))
                        .collect::<Vec<_>>(),
                )
            }))
        })
    }

    #[pyo3(signature = (state, facts, request = None))]
    fn settle_success<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        facts: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let facts: SuccessFacts = decoded(py, facts)?;
        let state = Arc::clone(&state.0);
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter
                .settle_success(&state, facts, request.as_ref())
                .await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, facts, request = None))]
    fn settle_turn<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        facts: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let facts: SuccessFacts = decoded(py, facts)?;
        let state = Arc::clone(&state.0);
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.settle_turn(&state, facts, request.as_ref()).await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, facts, request = None))]
    fn settle_failure<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        facts: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let facts: FailureFacts = decoded(py, facts)?;
        let state = Arc::clone(&state.0);
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter
                .settle_failure(&state, facts, request.as_ref())
                .await;
            Ok(())
        })
    }

    fn post_call_failure<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        facts: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let facts: PostCallFailureFacts = decoded(py, facts)?;
        let state = Arc::clone(&state.0);
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.post_call_failure(&state, facts).await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, call_id, request = None))]
    fn release_for_settlement<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
        call_id: Option<String>,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&state.0);
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter
                .release_for_settlement(&state, call_id.as_deref(), request.as_ref())
                .await;
            Ok(())
        })
    }

    fn release_on_disconnect<'py>(
        &self,
        py: Python<'py>,
        state: PyRef<'py, NativeParallelRequestLimiterState>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&state.0);
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            limiter.release_on_disconnect(&state).await;
            Ok(())
        })
    }

    #[pyo3(signature = (state, descriptors, model, policy, request = None))]
    fn fallback_target_allowed<'py>(
        &self,
        py: Python<'py>,
        state: Option<PyRef<'py, NativeParallelRequestLimiterState>>,
        descriptors: &str,
        model: String,
        policy: &str,
        request: Option<PyRef<'py, NativeRequestRedis>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let descriptors: Vec<Descriptor> = decoded(py, descriptors)?;
        let policy: PolicyFacts = decoded(py, policy)?;
        let state = state.map(|state| Arc::clone(&state.0));
        let request = request.map(|request| request.0.clone());
        let limiter = Arc::clone(&self.limiter);
        limiter_async(py, async move {
            let verdict = limiter
                .fallback_target_allowed(
                    state.as_deref(),
                    descriptors,
                    &model,
                    policy,
                    request.as_ref(),
                )
                .await;
            Ok(Pythonized(match verdict {
                Ok(allowed) => json!({"allowed": allowed}),
                Err(abort) => json!({"allowed": false, "abort": Verdict::Abort(*abort)}),
            }))
        })
    }
}
