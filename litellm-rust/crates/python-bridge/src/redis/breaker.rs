use std::{
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};

use litellm_circuit_breaker::{Admission, Breaker, Config, Failure};
use litellm_redis_client::RedisEvents;
use litellm_redis_types::Endpoint;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::redis::{
    client::{RedisSpec, resolved_endpoint},
    events::{ServiceLoggingEvents, state_label},
    hosts,
};

/// `litellm.redis.store_settings.RedisBreakerSettings`, filled from `litellm.constants`.
#[derive(FromPyObject)]
pub(crate) struct BreakerSettings {
    failure_threshold: u32,
    recovery_timeout: f64,
    timeout_min_duration: f64,
}

impl BreakerSettings {
    pub(crate) fn config(&self) -> PyResult<Config> {
        let seconds = |seconds: f64| {
            Duration::try_from_secs_f64(seconds)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        };
        Ok(Config {
            failure_threshold: self.failure_threshold,
            recovery_timeout: seconds(self.recovery_timeout)?,
            timeout_min_duration: seconds(self.timeout_min_duration)?,
        })
    }
}

/// The one health signal for a Redis, shared by its `RedisCache` and its native client, so a Redis the
/// cache found wedged is fast-failed by every store too.
#[pyclass(frozen, name = "NativeRedisBreaker")]
pub(crate) struct NativeRedisBreaker(pub(crate) Breaker);

impl NativeRedisBreaker {
    fn build(
        py: Python<'_>,
        settings: &BreakerSettings,
        endpoint: Option<Endpoint>,
    ) -> PyResult<Self> {
        let config = settings.config()?;
        let runtime = litellm_host_python::runtime_handle()?;
        let events: Arc<dyn RedisEvents> =
            Arc::new(ServiceLoggingEvents::transitions_only(py, runtime)?);
        Ok(Self(py.detach(|| {
            hosts().breaker(endpoint, config, Some(events))
        })))
    }
}

#[pymethods]
impl NativeRedisBreaker {
    #[new]
    fn new(py: Python<'_>, settings: BreakerSettings) -> PyResult<Self> {
        Self::build(py, &settings, None)
    }

    /// The breaker of the Redis server `redis_kwargs` names, which every cache and client on that server shares
    /// whatever its db, credentials or namespace. The first caller's settings win.
    #[staticmethod]
    fn shared(
        py: Python<'_>,
        redis_kwargs: RedisSpec,
        settings: BreakerSettings,
    ) -> PyResult<Self> {
        let endpoint = resolved_endpoint(&redis_kwargs);
        Self::build(py, &settings, endpoint)
    }

    fn admit(&self) -> Option<NativeAdmission> {
        self.0
            .admit(Instant::now())
            .ok()
            .map(|admission| NativeAdmission(Mutex::new(Some(admission))))
    }

    fn failed(&self, timeout: bool) {
        self.0.failed(failure(timeout), Instant::now());
    }

    #[getter]
    fn state(&self) -> &'static str {
        state_label(&self.0.state())
    }
}

/// One admitted call. Settling it twice records nothing more, and dropping an unsettled probe reopens the breaker.
#[pyclass(frozen, name = "NativeAdmission")]
pub(crate) struct NativeAdmission(Mutex<Option<Admission>>);

impl NativeAdmission {
    fn take(&self) -> Option<Admission> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

#[pymethods]
impl NativeAdmission {
    fn succeeded(&self) {
        if let Some(admission) = self.take() {
            admission.succeeded();
        }
    }

    fn failed(&self, timeout: bool) {
        if let Some(admission) = self.take() {
            admission.failed(failure(timeout), Instant::now());
        }
    }
}

fn failure(timeout: bool) -> Failure {
    match timeout {
        true => Failure::Timeout,
        false => Failure::Hard,
    }
}

#[pyfunction]
pub(crate) fn native_redis_breaker_states(py: Python<'_>) -> Vec<&'static str> {
    py.detach(|| hosts().breakers())
        .iter()
        .map(|breaker| state_label(&breaker.state()))
        .collect()
}
