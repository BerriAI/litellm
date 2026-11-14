use std::{sync::Arc, time::Duration};

use litellm_redis_store::{
    Clock,
    router_session_pins::{Eligible, PinKey, RouterSessionPins},
    system_clock,
};
use pyo3::prelude::*;

use super::{
    NativePending, duration, kind_error, pending, policy::router_tier_policies, redis_async,
    remote, request::NativeSend, unwrap_values, value::PyValue,
};
use crate::redis::client::NativeRedisClient;

fn pin(value: Option<litellm_python_compat::Value>) -> Option<PyValue> {
    value.map(PyValue)
}

/// The `PinKey` dataclasses in `litellm.redis.stores.router_session_pins.keys`. A key whose fields
/// are a subset of another's comes after it, so extraction never stops at the smaller one.
#[derive(FromPyObject)]
enum PyPinKey {
    TierModel {
        router: String,
        tiers: String,
        caller: Option<String>,
        session_id: String,
        tier: String,
        replay_fallback: Option<String>,
    },
    TierSession {
        router: String,
        caller: Option<String>,
        session_id: String,
    },
    DeploymentSession {
        model_group: String,
        caller: Option<String>,
        session_id: String,
    },
    Deployment {
        model_group: String,
        caller: String,
    },
}

impl From<PyPinKey> for PinKey {
    fn from(key: PyPinKey) -> Self {
        match key {
            PyPinKey::TierModel {
                router,
                tiers,
                caller,
                session_id,
                tier,
                replay_fallback,
            } => Self::TierModel {
                router,
                tiers,
                caller,
                session_id,
                tier,
                replay_fallback,
            },
            PyPinKey::TierSession {
                router,
                caller,
                session_id,
            } => Self::TierSession {
                router,
                caller,
                session_id,
            },
            PyPinKey::DeploymentSession {
                model_group,
                caller,
                session_id,
            } => Self::DeploymentSession {
                model_group,
                caller,
                session_id,
            },
            PyPinKey::Deployment {
                model_group,
                caller,
            } => Self::Deployment {
                model_group,
                caller,
            },
        }
    }
}

/// `InMemoryCache(clock=...)`: a callable returning epoch seconds, which tests use to move time.
fn python_clock(clock: Py<PyAny>) -> Clock {
    Arc::new(move || {
        Python::attach(|py| clock.call0(py)?.extract::<f64>(py))
            .ok()
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .unwrap_or_default()
    })
}

/// The router's session affinity pins, behind `RouterSessionPinStore`.
#[pyclass(frozen, name = "NativeRouterSessionPins")]
pub(crate) struct NativeRouterSessionPins(Arc<RouterSessionPins>);

#[pymethods]
impl NativeRouterSessionPins {
    #[new]
    #[pyo3(signature = (client, clock=None))]
    fn new(
        py: Python<'_>,
        client: Option<PyRef<'_, NativeRedisClient>>,
        clock: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(Arc::new(RouterSessionPins::with_clock(
            client.map(|client| remote(&client)),
            router_tier_policies(py)?.default_in_memory_ttl,
            clock.map_or_else(system_clock, python_clock),
        ))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    fn attach(&self, client: PyRef<'_, NativeRedisClient>) -> bool {
        self.0.attach(remote(&client))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn get<'py>(&self, py: Python<'py>, key: PyPinKey) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let key = PinKey::from(key);
        redis_async(py, "async_get_cache", kind_error, async move {
            store.get(&key).await.map(pin)
        })
    }

    fn get_local(&self, key: PyPinKey) -> PyResult<Option<PyValue>> {
        self.0.get_local(&key.into()).map(pin).map_err(kind_error)
    }

    fn set_local(&self, key: PyPinKey, value: PyValue, ttl_seconds: f64) -> PyResult<()> {
        self.0
            .set_local(&key.into(), value.0, duration(ttl_seconds)?)
            .map_err(kind_error)
    }

    fn delete_local(&self, key: PyPinKey) -> PyResult<()> {
        self.0.delete_local(&key.into()).map_err(kind_error)
    }

    fn set<'py>(
        &self,
        py: Python<'py>,
        key: PyPinKey,
        value: PyValue,
        ttl_seconds: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let key = PinKey::from(key);
        let ttl = duration(ttl_seconds)?;
        redis_async(py, "async_set_cache", kind_error, async move {
            store.set(&key, value.0, ttl).await
        })
    }

    fn get_shared<'py>(&self, py: Python<'py>, key: PyPinKey) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let key = PinKey::from(key);
        redis_async(py, "async_get_cache", kind_error, async move {
            store.get_shared(&key).await.map(pin)
        })
    }

    /// The read script, declared on the batch the caller's `NativeSend` picked.
    fn read(&self, key: PyPinKey, ttl_seconds: f64, send: &NativeSend) -> PyResult<NativePending> {
        let declared = self
            .0
            .read(send.send(), &key.into(), duration(ttl_seconds)?)
            .map_err(kind_error)?;
        Ok(pending("async_register_script", kind_error, declared, pin))
    }

    #[pyo3(signature = (key, value, ttl_seconds, eligible_values, preserve_legacy_model_ids))]
    fn claim<'py>(
        &self,
        py: Python<'py>,
        key: PyPinKey,
        value: PyValue,
        ttl_seconds: f64,
        eligible_values: Option<Vec<PyValue>>,
        preserve_legacy_model_ids: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let key = PinKey::from(key);
        let ttl = duration(ttl_seconds)?;
        let eligible = eligible(eligible_values, preserve_legacy_model_ids);
        redis_async(py, "async_register_script", kind_error, async move {
            store.claim(&key, &value.0, ttl, &eligible).await.map(pin)
        })
    }

    #[pyo3(signature = (key, value, ttl_seconds, eligible_values, preserve_legacy_model_ids))]
    fn claim_local(
        &self,
        key: PyPinKey,
        value: PyValue,
        ttl_seconds: f64,
        eligible_values: Option<Vec<PyValue>>,
        preserve_legacy_model_ids: bool,
    ) -> PyResult<PyValue> {
        self.0
            .claim_local(
                &key.into(),
                &value.0,
                duration(ttl_seconds)?,
                &eligible(eligible_values, preserve_legacy_model_ids),
            )
            .map(PyValue)
            .map_err(kind_error)
    }
}

fn eligible(values: Option<Vec<PyValue>>, preserve_legacy_model_ids: bool) -> Eligible {
    Eligible {
        values: values.map(unwrap_values),
        preserve_legacy_model_ids,
    }
}
