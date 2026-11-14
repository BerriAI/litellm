use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use litellm_redis_store::router_usage::{
    Counter, Increment, Kind, RouterUsage, Scope, UsageReservation,
};
use pyo3::prelude::*;

use super::{
    NativePending, PyNumber, Spelled, kind_error, pending,
    policy::router_tier_policies,
    redis_async, redis_sync, remote,
    request::{NativeRequestRedis, NativeSend},
    ttl,
};
use crate::redis::client::NativeRedisClient;

/// `litellm.redis.stores.router_usage.UsageCounter`.
#[derive(FromPyObject)]
struct PyCounter {
    scope: Spelled<Scope>,
    deployment_id: String,
    model: Option<String>,
    window: String,
    kind: Spelled<Kind>,
}

impl From<PyCounter> for Counter {
    fn from(counter: PyCounter) -> Self {
        Self {
            scope: counter.scope.0,
            deployment_id: counter.deployment_id,
            model: counter.model,
            window: counter.window,
            kind: counter.kind.0,
        }
    }
}

fn counters(counters: Vec<PyCounter>) -> Vec<Counter> {
    counters.into_iter().map(Counter::from).collect()
}

/// `litellm.redis.stores.router_usage.CounterIncrement`.
#[derive(FromPyObject)]
struct PyIncrement {
    counter: PyCounter,
    amount: PyNumber,
    ttl: Option<f64>,
}

impl TryFrom<PyIncrement> for Increment {
    type Error = PyErr;

    fn try_from(increment: PyIncrement) -> PyResult<Self> {
        Ok(Self {
            counter: increment.counter.into(),
            amount: increment.amount.0,
            ttl: ttl(increment.ttl)?,
        })
    }
}

fn increments(increments: Vec<PyIncrement>) -> PyResult<Vec<Increment>> {
    increments.into_iter().map(Increment::try_from).collect()
}

fn numbers(read: Vec<Option<litellm_redis_store::Number>>) -> Vec<Option<PyNumber>> {
    read.into_iter().map(|count| count.map(PyNumber)).collect()
}

fn totals(totals: Option<Vec<litellm_redis_store::Number>>) -> Option<Vec<PyNumber>> {
    totals.map(|totals| totals.into_iter().map(PyNumber).collect())
}

/// One router's global TPM/RPM counters. The `_sync` methods block with the GIL released, for the
/// Rust router's sync calls, which have no event loop to await on.
#[pyclass(frozen, name = "NativeRouterUsage")]
pub(crate) struct NativeRouterUsage(Arc<RouterUsage>);

/// A usage read declared on a batch, with the refreshes it reserved, given back at most once.
#[pyclass(frozen, name = "NativeUsageRead")]
pub(crate) struct NativeUsageRead {
    #[pyo3(get)]
    read: Py<NativePending>,
    store: Arc<RouterUsage>,
    reservation: Mutex<Option<UsageReservation>>,
}

#[pymethods]
impl NativeUsageRead {
    fn release(&self) {
        let reservation = self
            .reservation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(reservation) = reservation {
            self.store.release(reservation);
        }
    }
}

#[pymethods]
impl NativeRouterUsage {
    #[new]
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>) -> PyResult<Self> {
        let remote = client.map(|client| remote(&client));
        let policy = router_tier_policies(py)?.usage;
        Ok(Self(Arc::new(RouterUsage::new(remote, policy))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    fn attach(&self, client: PyRef<'_, NativeRedisClient>) -> bool {
        self.0.attach(remote(&client))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        counter: PyCounter,
        value: PyNumber,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counter = Counter::from(counter);
        redis_async(py, "async_increment", kind_error, async move {
            Ok(store
                .increment(&counter, value.0, Duration::from_secs(ttl_seconds))
                .await
                .map(PyNumber))
        })
    }

    /// `_increment_deployment`'s `async_increment_pipeline`, always on its own round trip.
    fn increment_deployment<'py>(
        &self,
        py: Python<'py>,
        deployment_id: String,
        model: Option<String>,
        minute: String,
        tokens: PyNumber,
        requests: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment_pipeline", kind_error, async move {
            store
                .increment_deployment(
                    &deployment_id,
                    model.as_deref(),
                    &minute,
                    tokens.0,
                    requests,
                )
                .await
        })
    }

    fn get_many<'py>(
        &self,
        py: Python<'py>,
        counters: Vec<PyCounter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counters = self::counters(counters);
        redis_async(py, "async_batch_get_cache", kind_error, async move {
            let read = store.get_many(&counters).await?;
            Ok(read
                .into_iter()
                .map(|count| count.map(PyNumber))
                .collect::<Vec<_>>())
        })
    }

    fn increment_sync(
        &self,
        py: Python<'_>,
        counter: PyCounter,
        value: PyNumber,
        ttl_seconds: u64,
    ) -> PyResult<Option<PyNumber>> {
        let store = self.0.clone();
        let counter = Counter::from(counter);
        redis_sync(py, "increment_cache", kind_error, async move {
            Ok(store
                .increment(&counter, value.0, Duration::from_secs(ttl_seconds))
                .await
                .map(PyNumber))
        })
    }

    fn increment_deployment_sync(
        &self,
        py: Python<'_>,
        deployment_id: String,
        model: Option<String>,
        minute: String,
        tokens: PyNumber,
        requests: i64,
    ) -> PyResult<()> {
        let store = self.0.clone();
        redis_sync(py, "increment_pipeline", kind_error, async move {
            store
                .increment_deployment(
                    &deployment_id,
                    model.as_deref(),
                    &minute,
                    tokens.0,
                    requests,
                )
                .await
        })
    }

    fn get_many_sync(
        &self,
        py: Python<'_>,
        counters: Vec<PyCounter>,
    ) -> PyResult<Vec<Option<PyNumber>>> {
        let store = self.0.clone();
        let counters = self::counters(counters);
        redis_sync(py, "batch_get_cache", kind_error, async move {
            let read = store.get_many(&counters).await?;
            Ok(read.into_iter().map(|count| count.map(PyNumber)).collect())
        })
    }

    /// `get_many` declared on the batch `send` picks, with the refreshes it reserved.
    fn usage(
        &self,
        py: Python<'_>,
        counters: Vec<PyCounter>,
        send: &NativeSend,
    ) -> PyResult<NativeUsageRead> {
        let declared = self
            .0
            .declare_get_many(send.send(), &self::counters(counters))
            .map_err(kind_error)?;
        Ok(NativeUsageRead {
            read: Py::new(
                py,
                pending("async_batch_get_cache", kind_error, declared.read, numbers),
            )?,
            store: Arc::clone(&self.0),
            reservation: Mutex::new(Some(declared.reservation)),
        })
    }

    fn increment_pipeline<'py>(
        &self,
        py: Python<'py>,
        increments: Vec<PyIncrement>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments = self::increments(increments)?;
        redis_async(py, "async_increment_pipeline", kind_error, async move {
            Ok(totals(store.increment_pipeline(&increments).await))
        })
    }

    /// Counts in memory now and declares the increments on the request's post-call batch.
    fn increment_post_call(
        &self,
        request: &NativeRequestRedis,
        increments: Vec<PyIncrement>,
    ) -> PyResult<()> {
        self.0
            .increment_post_call(&request.0, &self::increments(increments)?);
        Ok(())
    }

    fn increment_remote<'py>(
        &self,
        py: Python<'py>,
        increments: Vec<PyIncrement>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments = self::increments(increments)?;
        redis_async(py, "async_increment_pipeline", kind_error, async move {
            store.increment_remote(&increments).await.map(totals)
        })
    }

    fn increment_local(&self, increment: PyIncrement) -> PyResult<PyNumber> {
        self.0
            .increment_local(&increment.try_into()?)
            .map(PyNumber)
            .map_err(kind_error)
    }

    fn get_local(&self, counter: PyCounter) -> PyResult<Option<PyNumber>> {
        self.0
            .get_local(&counter.into())
            .map(|count| count.map(PyNumber))
            .map_err(kind_error)
    }

    fn get_local_many(&self, counters: Vec<PyCounter>) -> PyResult<Vec<Option<PyNumber>>> {
        self.0
            .get_local_many(&self::counters(counters))
            .map(numbers)
            .map_err(kind_error)
    }

    fn set_local(&self, counter: PyCounter, value: PyNumber) -> PyResult<()> {
        self.0
            .set_local(&counter.into(), value.0)
            .map_err(kind_error)
    }

    fn current<'py>(
        &self,
        py: Python<'py>,
        counter: PyCounter,
        limit: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counter = Counter::from(counter);
        redis_async(py, "async_get_cache", kind_error, async move {
            Ok(store.current(&counter, limit.0).await?.map(PyNumber))
        })
    }

    fn shared_or_local<'py>(
        &self,
        py: Python<'py>,
        counter: PyCounter,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counter = Counter::from(counter);
        redis_async(py, "async_get_cache", kind_error, async move {
            Ok(store.shared_or_local(&counter).await?.map(PyNumber))
        })
    }

    fn current_sync(
        &self,
        py: Python<'_>,
        counter: PyCounter,
        limit: PyNumber,
    ) -> PyResult<Option<PyNumber>> {
        let store = self.0.clone();
        let counter = Counter::from(counter);
        redis_sync(py, "get_cache", kind_error, async move {
            Ok(store.current(&counter, limit.0).await?.map(PyNumber))
        })
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(kind_error)
    }

    fn snapshot(&self) -> PyResult<HashMap<String, PyNumber>> {
        Ok(self
            .0
            .snapshot()
            .map_err(kind_error)?
            .into_iter()
            .map(|(key, count)| (key, PyNumber(count)))
            .collect())
    }
}
