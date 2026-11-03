use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use litellm_redis_store::spend_counters::{
    DeclaredReads, Error, LostSeed, Reading, SpendCounter, SpendCounters,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use super::{PyNumber, call, request::NativeRequestRedis, store_error};
use crate::execution::run_async_value;
use crate::redis::client::NativeRedisClient;

pyo3::create_exception!(
    _native,
    NativeSpendCounterError,
    pyo3::exceptions::PyException,
    "A spend counter operation failed. Args are (kind, message), kind one of \"breaker_open\", \"timeout\", \"transport\", \"rejected\", \"decode\" or \"memory\"."
);

fn spend_error(error: Error) -> PyErr {
    use litellm_redis_client::Error as RedisError;
    let kind = match &error {
        Error::Redis(RedisError::BreakerOpen) => "breaker_open",
        Error::Redis(RedisError::Transport(error)) if error.is_timeout() => "timeout",
        Error::Redis(RedisError::Transport(_)) => "transport",
        Error::Redis(RedisError::Server(_)) => "rejected",
        Error::Redis(RedisError::Decode(_)) | Error::Decode(_) => "decode",
        Error::Memory(_) => "memory",
    };
    NativeSpendCounterError::new_err((kind, litellm_error::Report::new(&error).to_string()))
}

/// A `spend:*` key, refused when it names anything outside the spend counter keyspace.
pub(crate) struct PySpendCounter(SpendCounter);

impl<'a, 'py> FromPyObject<'a, 'py> for PySpendCounter {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let key: String = value.extract()?;
        SpendCounter::parse(&key)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

fn counters(counters: Vec<PySpendCounter>) -> Vec<SpendCounter> {
    counters.into_iter().map(|counter| counter.0).collect()
}

fn increments(
    increments: Vec<(PySpendCounter, PyNumber)>,
) -> Vec<(SpendCounter, litellm_redis_store::Number)> {
    increments
        .into_iter()
        .map(|(counter, amount)| (counter.0, amount.0))
        .collect()
}

fn ttl(seconds: Option<f64>) -> Option<Duration> {
    seconds.map(|seconds| Duration::from_secs_f64(seconds.max(0.0)))
}

fn numbers(values: Vec<Option<litellm_redis_store::Number>>) -> Vec<Option<PyNumber>> {
    values
        .into_iter()
        .map(|value| value.map(PyNumber))
        .collect()
}

/// The proxy's spend counters, Redis first with this worker's memory copy behind it.
#[pyclass(frozen, name = "NativeSpendCounters")]
pub(crate) struct NativeSpendCounters(Arc<SpendCounters>);

#[pymethods]
impl NativeSpendCounters {
    #[new]
    #[pyo3(signature = (client, redis_ttl_seconds = 60))]
    fn new(client: Option<PyRef<'_, NativeRedisClient>>, redis_ttl_seconds: u64) -> Self {
        Self(Arc::new(SpendCounters::new(
            client.map(|client| client.0.clone()),
            Duration::from_secs(redis_ttl_seconds),
        )))
    }

    /// `("redis" | "memory", value)`: where the answer came from, a Redis miss included.
    fn read<'py>(&self, py: Python<'py>, counter: PySpendCounter) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_get_cache")?;
        run_async_value(py, async move {
            let reading = scope
                .scope(store.read(&counter.0))
                .await
                .map_err(store_error)?;
            Ok(match reading {
                Reading::Redis(value) => ("redis", value.map(PyNumber)),
                Reading::Memory(value) => ("memory", value.map(PyNumber)),
            })
        })
    }

    fn read_many<'py>(
        &self,
        py: Python<'py>,
        counters: Vec<PySpendCounter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counters = self::counters(counters);
        let scope = call(py, "async_batch_get_cache")?;
        run_async_value(py, async move {
            Ok(scope.scope(store.read_many(&counters)).await.map(numbers))
        })
    }

    fn declare_read_many(
        &self,
        request: &NativeRequestRedis,
        counters: Vec<PySpendCounter>,
    ) -> Option<NativeDeclaredSpendCounters> {
        self.0
            .declare_read_many_pre_call(&request.0, &self::counters(counters))
            .map(|declared| NativeDeclaredSpendCounters(Mutex::new(Some(declared))))
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        counter: PySpendCounter,
        amount: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_increment")?;
        run_async_value(py, async move {
            scope
                .scope(store.increment(&counter.0, amount.0))
                .await
                .map(PyNumber)
                .map_err(spend_error)
        })
    }

    fn increment_many<'py>(
        &self,
        py: Python<'py>,
        increments: Vec<(PySpendCounter, PyNumber)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments = self::increments(increments);
        let scope = call(py, "async_increment_pipeline")?;
        run_async_value(py, async move {
            let totals = scope
                .scope(store.increment_many(&increments))
                .await
                .map_err(spend_error)?;
            Ok(totals.into_iter().map(PyNumber).collect::<Vec<_>>())
        })
    }

    /// Declares the increments on the request's post-call batch; `False` when there is no Redis.
    fn increment_many_post_call(
        &self,
        request: &NativeRequestRedis,
        increments: Vec<(PySpendCounter, PyNumber)>,
    ) -> bool {
        self.0
            .increment_many_post_call(&request.0, &self::increments(increments))
    }

    fn set_max<'py>(
        &self,
        py: Python<'py>,
        counter: PySpendCounter,
        spend: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_set_max")?;
        run_async_value(py, async move {
            scope
                .scope(store.set_max(&counter.0, spend.0))
                .await
                .map(|stored| stored.map(PyNumber))
                .map_err(spend_error)
        })
    }

    #[pyo3(signature = (counter, spend, *, increment_if_lost))]
    fn seed<'py>(
        &self,
        py: Python<'py>,
        counter: PySpendCounter,
        spend: PyNumber,
        increment_if_lost: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let lost = if increment_if_lost {
            LostSeed::Increment
        } else {
            LostSeed::Answer
        };
        let scope = call(py, "async_set_cache")?;
        run_async_value(py, async move {
            scope
                .scope(store.seed(&counter.0, spend.0, lost))
                .await
                .map(PyNumber)
                .map_err(spend_error)
        })
    }

    #[pyo3(signature = (counter, spend, ttl_seconds = None))]
    fn set<'py>(
        &self,
        py: Python<'py>,
        counter: PySpendCounter,
        spend: PyNumber,
        ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_set_cache")?;
        run_async_value(py, async move {
            scope
                .scope(store.set(&counter.0, spend.0, ttl(ttl_seconds)))
                .await
                .map_err(spend_error)
        })
    }

    fn delete<'py>(
        &self,
        py: Python<'py>,
        counters: Vec<PySpendCounter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counters = self::counters(counters);
        let scope = call(py, "async_delete_cache")?;
        run_async_value(py, async move {
            scope
                .scope(store.delete(&counters))
                .await
                .map_err(spend_error)
        })
    }

    fn refresh_ttl<'py>(
        &self,
        py: Python<'py>,
        counter: PySpendCounter,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_refresh_ttl")?;
        run_async_value(py, async move {
            scope
                .scope(store.refresh_ttl(&counter.0))
                .await
                .map_err(spend_error)
        })
    }

    fn get_local(&self, counter: PySpendCounter) -> PyResult<Option<PyNumber>> {
        Ok(self
            .0
            .get_local(&counter.0)
            .map_err(store_error)?
            .map(PyNumber))
    }

    #[pyo3(signature = (counter, spend, ttl_seconds = None))]
    fn set_local(
        &self,
        counter: PySpendCounter,
        spend: PyNumber,
        ttl_seconds: Option<f64>,
    ) -> PyResult<()> {
        self.0
            .set_local(&counter.0, spend.0, ttl(ttl_seconds))
            .map_err(store_error)
    }

    fn delete_local(&self, counter: PySpendCounter) -> PyResult<()> {
        self.0.delete_local(&counter.0).map_err(store_error)
    }

    fn increment_local(&self, counter: PySpendCounter, amount: PyNumber) -> PyResult<PyNumber> {
        self.0
            .increment_local(&counter.0, amount.0)
            .map(PyNumber)
            .map_err(store_error)
    }

    fn flush_local(&self) -> PyResult<()> {
        self.0.flush_local().map_err(store_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = PyDict::new(py);
        for (key, spend) in self.0.snapshot().map_err(store_error)? {
            snapshot.set_item(key, PyNumber(spend))?;
        }
        Ok(snapshot)
    }
}

/// A `read_many` declared on a request's pre-call batch. `resolve` answers it once.
#[pyclass(frozen, name = "NativeDeclaredSpendCounters")]
pub(crate) struct NativeDeclaredSpendCounters(Mutex<Option<DeclaredReads>>);

#[pymethods]
impl NativeDeclaredSpendCounters {
    fn resolve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let declared = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| {
                PyRuntimeError::new_err("this spend counter read was already resolved")
            })?;
        let scope = call(py, "async_batch_get_cache")?;
        run_async_value(py, async move {
            Ok(scope.scope(declared.resolve()).await.map(numbers))
        })
    }
}
