use std::{convert::Infallible, sync::Arc, time::Duration};

use litellm_redis_store::spend_counters::{Error, LostSeed, Reading, SpendCounter, SpendCounters};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use super::{
    NativePending, NativeRedisStoreError, PyNumber, redis_async, redis_error_kind,
    request::NativeRequestRedis, store_error, ttl,
};
use crate::redis::client::NativeRedisClient;

fn spend_error(error: Error) -> PyErr {
    let kind = match &error {
        Error::Redis(error) => redis_error_kind(error),
        Error::Decode(_) => "decode",
        Error::Memory(_) => "memory",
    };
    NativeRedisStoreError::new_err((kind, litellm_error::Report::new(&error).to_string()))
}

fn infallible(error: Infallible) -> PyErr {
    match error {}
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
        redis_async(py, "async_get_cache", store_error, async move {
            Ok(match store.read(&counter.0).await? {
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
        redis_async(py, "async_batch_get_cache", infallible, async move {
            Ok(store.read_many(&counters).await.map(numbers))
        })
    }

    fn declare_read_many(
        &self,
        request: &NativeRequestRedis,
        counters: Vec<PySpendCounter>,
    ) -> Option<NativePending> {
        self.0
            .declare_read_many_pre_call(&request.0, &self::counters(counters))
            .map(|declared| {
                NativePending::new("async_batch_get_cache", infallible, async move {
                    Ok(declared.resolve().await.map(numbers))
                })
            })
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        counter: PySpendCounter,
        amount: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment", spend_error, async move {
            store.increment(&counter.0, amount.0).await.map(PyNumber)
        })
    }

    fn increment_many<'py>(
        &self,
        py: Python<'py>,
        increments: Vec<(PySpendCounter, PyNumber)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments = self::increments(increments);
        redis_async(py, "async_increment_pipeline", spend_error, async move {
            let totals = store.increment_many(&increments).await?;
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
        redis_async(py, "async_set_max", spend_error, async move {
            store
                .set_max(&counter.0, spend.0)
                .await
                .map(|stored| stored.map(PyNumber))
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
        redis_async(py, "async_set_cache", spend_error, async move {
            store.seed(&counter.0, spend.0, lost).await.map(PyNumber)
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
        redis_async(py, "async_set_cache", spend_error, async move {
            store.set(&counter.0, spend.0, ttl(ttl_seconds)).await
        })
    }

    fn delete<'py>(
        &self,
        py: Python<'py>,
        counters: Vec<PySpendCounter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counters = self::counters(counters);
        redis_async(py, "async_delete_cache", spend_error, async move {
            store.delete(&counters).await
        })
    }

    fn refresh_ttl<'py>(
        &self,
        py: Python<'py>,
        counter: PySpendCounter,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_refresh_ttl", spend_error, async move {
            store.refresh_ttl(&counter.0).await
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
