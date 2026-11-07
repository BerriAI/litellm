use std::{collections::HashMap, sync::Arc};

use litellm_redis_store::{
    Number,
    spend_counters::{LostSeed, Reading, SpendCounter, SpendCounters},
};
use pyo3::prelude::*;

use super::{
    NativePending, PyData, PyNumber, kind_error, pending, redis_async, remote,
    request::{NativeRequestRedis, NativeSend},
    ttl, unwrap_all,
};
use crate::redis::client::NativeRedisClient;

/// A counter's value and whether Redis answered it, a clean miss included.
#[pyclass(frozen, eq, get_all, name = "SpendRead")]
#[derive(PartialEq)]
pub(crate) struct PySpendRead {
    value: Option<PyNumber>,
    from_redis: bool,
}

#[pymethods]
impl PySpendRead {
    #[new]
    fn new(value: Option<PyNumber>, from_redis: bool) -> Self {
        Self { value, from_redis }
    }
}

fn increments(increments: Vec<(PyData<SpendCounter>, PyNumber)>) -> Vec<(SpendCounter, Number)> {
    increments
        .into_iter()
        .map(|(counter, amount)| (counter.0, amount.0))
        .collect()
}

fn numbers(values: Option<Vec<Option<Number>>>) -> Option<Vec<Option<PyNumber>>> {
    values.map(|values| {
        values
            .into_iter()
            .map(|value| value.map(PyNumber))
            .collect()
    })
}

/// The proxy's spend counters, Redis first with this worker's memory copy behind it.
#[pyclass(frozen, name = "NativeSpendCounters")]
pub(crate) struct NativeSpendCounters(Arc<SpendCounters>);

#[pymethods]
impl NativeSpendCounters {
    #[new]
    fn new(client: Option<PyRef<'_, NativeRedisClient>>) -> Self {
        Self(Arc::new(SpendCounters::new(
            client.map(|client| remote(&client)),
        )))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    fn attach(&self, client: PyRef<'_, NativeRedisClient>) -> bool {
        self.0.attach(remote(&client))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    #[getter]
    fn redis_ttl_seconds(&self) -> Option<f64> {
        self.0.redis_ttl().map(|ttl| ttl.as_secs_f64())
    }

    /// `("redis" | "memory", value)`: where the answer came from, a Redis miss included.
    fn read<'py>(
        &self,
        py: Python<'py>,
        counter: PyData<SpendCounter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_get_cache", kind_error, async move {
            let reading = store.read(&counter.0).await?;
            Ok(PySpendRead {
                value: reading.value().map(PyNumber),
                from_redis: matches!(reading, Reading::Redis(_)),
            })
        })
    }

    /// One `MGET` on the batch `send` picks; `None` without Redis. It resolves to `None` when the
    /// read failed, so each caller reads on its own.
    fn read_many(
        &self,
        counters: Vec<PyData<SpendCounter>>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        let read = self.0.read_many(send.send(), &unwrap_all(counters))?;
        Some(pending("async_batch_get_cache", kind_error, read, numbers))
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        counter: PyData<SpendCounter>,
        amount: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment", kind_error, async move {
            store.increment(&counter.0, amount.0).await.map(PyNumber)
        })
    }

    fn increment_many<'py>(
        &self,
        py: Python<'py>,
        increments: Vec<(PyData<SpendCounter>, PyNumber)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments = self::increments(increments);
        redis_async(py, "async_increment_pipeline", kind_error, async move {
            let totals = store.increment_many(&increments).await?;
            Ok(totals.into_iter().map(PyNumber).collect::<Vec<_>>())
        })
    }

    /// Declares the increments on the request's post-call batch; `False` when there is no Redis.
    fn increment_many_post_call(
        &self,
        request: &NativeRequestRedis,
        increments: Vec<(PyData<SpendCounter>, PyNumber)>,
    ) -> bool {
        self.0
            .increment_many_post_call(&request.0, &self::increments(increments))
    }

    fn set_max<'py>(
        &self,
        py: Python<'py>,
        counter: PyData<SpendCounter>,
        spend: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_set_max", kind_error, async move {
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
        counter: PyData<SpendCounter>,
        spend: PyNumber,
        increment_if_lost: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let lost = if increment_if_lost {
            LostSeed::Increment
        } else {
            LostSeed::Answer
        };
        redis_async(py, "async_set_cache", kind_error, async move {
            store.seed(&counter.0, spend.0, lost).await.map(PyNumber)
        })
    }

    #[pyo3(signature = (counter, spend, ttl_seconds = None))]
    fn set<'py>(
        &self,
        py: Python<'py>,
        counter: PyData<SpendCounter>,
        spend: PyNumber,
        ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let ttl = ttl(ttl_seconds)?;
        redis_async(py, "async_set_cache", kind_error, async move {
            store.set(&counter.0, spend.0, ttl).await
        })
    }

    fn delete<'py>(
        &self,
        py: Python<'py>,
        counters: Vec<PyData<SpendCounter>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counters = unwrap_all(counters);
        redis_async(py, "async_delete_cache", kind_error, async move {
            store.delete(&counters).await
        })
    }

    fn refresh_ttl<'py>(
        &self,
        py: Python<'py>,
        counter: PyData<SpendCounter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_refresh_ttl", kind_error, async move {
            store.refresh_ttl(&counter.0).await
        })
    }

    fn get_local(&self, counter: PyData<SpendCounter>) -> PyResult<Option<PyNumber>> {
        Ok(self
            .0
            .get_local(&counter.0)
            .map_err(kind_error)?
            .map(PyNumber))
    }

    #[pyo3(signature = (counter, spend, ttl_seconds = None))]
    fn set_local(
        &self,
        counter: PyData<SpendCounter>,
        spend: PyNumber,
        ttl_seconds: Option<f64>,
    ) -> PyResult<()> {
        self.0
            .set_local(&counter.0, spend.0, ttl(ttl_seconds)?)
            .map_err(kind_error)
    }

    fn delete_local(&self, counter: PyData<SpendCounter>) -> PyResult<()> {
        self.0.delete_local(&counter.0).map_err(kind_error)
    }

    fn increment_local(
        &self,
        counter: PyData<SpendCounter>,
        amount: PyNumber,
    ) -> PyResult<PyNumber> {
        self.0
            .increment_local(&counter.0, amount.0)
            .map(PyNumber)
            .map_err(kind_error)
    }

    fn flush_local(&self) -> PyResult<()> {
        self.0.flush_local().map_err(kind_error)
    }

    fn snapshot(&self) -> PyResult<HashMap<String, PyNumber>> {
        Ok(self
            .0
            .snapshot()
            .map_err(kind_error)?
            .into_iter()
            .map(|(key, spend)| (key, PyNumber(spend)))
            .collect())
    }
}
