use std::{sync::Arc, time::Duration};

use litellm_redis_store::{
    Pending,
    rate_limits::{
        AcquireReply, Applied, AtomicCounter, AtomicReply, Counter, Entry, Gauge,
        GuardedIncrement, GuardedReply, Increment, Local, RateLimits, Scope, WindowCount,
    },
};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyString},
};

use super::{
    NativePending, PyData, PyNumber, redis_async, remote,
    request::{NativeRequestRedis, NativeSend},
    runtime_error, ttl, unwrap_all,
};
use crate::redis::client::NativeRedisClient;

/// The v3 rate limiter's keyspace. Memory reads and writes are synchronous like `InMemoryCache`;
/// every Redis call is a future.
#[pyclass(frozen, name = "NativeRateLimits")]
pub(crate) struct NativeRateLimits(Arc<RateLimits>);

/// An `int` or `float` counter, a `str` window start, or a `{slot_id: acquired_at}` registry.
struct PyLocal(Local);

impl<'a, 'py> FromPyObject<'a, 'py> for PyLocal {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(number) = value.extract::<PyNumber>() {
            return Ok(Self(Local::Number(number.0)));
        }
        if let Ok(text) = value.cast::<PyString>() {
            return Ok(Self(Local::Text(text.to_str()?.to_owned())));
        }
        if let Ok(slots) = value.cast::<PyDict>() {
            return slots
                .iter()
                .map(|(slot, acquired_at)| Ok((slot.extract()?, acquired_at.extract()?)))
                .collect::<PyResult<_>>()
                .map(|slots| Self(Local::Slots(slots)));
        }
        Err(PyTypeError::new_err(
            "a rate-limit memory value is an int, a float, a str or a dict of slot timestamps",
        ))
    }
}

impl<'py> IntoPyObject<'py> for PyLocal {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(match self.0 {
            Local::Number(number) => PyNumber(number).into_pyobject(py)?,
            Local::Text(text) => text.into_pyobject(py)?.into_any(),
            Local::Slots(slots) => {
                let registry = PyDict::new(py);
                for (slot, acquired_at) in slots {
                    registry.set_item(slot, acquired_at)?;
                }
                registry.into_any()
            }
        })
    }
}

type OverLimit = Option<(usize, i64, i64)>;

fn window_counts(counts: Vec<WindowCount>) -> Vec<(String, i64)> {
    counts
        .into_iter()
        .map(
            |WindowCount {
                 window_start,
                 count,
             }| (window_start, count),
        )
        .collect()
}

fn atomic(reply: AtomicReply) -> (OverLimit, Vec<(i64, i64)>) {
    match reply {
        AtomicReply::Applied(applied) => (
            None,
            applied
                .into_iter()
                .map(
                    |Applied {
                         count,
                         window_start,
                     }| (count, window_start),
                )
                .collect(),
        ),
        AtomicReply::OverLimit {
            index,
            current,
            limit,
        } => (Some((index, current, limit)), Vec::new()),
    }
}

fn guarded(GuardedReply { applied, count }: GuardedReply) -> (bool, i64) {
    (applied, count)
}

fn acquire(reply: AcquireReply) -> (OverLimit, Vec<i64>) {
    match reply {
        AcquireReply::Acquired(counts) => (None, counts),
        AcquireReply::OverLimit {
            index,
            in_flight,
            limit,
        } => (Some((index, in_flight, limit)), Vec::new()),
    }
}

/// One script call declared on the batch the caller's `NativeSend` picked; `None` without Redis.
fn script<T, R>(call_type: &'static str, declared: Option<Pending<T>>, reply: fn(T) -> R) -> Option<NativePending>
where
    T: Send + 'static,
    R: for<'a> IntoPyObject<'a> + Send + 'static,
{
    Some(super::pending(call_type, runtime_error, declared?, reply))
}

#[pymethods]
impl NativeRateLimits {
    #[new]
    #[pyo3(signature = (client, backfill_ttl_seconds = None, redis_ttl_seconds = 60))]
    fn new(
        client: Option<PyRef<'_, NativeRedisClient>>,
        backfill_ttl_seconds: Option<f64>,
        redis_ttl_seconds: u64,
    ) -> Self {
        Self(Arc::new(RateLimits::new(
            client.map(|client| remote(&client, redis_ttl_seconds)),
            ttl(backfill_ttl_seconds),
        )))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn slot_groups(&self, scopes: Vec<PyData<Scope>>, force: bool) -> Vec<Vec<usize>> {
        self.0.slot_groups(&unwrap_all(scopes), force)
    }

    fn get_local(&self, entry: PyData<Entry>) -> PyResult<Option<PyLocal>> {
        Ok(self.0.get_local(&entry.0).map_err(runtime_error)?.map(PyLocal))
    }

    #[pyo3(signature = (entry, value, ttl_seconds))]
    fn set_local(
        &self,
        entry: PyData<Entry>,
        value: PyLocal,
        ttl_seconds: Option<f64>,
    ) -> PyResult<()> {
        self.0
            .set_local(&entry.0, value.0, ttl(ttl_seconds))
            .map_err(runtime_error)
    }

    #[pyo3(signature = (entry, *, backfill_ttl_seconds = None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        entry: PyData<Entry>,
        backfill_ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let backfill_ttl = ttl(backfill_ttl_seconds);
        redis_async(py, "async_get_cache", runtime_error, async move {
            store
                .get(&entry.0, backfill_ttl)
                .await
                .map(|local| local.map(PyLocal))
        })
    }

    fn sliding_window(
        &self,
        counters: Vec<PyData<Counter>>,
        now: i64,
        window_size: i64,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "batch_rate_limiter_script",
            self.0
                .sliding_window(send.send(), &unwrap_all(counters), now, window_size),
            window_counts,
        )
    }

    fn read_counters(
        &self,
        counters: Vec<PyData<Counter>>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "batch_counter_read_script",
            self.0.read_counters(send.send(), &unwrap_all(counters)),
            std::convert::identity,
        )
    }

    fn check_and_increment(
        &self,
        counters: Vec<PyData<AtomicCounter>>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "check_and_increment_by_n_script",
            self.0
                .check_and_increment(send.send(), &unwrap_all(counters)),
            atomic,
        )
    }

    fn guarded_increment(
        &self,
        counter: PyData<Counter>,
        expected_window_start: String,
        amount: PyNumber,
        ttl_seconds: Option<u64>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        let increment = GuardedIncrement {
            counter: counter.0,
            expected_window_start,
            amount: amount.0,
            ttl: ttl_seconds.map(Duration::from_secs),
        };
        script(
            "window_guarded_token_increment_script",
            self.0.guarded_increment(send.send(), &increment),
            guarded,
        )
    }

    fn parallel_acquire(
        &self,
        gauges: Vec<PyData<Gauge>>,
        slot_ttl_seconds: u64,
        slot_id: &str,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "parallel_acquire_script",
            self.0.parallel_acquire(
                send.send(),
                &unwrap_all(gauges),
                Duration::from_secs(slot_ttl_seconds),
                slot_id,
            ),
            acquire,
        )
    }

    fn parallel_release(
        &self,
        gauges: Vec<PyData<Scope>>,
        slot_id: &str,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "parallel_release_script",
            self.0
                .parallel_release(send.send(), &unwrap_all(gauges), slot_id),
            std::convert::identity,
        )
    }

    fn parallel_count(
        &self,
        gauges: Vec<PyData<Scope>>,
        slot_ttl_seconds: u64,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "parallel_count_script",
            self.0.parallel_count(
                send.send(),
                &unwrap_all(gauges),
                Duration::from_secs(slot_ttl_seconds),
            ),
            std::convert::identity,
        )
    }

    fn token_increment(
        &self,
        increments: Vec<PyData<Increment>>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "token_increment_script",
            self.0.token_increment(send.send(), &unwrap_all(increments)),
            std::convert::identity,
        )
    }

    /// `False` without Redis.
    fn token_increment_post_call(
        &self,
        request: &NativeRequestRedis,
        increments: Vec<PyData<Increment>>,
    ) -> bool {
        self.0.token_increment_post_call(&request.0, unwrap_all(increments))
    }

    /// `False` without Redis.
    fn parallel_release_post_call(
        &self,
        request: &NativeRequestRedis,
        gauges: Vec<PyData<Scope>>,
        slot_id: &str,
    ) -> bool {
        self.0
            .parallel_release_post_call(&request.0, &unwrap_all(gauges), slot_id)
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        increment: PyData<Increment>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment", runtime_error, async move {
            store.increment(&increment.0).await.map(PyNumber)
        })
    }

    fn increment_many<'py>(
        &self,
        py: Python<'py>,
        increments: Vec<PyData<Increment>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments = unwrap_all(increments);
        redis_async(py, "async_increment_pipeline", runtime_error, async move {
            store.increment_many(&increments).await
        })
    }

    fn increment_many_post_call(
        &self,
        request: &NativeRequestRedis,
        increments: Vec<PyData<Increment>>,
    ) -> PyResult<()> {
        self.0
            .increment_many_post_call(&request.0, &unwrap_all(increments))
            .map_err(runtime_error)
    }
}
