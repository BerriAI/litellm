use std::{sync::Arc, time::Duration};

use litellm_redis_store::rate_limits::{
    AcquireReply, Applied, AtomicCounter, AtomicReply, Counter, Entry, Error, Gauge,
    GuardedIncrement, GuardedReply, Increment, Local, Pending, RateLimits, Scope, WindowCount,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyString},
};

use super::{NativePending, PyNumber, redis_async, request::NativeRequestRedis, store_error, ttl};
use crate::execution::run_async_value;
use crate::redis::client::NativeRedisClient;

/// The v3 rate limiter's keyspace. Memory reads and writes are synchronous like `InMemoryCache`;
/// every Redis call is a future.
#[pyclass(frozen, name = "NativeRateLimits")]
pub(crate) struct NativeRateLimits(Arc<RateLimits>);

/// `(descriptor_key, descriptor_value)`.
struct PyScope(Scope);

impl<'a, 'py> FromPyObject<'a, 'py> for PyScope {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (descriptor, value): (String, String) = value.extract()?;
        Ok(Self(Scope { descriptor, value }))
    }
}

/// `(descriptor_key, descriptor_value, kind)`, the kind spelled as its key suffix.
struct PyCounter(Counter);

impl<'a, 'py> FromPyObject<'a, 'py> for PyCounter {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (descriptor, value, kind): (String, String, String) = value.extract()?;
        Ok(Self(Counter {
            scope: Scope { descriptor, value },
            kind: kind.parse().map_err(|_| unknown_suffix(&kind))?,
        }))
    }
}

fn unknown_suffix(suffix: &str) -> PyErr {
    PyValueError::new_err(format!("unknown rate-limit key suffix {suffix:?}"))
}

/// `(key_suffix, descriptor_key, descriptor_value)`, read by `Entry::new`.
struct PyEntry(Entry);

impl<'a, 'py> FromPyObject<'a, 'py> for PyEntry {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (suffix, descriptor, value): (String, String, String) = value.extract()?;
        Entry::new(&suffix, Scope { descriptor, value })
            .map(Self)
            .map_err(|_| unknown_suffix(&suffix))
    }
}

/// `(counter, amount, ttl_seconds | None)`.
struct PyIncrement(Increment);

impl<'a, 'py> FromPyObject<'a, 'py> for PyIncrement {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (counter, amount, ttl): (PyCounter, PyNumber, Option<u64>) = value.extract()?;
        Ok(Self(Increment {
            counter: counter.0,
            amount: amount.0,
            ttl: ttl.map(Duration::from_secs),
        }))
    }
}

/// `(counter, limit, increment, ttl_seconds, window_size_seconds)`.
struct PyAtomicCounter(AtomicCounter);

impl<'a, 'py> FromPyObject<'a, 'py> for PyAtomicCounter {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (counter, limit, increment, ttl_seconds, window_size_seconds): (
            PyCounter,
            i64,
            i64,
            i64,
            i64,
        ) = value.extract()?;
        Ok(Self(AtomicCounter {
            counter: counter.0,
            limit,
            increment,
            ttl_seconds,
            window_size_seconds,
        }))
    }
}

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

/// One script call declared on a batch: the request's pre-call batch, or a batch of its own when
/// no request was passed. Resolving it flushes that batch if it has not gone out.
fn pending<T, R>(
    call_type: &'static str,
    pending: Result<Pending<T>, Error>,
    reply: fn(T) -> R,
) -> PyResult<NativePending>
where
    T: Send + 'static,
    R: for<'a> IntoPyObject<'a> + Send + 'static,
{
    let pending = pending.map_err(rate_limit_error)?;
    Ok(NativePending::new(
        call_type,
        rate_limit_error,
        async move { pending.resolve().await.map(reply) },
    ))
}

fn rate_limit_error(error: Error) -> PyErr {
    PyRuntimeError::new_err(litellm_error::Report::new(&error).to_string())
}

fn scopes(scopes: Vec<PyScope>) -> Vec<Scope> {
    scopes.into_iter().map(|scope| scope.0).collect()
}

fn counters(counters: Vec<PyCounter>) -> Vec<Counter> {
    counters.into_iter().map(|counter| counter.0).collect()
}

fn increments(increments: Vec<PyIncrement>) -> Vec<Increment> {
    increments
        .into_iter()
        .map(|increment| increment.0)
        .collect()
}

fn request_of(
    request: Option<PyRef<'_, NativeRequestRedis>>,
) -> Option<litellm_redis_store::RequestRedis> {
    request.map(|request| request.0.clone())
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
        let backfill_ttl = ttl(backfill_ttl_seconds);
        Self(Arc::new(RateLimits::new(
            client.map(|client| client.0.clone()),
            backfill_ttl,
            Duration::from_secs(redis_ttl_seconds),
        )))
    }

    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn slot_groups(&self, scopes_: Vec<PyScope>, force: bool) -> Vec<Vec<usize>> {
        self.0.slot_groups(&scopes(scopes_), force)
    }

    fn local(&self, entries: Vec<PyEntry>) -> PyResult<Vec<Option<PyLocal>>> {
        entries
            .iter()
            .map(|entry| {
                self.0
                    .local(&entry.0)
                    .map(|local| local.map(PyLocal))
                    .map_err(store_error)
            })
            .collect()
    }

    #[pyo3(signature = (entry, value, ttl_seconds))]
    fn set_local(&self, entry: PyEntry, value: PyLocal, ttl_seconds: Option<f64>) -> PyResult<()> {
        self.0
            .set_local(&entry.0, value.0, ttl(ttl_seconds))
            .map_err(store_error)
    }

    #[pyo3(signature = (entry, *, backfill_ttl_seconds = None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        entry: PyEntry,
        backfill_ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let backfill_ttl = ttl(backfill_ttl_seconds);
        redis_async(py, "async_get_cache", store_error, async move {
            store
                .get(&entry.0, backfill_ttl)
                .await
                .map(|local| local.map(PyLocal))
        })
    }

    #[pyo3(signature = (counters_, now, window_size, *, request = None))]
    fn sliding_window(
        &self,
        counters_: Vec<PyCounter>,
        now: i64,
        window_size: i64,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        pending(
            "batch_rate_limiter_script",
            self.0
                .sliding_window(request.as_ref(), &counters(counters_), now, window_size),
            window_counts,
        )
    }

    #[pyo3(signature = (counters_, *, request = None))]
    fn read_counters(
        &self,
        counters_: Vec<PyCounter>,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        pending(
            "batch_counter_read_script",
            self.0.read_counters(request.as_ref(), &counters(counters_)),
            std::convert::identity,
        )
    }

    #[pyo3(signature = (counters_, *, request = None))]
    fn check_and_increment(
        &self,
        counters_: Vec<PyAtomicCounter>,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        let counters_ = counters_
            .into_iter()
            .map(|counter| counter.0)
            .collect::<Vec<_>>();
        pending(
            "check_and_increment_by_n_script",
            self.0.check_and_increment(request.as_ref(), &counters_),
            atomic,
        )
    }

    #[pyo3(signature = (counter, expected_window_start, amount, ttl_seconds, *, request = None))]
    fn guarded_increment(
        &self,
        counter: PyCounter,
        expected_window_start: String,
        amount: PyNumber,
        ttl_seconds: Option<u64>,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        let increment = GuardedIncrement {
            counter: counter.0,
            expected_window_start,
            amount: amount.0,
            ttl: ttl_seconds.map(Duration::from_secs),
        };
        pending(
            "window_guarded_token_increment_script",
            self.0.guarded_increment(request.as_ref(), &increment),
            guarded,
        )
    }

    #[pyo3(signature = (gauges, slot_ttl_seconds, slot_id, *, request = None))]
    fn parallel_acquire(
        &self,
        gauges: Vec<(PyScope, i64)>,
        slot_ttl_seconds: u64,
        slot_id: &str,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        let gauges = gauges
            .into_iter()
            .map(|(scope, limit)| Gauge {
                scope: scope.0,
                limit,
            })
            .collect::<Vec<_>>();
        pending(
            "parallel_acquire_script",
            self.0.parallel_acquire(
                request.as_ref(),
                &gauges,
                Duration::from_secs(slot_ttl_seconds),
                slot_id,
            ),
            acquire,
        )
    }

    #[pyo3(signature = (gauges, slot_id, *, request = None))]
    fn parallel_release(
        &self,
        gauges: Vec<PyScope>,
        slot_id: &str,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        pending(
            "parallel_release_script",
            self.0
                .parallel_release(request.as_ref(), &scopes(gauges), slot_id),
            std::convert::identity,
        )
    }

    #[pyo3(signature = (gauges, slot_ttl_seconds, *, request = None))]
    fn parallel_count(
        &self,
        gauges: Vec<PyScope>,
        slot_ttl_seconds: u64,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        pending(
            "parallel_count_script",
            self.0.parallel_count(
                request.as_ref(),
                &scopes(gauges),
                Duration::from_secs(slot_ttl_seconds),
            ),
            std::convert::identity,
        )
    }

    #[pyo3(signature = (increments_, *, request = None))]
    fn token_increment(
        &self,
        increments_: Vec<PyIncrement>,
        request: Option<PyRef<'_, NativeRequestRedis>>,
    ) -> PyResult<NativePending> {
        let request = request_of(request);
        pending(
            "token_increment_script",
            self.0
                .token_increment(request.as_ref(), &increments(increments_)),
            std::convert::identity,
        )
    }

    fn token_increment_post_call<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, NativeRequestRedis>,
        increments_: Vec<PyIncrement>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let request = request.0.clone();
        let increments_ = increments(increments_);
        run_async_value(py, async move {
            store
                .token_increment_post_call(&request, increments_)
                .map_err(rate_limit_error)
        })
    }

    fn parallel_release_post_call<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, NativeRequestRedis>,
        gauges: Vec<PyScope>,
        slot_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let request = request.0.clone();
        let gauges = scopes(gauges);
        run_async_value(py, async move {
            store
                .parallel_release_post_call(&request, &gauges, &slot_id)
                .map_err(rate_limit_error)
        })
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        increment: PyIncrement,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment", store_error, async move {
            store.increment(&increment.0).await.map(PyNumber)
        })
    }

    fn increment_many<'py>(
        &self,
        py: Python<'py>,
        increments_: Vec<PyIncrement>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments_ = increments(increments_);
        redis_async(py, "async_increment_pipeline", store_error, async move {
            store.increment_many(&increments_).await
        })
    }

    fn increment_many_post_call<'py>(
        &self,
        py: Python<'py>,
        request: PyRef<'_, NativeRequestRedis>,
        increments_: Vec<PyIncrement>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let request = request.0.clone();
        let increments_ = increments(increments_);
        run_async_value(py, async move {
            store
                .increment_many_post_call(&request, &increments_)
                .map_err(store_error)
        })
    }
}
