use std::{sync::Arc, time::Duration};

use indexmap::IndexMap;
use litellm_redis_store::{
    Pending,
    rate_limits::{
        AcquireReply, Applied, AtomicCounter, AtomicReply, Counter, Entry, Gauge, GuardedIncrement,
        GuardedReply, Increment, Kind, Local, RateLimits, Scope, WindowCount,
    },
};
use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    NativePending, PyNumber, Spelled, kind_error, redis_async, remote,
    request::{NativeRequestRedis, NativeSend},
    ttl,
};
use crate::redis::client::NativeRedisClient;

/// The v3 rate limiter's keyspace. Memory reads and writes are synchronous like `InMemoryCache`;
/// every Redis call is a future.
#[pyclass(frozen, name = "NativeRateLimits")]
pub(crate) struct NativeRateLimits(Arc<RateLimits>);

/// An `int` or `float` counter, a `str` window start, or a `{slot_id: acquired_at}` registry.
#[derive(FromPyObject)]
enum PyLocal {
    Number(PyNumber),
    Text(String),
    Slots(IndexMap<String, f64>),
}

impl From<PyLocal> for Local {
    fn from(local: PyLocal) -> Self {
        match local {
            PyLocal::Number(number) => Self::Number(number.0),
            PyLocal::Text(text) => Self::Text(text),
            PyLocal::Slots(slots) => Self::Slots(slots.into_iter().collect()),
        }
    }
}

/// What the memory tier holds, back as the Python value it was written as.
struct LocalValue(Local);

impl<'py> IntoPyObject<'py> for LocalValue {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(match self.0 {
            Local::Number(number) => PyNumber(number).into_pyobject(py)?,
            Local::Text(text) => text.into_pyobject(py)?.into_any(),
            Local::Slots(slots) => slots
                .into_iter()
                .collect::<IndexMap<_, _>>()
                .into_pyobject(py)?
                .into_any(),
        })
    }
}

/// `litellm.redis.stores.rate_limits.RateLimitScope`.
#[derive(FromPyObject)]
struct PyScope {
    descriptor_key: String,
    descriptor_value: String,
}

impl From<PyScope> for Scope {
    fn from(scope: PyScope) -> Self {
        Self {
            descriptor: scope.descriptor_key,
            value: scope.descriptor_value,
        }
    }
}

/// `RateLimitCounter`.
#[derive(FromPyObject)]
struct PyCounter {
    scope: PyScope,
    kind: Spelled<Kind>,
}

impl From<PyCounter> for Counter {
    fn from(counter: PyCounter) -> Self {
        Self {
            scope: counter.scope.into(),
            kind: counter.kind.0,
        }
    }
}

/// `RateLimitWindow`, `RateLimitCounter` or `RateLimitGauge`, told apart by the key suffix each names.
#[derive(FromPyObject)]
struct PyEntry {
    suffix: String,
    scope: PyScope,
}

impl TryFrom<PyEntry> for Entry {
    type Error = PyErr;

    fn try_from(entry: PyEntry) -> PyResult<Self> {
        Self::new(&entry.suffix, entry.scope.into())
            .map_err(|_| PyValueError::new_err(format!("unknown key suffix {:?}", entry.suffix)))
    }
}

/// `CounterIncrement`.
#[derive(FromPyObject)]
struct PyIncrement {
    counter: PyCounter,
    amount: PyNumber,
    ttl: Option<u64>,
}

impl From<PyIncrement> for Increment {
    fn from(increment: PyIncrement) -> Self {
        Self {
            counter: increment.counter.into(),
            amount: increment.amount.0,
            ttl: increment.ttl.map(Duration::from_secs),
        }
    }
}

/// `AtomicCounter`.
#[derive(FromPyObject)]
struct PyAtomicCounter {
    counter: PyCounter,
    limit: i64,
    increment: i64,
    ttl: i64,
    window_size: i64,
}

impl From<PyAtomicCounter> for AtomicCounter {
    fn from(counter: PyAtomicCounter) -> Self {
        Self {
            counter: counter.counter.into(),
            limit: counter.limit,
            increment: counter.increment,
            ttl_seconds: counter.ttl,
            window_size_seconds: counter.window_size,
        }
    }
}

fn all<P, T: From<P>>(records: Vec<P>) -> Vec<T> {
    records.into_iter().map(T::from).collect()
}

#[pyclass(frozen, eq, get_all, name = "WindowCount")]
#[derive(PartialEq)]
pub(crate) struct PyWindowCount {
    window_start: String,
    count: i64,
}

#[pymethods]
impl PyWindowCount {
    #[new]
    fn new(window_start: String, count: i64) -> Self {
        Self {
            window_start,
            count,
        }
    }
}

/// Per counter, the count after the increment and the start of the window it landed in.
#[pyclass(frozen, eq, get_all, name = "AtomicApplied")]
#[derive(PartialEq)]
pub(crate) struct PyAtomicApplied {
    counters: Vec<(i64, i64)>,
}

#[pymethods]
impl PyAtomicApplied {
    #[new]
    fn new(counters: Vec<(i64, i64)>) -> Self {
        Self { counters }
    }
}

#[pyclass(frozen, eq, get_all, name = "AtomicOverLimit")]
#[derive(PartialEq)]
pub(crate) struct PyAtomicOverLimit {
    index: usize,
    current: i64,
    limit: i64,
}

#[pymethods]
impl PyAtomicOverLimit {
    #[new]
    fn new(index: usize, current: i64, limit: i64) -> Self {
        Self {
            index,
            current,
            limit,
        }
    }
}

#[derive(IntoPyObject)]
enum PyAtomicReply {
    Applied(PyAtomicApplied),
    OverLimit(PyAtomicOverLimit),
}

#[pyclass(frozen, eq, get_all, name = "GuardedReply")]
#[derive(PartialEq)]
pub(crate) struct PyGuardedReply {
    applied: bool,
    count: i64,
}

#[pymethods]
impl PyGuardedReply {
    #[new]
    fn new(applied: bool, count: i64) -> Self {
        Self { applied, count }
    }
}

#[pyclass(frozen, eq, get_all, name = "GaugesAcquired")]
#[derive(PartialEq)]
pub(crate) struct PyGaugesAcquired {
    in_flight: Vec<i64>,
}

#[pymethods]
impl PyGaugesAcquired {
    #[new]
    fn new(in_flight: Vec<i64>) -> Self {
        Self { in_flight }
    }
}

#[pyclass(frozen, eq, get_all, name = "GaugeOverLimit")]
#[derive(PartialEq)]
pub(crate) struct PyGaugeOverLimit {
    index: usize,
    in_flight: i64,
    limit: i64,
}

#[pymethods]
impl PyGaugeOverLimit {
    #[new]
    fn new(index: usize, in_flight: i64, limit: i64) -> Self {
        Self {
            index,
            in_flight,
            limit,
        }
    }
}

#[derive(IntoPyObject)]
enum PyAcquireReply {
    Acquired(PyGaugesAcquired),
    OverLimit(PyGaugeOverLimit),
}

fn window_counts(counts: Vec<WindowCount>) -> Vec<PyWindowCount> {
    counts
        .into_iter()
        .map(
            |WindowCount {
                 window_start,
                 count,
             }| PyWindowCount {
                window_start,
                count,
            },
        )
        .collect()
}

fn atomic(reply: AtomicReply) -> PyAtomicReply {
    match reply {
        AtomicReply::Applied(applied) => PyAtomicReply::Applied(PyAtomicApplied {
            counters: applied
                .into_iter()
                .map(
                    |Applied {
                         count,
                         window_start,
                     }| (count, window_start),
                )
                .collect(),
        }),
        AtomicReply::OverLimit {
            index,
            current,
            limit,
        } => PyAtomicReply::OverLimit(PyAtomicOverLimit {
            index,
            current,
            limit,
        }),
    }
}

fn guarded(GuardedReply { applied, count }: GuardedReply) -> PyGuardedReply {
    PyGuardedReply { applied, count }
}

fn acquire(reply: AcquireReply) -> PyAcquireReply {
    match reply {
        AcquireReply::Acquired(in_flight) => {
            PyAcquireReply::Acquired(PyGaugesAcquired { in_flight })
        }
        AcquireReply::OverLimit {
            index,
            in_flight,
            limit,
        } => PyAcquireReply::OverLimit(PyGaugeOverLimit {
            index,
            in_flight,
            limit,
        }),
    }
}

/// One script call declared on the batch the caller's `NativeSend` picked; `None` without Redis.
fn script<T, R>(
    call_type: &'static str,
    declared: Option<Pending<T>>,
    reply: fn(T) -> R,
) -> Option<NativePending>
where
    T: Send + 'static,
    R: for<'a> IntoPyObject<'a> + Send + 'static,
{
    Some(super::pending(call_type, kind_error, declared?, reply))
}

#[pymethods]
impl NativeRateLimits {
    #[new]
    #[pyo3(signature = (client, backfill_ttl_seconds = None))]
    fn new(
        client: Option<PyRef<'_, NativeRedisClient>>,
        backfill_ttl_seconds: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(Arc::new(RateLimits::new(
            client.map(|client| remote(&client)),
            ttl(backfill_ttl_seconds)?,
        ))))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn slot_groups(&self, scopes: Vec<PyScope>, force: bool) -> Vec<Vec<usize>> {
        self.0.slot_groups(&all(scopes), force)
    }

    fn get_local(&self, entry: PyEntry) -> PyResult<Option<LocalValue>> {
        Ok(self
            .0
            .get_local(&entry.try_into()?)
            .map_err(kind_error)?
            .map(LocalValue))
    }

    #[pyo3(signature = (entry, value, ttl_seconds))]
    fn set_local(&self, entry: PyEntry, value: PyLocal, ttl_seconds: Option<f64>) -> PyResult<()> {
        self.0
            .set_local(&entry.try_into()?, value.into(), ttl(ttl_seconds)?)
            .map_err(kind_error)
    }

    /// `request` answers a key its pre-call reads already found absent without another `GET`.
    #[pyo3(signature = (entry, *, backfill_ttl_seconds = None, request = None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        entry: PyEntry,
        backfill_ttl_seconds: Option<f64>,
        request: Option<&NativeRequestRedis>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let entry = Entry::try_from(entry)?;
        let backfill_ttl = ttl(backfill_ttl_seconds)?;
        let request = request.map(|request| request.0.clone());
        redis_async(py, "async_get_cache", kind_error, async move {
            store
                .get(request.as_ref(), &entry, backfill_ttl)
                .await
                .map(|local| local.map(LocalValue))
        })
    }

    fn sliding_window(
        &self,
        counters: Vec<PyCounter>,
        now: i64,
        window_size: i64,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "batch_rate_limiter_script",
            self.0
                .sliding_window(send.send(), &all(counters), now, window_size),
            window_counts,
        )
    }

    fn read_counters(&self, counters: Vec<PyCounter>, send: &NativeSend) -> Option<NativePending> {
        script(
            "batch_counter_read_script",
            self.0.read_counters(send.send(), &all(counters)),
            std::convert::identity,
        )
    }

    fn check_and_increment(
        &self,
        counters: Vec<PyAtomicCounter>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "check_and_increment_by_n_script",
            self.0.check_and_increment(send.send(), &all(counters)),
            atomic,
        )
    }

    fn guarded_increment(
        &self,
        counter: PyCounter,
        expected_window_start: String,
        amount: PyNumber,
        ttl_seconds: Option<u64>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        let increment = GuardedIncrement {
            counter: counter.into(),
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
        gauges: Vec<(PyScope, i64)>,
        slot_ttl_seconds: u64,
        slot_id: &str,
        send: &NativeSend,
    ) -> Option<NativePending> {
        let gauges: Vec<Gauge> = gauges
            .into_iter()
            .map(|(scope, limit)| Gauge {
                scope: scope.into(),
                limit,
            })
            .collect();
        script(
            "parallel_acquire_script",
            self.0.parallel_acquire(
                send.send(),
                &gauges,
                Duration::from_secs(slot_ttl_seconds),
                slot_id,
            ),
            acquire,
        )
    }

    fn parallel_release(
        &self,
        gauges: Vec<PyScope>,
        slot_id: &str,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "parallel_release_script",
            self.0.parallel_release(send.send(), &all(gauges), slot_id),
            std::convert::identity,
        )
    }

    fn parallel_count(
        &self,
        gauges: Vec<PyScope>,
        slot_ttl_seconds: u64,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "parallel_count_script",
            self.0.parallel_count(
                send.send(),
                &all(gauges),
                Duration::from_secs(slot_ttl_seconds),
            ),
            std::convert::identity,
        )
    }

    fn token_increment(
        &self,
        increments: Vec<PyIncrement>,
        send: &NativeSend,
    ) -> Option<NativePending> {
        script(
            "token_increment_script",
            self.0.token_increment(send.send(), &all(increments)),
            std::convert::identity,
        )
    }

    /// `False` without Redis.
    fn token_increment_post_call(
        &self,
        request: &NativeRequestRedis,
        increments: Vec<PyIncrement>,
    ) -> bool {
        self.0
            .token_increment_post_call(&request.0, all(increments))
    }

    /// `False` without Redis.
    fn parallel_release_post_call(
        &self,
        request: &NativeRequestRedis,
        gauges: Vec<PyScope>,
        slot_id: &str,
    ) -> bool {
        self.0
            .parallel_release_post_call(&request.0, &all(gauges), slot_id)
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        increment: PyIncrement,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increment = Increment::from(increment);
        redis_async(py, "async_increment", kind_error, async move {
            Ok(store.increment(&increment).await.map(PyNumber))
        })
    }

    fn increment_many<'py>(
        &self,
        py: Python<'py>,
        increments: Vec<PyIncrement>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let increments: Vec<Increment> = all(increments);
        redis_async(py, "async_increment_pipeline", kind_error, async move {
            store.increment_many(&increments).await;
            Ok(())
        })
    }

    fn increment_many_post_call(&self, request: &NativeRequestRedis, increments: Vec<PyIncrement>) {
        self.0
            .increment_many_post_call(&request.0, &all(increments));
    }
}
