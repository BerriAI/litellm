use std::{sync::LazyLock, time::Duration};

use litellm_python_compat::repr::float_repr;
use litellm_redis_client::{Declared, RedisBatch, Script, Value};

use super::{Counter, Error, Increment, Scope, lua};
use crate::Number;

static BATCH_RATE_LIMITER: LazyLock<Script> =
    LazyLock::new(|| Script::new(lua::BATCH_RATE_LIMITER));
static BATCH_COUNTER_READ: LazyLock<Script> =
    LazyLock::new(|| Script::new(lua::BATCH_COUNTER_READ));
static CHECK_AND_INCREMENT_BY_N: LazyLock<Script> =
    LazyLock::new(|| Script::new(lua::CHECK_AND_INCREMENT_BY_N));
static WINDOW_GUARDED_TOKEN_INCREMENT: LazyLock<Script> =
    LazyLock::new(|| Script::new(lua::WINDOW_GUARDED_TOKEN_INCREMENT));
static PARALLEL_ACQUIRE: LazyLock<Script> = LazyLock::new(|| Script::new(lua::PARALLEL_ACQUIRE));
static PARALLEL_RELEASE: LazyLock<Script> = LazyLock::new(|| Script::new(lua::PARALLEL_RELEASE));
static PARALLEL_COUNT: LazyLock<Script> = LazyLock::new(|| Script::new(lua::PARALLEL_COUNT));
static TOKEN_INCREMENT: LazyLock<Script> = LazyLock::new(|| Script::new(lua::TOKEN_INCREMENT));

/// One script call declared on a batch, decoded into its typed reply once the batch has run.
#[must_use]
pub struct Pending<T> {
    reply: Declared<Value>,
    decode: fn(Value) -> Result<T, Error>,
}

impl<T> Pending<T> {
    fn new(reply: Declared<Value>, decode: fn(Value) -> Result<T, Error>) -> Self {
        Self { reply, decode }
    }

    /// Flushes the batch when this op has not gone out yet.
    pub async fn resolve(self) -> Result<T, Error> {
        (self.decode)(self.reply.resolve().await?)
    }

    /// Waits for whoever flushes the batch.
    pub async fn settled(self) -> Result<T, Error> {
        (self.decode)(self.reply.settled().await?)
    }
}

/// `BATCH_RATE_LIMITER_SCRIPT`'s pair for one counter: the window start it saw or opened and the
/// counter after its `+1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowCount {
    pub window_start: String,
    pub count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtomicCounter {
    pub counter: Counter,
    pub limit: i64,
    pub increment: i64,
    pub ttl_seconds: i64,
    pub window_size_seconds: i64,
}

/// `CHECK_AND_INCREMENT_BY_N_SCRIPT`'s reply. `index` is 0-based where the Lua's is 1-based.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomicReply {
    Applied(Vec<Applied>),
    OverLimit {
        index: usize,
        current: i64,
        limit: i64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Applied {
    pub count: i64,
    pub window_start: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GuardedIncrement {
    pub counter: Counter,
    pub expected_window_start: String,
    pub amount: Number,
    pub ttl: Option<Duration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardedReply {
    pub applied: bool,
    pub count: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gauge {
    pub scope: Scope,
    pub limit: i64,
}

/// `PARALLEL_ACQUIRE_SCRIPT`'s reply. `index` is 0-based where the Lua's is 1-based.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AcquireReply {
    Acquired(Vec<i64>),
    OverLimit {
        index: usize,
        in_flight: i64,
        limit: i64,
    },
}

pub(super) fn sliding_window(
    batch: &RedisBatch,
    counters: &[Counter],
    now: i64,
    window_size: i64,
) -> Pending<Vec<WindowCount>> {
    let keys = counters
        .iter()
        .flat_map(|counter| [counter.scope.window_key(), counter.key()])
        .collect::<Vec<_>>();
    Pending::new(
        eval(
            batch,
            &BATCH_RATE_LIMITER,
            &keys,
            vec![now.to_string(), window_size.to_string()],
        ),
        window_counts,
    )
}

/// `BATCH_COUNTER_READ_SCRIPT` over each counter's window and counter keys, in that order.
pub(super) fn read_counters(
    batch: &RedisBatch,
    counters: &[Counter],
) -> Pending<Vec<Option<String>>> {
    let keys = counters
        .iter()
        .flat_map(|counter| [counter.scope.window_key(), counter.key()])
        .collect::<Vec<_>>();
    Pending::new(
        eval(batch, &BATCH_COUNTER_READ, &keys, Vec::new()),
        optional_texts,
    )
}

pub(super) fn check_and_increment(
    batch: &RedisBatch,
    counters: &[AtomicCounter],
) -> Pending<AtomicReply> {
    let keys = counters
        .iter()
        .flat_map(|counter| [counter.counter.scope.window_key(), counter.counter.key()])
        .collect::<Vec<_>>();
    let args = counters
        .iter()
        .flat_map(|counter| {
            [
                counter.limit,
                counter.increment,
                counter.ttl_seconds,
                counter.window_size_seconds,
            ]
            .map(|arg| arg.to_string())
        })
        .collect();
    Pending::new(
        eval(batch, &CHECK_AND_INCREMENT_BY_N, &keys, args),
        atomic_reply,
    )
}

pub(super) fn guarded_increment(
    batch: &RedisBatch,
    increment: &GuardedIncrement,
) -> Pending<GuardedReply> {
    let keys = [
        increment.counter.scope.window_key(),
        increment.counter.key(),
    ];
    let args = vec![
        increment.expected_window_start.clone(),
        python_arg(increment.amount),
        ttl_arg(increment.ttl),
    ];
    Pending::new(
        eval(batch, &WINDOW_GUARDED_TOKEN_INCREMENT, &keys, args),
        guarded_reply,
    )
}

pub(super) fn parallel_acquire(
    batch: &RedisBatch,
    gauges: &[Gauge],
    slot_ttl: Duration,
    slot_id: &str,
) -> Pending<AcquireReply> {
    let keys = gauges
        .iter()
        .map(|gauge| gauge.scope.gauge_key())
        .collect::<Vec<_>>();
    let args = gauges
        .iter()
        .flat_map(|gauge| {
            [
                gauge.limit.to_string(),
                slot_ttl.as_secs().to_string(),
                slot_id.to_owned(),
            ]
        })
        .collect();
    Pending::new(eval(batch, &PARALLEL_ACQUIRE, &keys, args), acquire_reply)
}

pub(super) fn parallel_release(
    batch: &RedisBatch,
    gauges: &[Scope],
    slot_id: &str,
) -> Pending<Vec<i64>> {
    let keys = gauges.iter().map(Scope::gauge_key).collect::<Vec<_>>();
    let args = gauges.iter().map(|_| slot_id.to_owned()).collect();
    Pending::new(eval(batch, &PARALLEL_RELEASE, &keys, args), integers)
}

pub(super) fn parallel_count(
    batch: &RedisBatch,
    gauges: &[Scope],
    slot_ttl: Duration,
) -> Pending<Vec<i64>> {
    let keys = gauges.iter().map(Scope::gauge_key).collect::<Vec<_>>();
    let args = gauges
        .iter()
        .map(|_| slot_ttl.as_secs().to_string())
        .collect();
    Pending::new(eval(batch, &PARALLEL_COUNT, &keys, args), integers)
}

pub(super) fn token_increment(batch: &RedisBatch, increments: &[Increment]) -> Pending<Vec<f64>> {
    let keys = increments
        .iter()
        .map(|increment| increment.counter.key())
        .collect::<Vec<_>>();
    let args = increments
        .iter()
        .flat_map(|increment| [python_arg(increment.amount), ttl_arg(increment.ttl)])
        .collect();
    Pending::new(eval(batch, &TOKEN_INCREMENT, &keys, args), floats)
}

/// `RedisBatch`'s `_Increment`: `INCRBYFLOAT`, then `EXPIRE` when the increment carries a TTL.
pub(super) fn counter_increment(batch: &RedisBatch, increment: &Increment) -> Declared<f64> {
    batch.incr_by_float(
        &increment.counter.key(),
        increment.amount.into(),
        increment.ttl,
    )
}

fn eval(
    batch: &RedisBatch,
    script: &Script,
    keys: &[String],
    args: Vec<String>,
) -> Declared<Value> {
    let keys = keys.iter().map(String::as_str).collect::<Vec<_>>();
    batch.eval(script, &keys, args)
}

/// The bytes redis-py sends for a Python `int` or `float` argument.
fn python_arg(amount: Number) -> String {
    match amount {
        Number::Int(value) => value.to_string(),
        Number::Float(value) => float_repr(value),
    }
}

/// `ttl or 0`: the scripts read a zero TTL as "leave the expiry alone".
fn ttl_arg(ttl: Option<Duration>) -> String {
    ttl.map_or(0, |ttl| ttl.as_secs()).to_string()
}

fn items(reply: Value) -> Result<Vec<Value>, Error> {
    match reply {
        Value::Array(items) => Ok(items),
        _ => Err(Error::Reply("is not a list")),
    }
}

fn integer(value: &Value) -> Result<i64, Error> {
    match value {
        Value::Int(value) => Ok(*value),
        _ => Err(Error::Reply("holds a non-integer where a count belongs")),
    }
}

fn text(value: &Value) -> Result<Option<String>, Error> {
    match value {
        Value::Nil => Ok(None),
        Value::BulkString(bytes) => String::from_utf8(bytes.clone())
            .map(Some)
            .map_err(|_| Error::Reply("holds a value that is not UTF-8")),
        Value::SimpleString(text) => Ok(Some(text.clone())),
        Value::Int(value) => Ok(Some(value.to_string())),
        _ => Err(Error::Reply("holds a nested value")),
    }
}

fn float(reply: Value) -> Result<f64, Error> {
    text(&reply)?
        .and_then(|text| text.parse().ok())
        .ok_or(Error::Reply("is not a float"))
}

fn index(raw: &Value) -> Result<usize, Error> {
    usize::try_from(integer(raw)? - 1).map_err(|_| Error::Reply("names an index below 1"))
}

fn window_counts(reply: Value) -> Result<Vec<WindowCount>, Error> {
    items(reply)?
        .chunks(2)
        .map(|pair| match pair {
            [window_start, count] => Ok(WindowCount {
                window_start: text(window_start)?
                    .ok_or(Error::Reply("holds a nil window start"))?,
                count: integer(count)?,
            }),
            _ => Err(Error::Reply("has an odd length")),
        })
        .collect()
}

fn optional_texts(reply: Value) -> Result<Vec<Option<String>>, Error> {
    items(reply)?.iter().map(text).collect()
}

fn integers(reply: Value) -> Result<Vec<i64>, Error> {
    items(reply)?.iter().map(integer).collect()
}

fn floats(reply: Value) -> Result<Vec<f64>, Error> {
    items(reply)?.into_iter().map(float).collect()
}

fn atomic_reply(reply: Value) -> Result<AtomicReply, Error> {
    let items = items(reply)?;
    match items.split_first() {
        Some((Value::Int(0), rest)) => rest
            .chunks(2)
            .map(|pair| match pair {
                [count, window_start] => Ok(Applied {
                    count: integer(count)?,
                    window_start: integer(window_start)?,
                }),
                _ => Err(Error::Reply("has an odd length")),
            })
            .collect::<Result<_, _>>()
            .map(AtomicReply::Applied),
        Some((Value::Int(1), [index_value, current, limit])) => Ok(AtomicReply::OverLimit {
            index: index(index_value)?,
            current: integer(current)?,
            limit: integer(limit)?,
        }),
        _ => Err(Error::Reply("has no status")),
    }
}

fn guarded_reply(reply: Value) -> Result<GuardedReply, Error> {
    match items(reply)?.as_slice() {
        [applied, count] => Ok(GuardedReply {
            applied: integer(applied)? == 1,
            count: integer(count)?,
        }),
        _ => Err(Error::Reply("is not an (applied, count) pair")),
    }
}

fn acquire_reply(reply: Value) -> Result<AcquireReply, Error> {
    let items = items(reply)?;
    match items.split_first() {
        Some((Value::Int(0), rest)) => rest
            .iter()
            .map(integer)
            .collect::<Result<_, _>>()
            .map(AcquireReply::Acquired),
        Some((Value::Int(1), [index_value, in_flight, limit])) => Ok(AcquireReply::OverLimit {
            index: index(index_value)?,
            in_flight: integer(in_flight)?,
            limit: integer(limit)?,
        }),
        _ => Err(Error::Reply("has no status")),
    }
}
