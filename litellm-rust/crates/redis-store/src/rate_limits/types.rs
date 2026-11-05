use std::time::Duration;

use litellm_python_compat::Value;

use super::keys::{Counter, Entry, Scope};
use crate::{DecodeError, FromValue, Number, codec::python_value, value};

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

const SCOPE: &str = "a rate-limit scope is (descriptor_key, descriptor_value)";
const COUNTER: &str =
    "a rate-limit counter is (descriptor_key, descriptor_value, \"requests\" | \"tokens\")";
const ENTRY: &str = "a rate-limit entry is (key_suffix, descriptor_key, descriptor_value)";
const INCREMENT: &str = "a rate-limit increment is (counter, amount, ttl_seconds | None)";
const ATOMIC: &str =
    "an atomic counter is (counter, limit, increment, ttl_seconds, window_size_seconds)";
const GAUGE_LIMIT: &str = "a gauge is (scope, limit)";

/// `(descriptor_key, descriptor_value)`.
impl FromValue for Scope {
    fn from_value(raw: &Value) -> Result<Self, DecodeError> {
        let [descriptor, value] = value::fields(raw, SCOPE)?;
        Ok(Self {
            descriptor: value::text(descriptor, SCOPE)?,
            value: value::text(value, SCOPE)?,
        })
    }
}

/// `(descriptor_key, descriptor_value, kind)`, the kind spelled as its key suffix.
impl FromValue for Counter {
    fn from_value(raw: &Value) -> Result<Self, DecodeError> {
        let [descriptor, value, kind] = value::fields(raw, COUNTER)?;
        Ok(Self {
            scope: Scope {
                descriptor: value::text(descriptor, COUNTER)?,
                value: value::text(value, COUNTER)?,
            },
            kind: value::text(kind, COUNTER)?
                .parse()
                .map_err(|_| DecodeError::Shape(COUNTER))?,
        })
    }
}

/// `(key_suffix, descriptor_key, descriptor_value)`, read by `Entry::new`.
impl FromValue for Entry {
    fn from_value(raw: &Value) -> Result<Self, DecodeError> {
        let [suffix, descriptor, value] = value::fields(raw, ENTRY)?;
        let scope = Scope {
            descriptor: value::text(descriptor, ENTRY)?,
            value: value::text(value, ENTRY)?,
        };
        Self::new(&value::text(suffix, ENTRY)?, scope).map_err(|_| DecodeError::Shape(ENTRY))
    }
}

/// `(counter, amount, ttl_seconds | None)`.
impl FromValue for Increment {
    fn from_value(raw: &Value) -> Result<Self, DecodeError> {
        let [counter, amount, ttl] = value::fields(raw, INCREMENT)?;
        Ok(Self {
            counter: Counter::from_value(counter)?,
            amount: value::number(amount, INCREMENT)?,
            ttl: match ttl {
                Value::None => None,
                ttl => Some(Duration::from_secs(
                    u64::try_from(value::integer(ttl, INCREMENT)?)
                        .map_err(|_| DecodeError::Shape(INCREMENT))?,
                )),
            },
        })
    }
}

/// `(counter, limit, increment, ttl_seconds, window_size_seconds)`.
impl FromValue for AtomicCounter {
    fn from_value(raw: &Value) -> Result<Self, DecodeError> {
        let [counter, limit, increment, ttl_seconds, window_size_seconds] =
            value::fields(raw, ATOMIC)?;
        Ok(Self {
            counter: Counter::from_value(counter)?,
            limit: value::integer(limit, ATOMIC)?,
            increment: value::integer(increment, ATOMIC)?,
            ttl_seconds: value::integer(ttl_seconds, ATOMIC)?,
            window_size_seconds: value::integer(window_size_seconds, ATOMIC)?,
        })
    }
}

/// `(scope, limit)`.
impl FromValue for Gauge {
    fn from_value(raw: &Value) -> Result<Self, DecodeError> {
        let [scope, limit] = value::fields(raw, GAUGE_LIMIT)?;
        Ok(Self {
            scope: Scope::from_value(scope)?,
            limit: value::integer(limit, GAUGE_LIMIT)?,
        })
    }
}

/// What the memory tier holds: counters and mirrored gauge counts, window starts, and the per-slot
/// registry a gauge keeps while Redis is unreachable.
#[derive(Clone, Debug, PartialEq)]
pub enum Local {
    Number(Number),
    Text(String),
    Slots(Vec<(String, f64)>),
}

/// A `RedisPipelineIncrementOperation`.
#[derive(Clone, Debug, PartialEq)]
pub struct Increment {
    pub counter: Counter,
    pub amount: Number,
    pub ttl: Option<Duration>,
}

pub(super) fn decode(raw: &[u8]) -> Result<Local, DecodeError> {
    match python_value(raw)? {
        Value::Str(text) => Ok(Local::Text(text)),
        value => Number::from_value(&value)
            .map(Local::Number)
            .ok_or(DecodeError::Shape(
                "a rate-limit value is a number or a string",
            )),
    }
}
