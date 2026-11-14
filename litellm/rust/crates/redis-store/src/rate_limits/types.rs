use std::time::Duration;

use litellm_python_compat::Value;

use super::keys::{Counter, Scope};
use crate::{DecodeError, FromValue, Number};

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

/// What the memory tier holds: counters and mirrored gauge counts, window starts, and the per-slot
/// registry a gauge keeps while Redis is unreachable.
#[derive(Clone, Debug, PartialEq)]
pub enum Local {
    Number(Number),
    Text(String),
    Slots(Vec<(String, f64)>),
}

impl From<Number> for Local {
    fn from(number: Number) -> Self {
        Self::Number(number)
    }
}

impl crate::memory::Counter for Local {
    fn as_number(&self) -> Option<Number> {
        match self {
            Self::Number(number) => Some(*number),
            Self::Text(_) | Self::Slots(_) => None,
        }
    }
}

/// A `RedisPipelineIncrementOperation`.
#[derive(Clone, Debug, PartialEq)]
pub struct Increment {
    pub counter: Counter,
    pub amount: Number,
    pub ttl: Option<Duration>,
}

/// A rate-limit value as `RedisCache._get_cache_logic` reads it.
impl FromValue for Local {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        match value {
            Value::Str(text) => Ok(Self::Text(text.clone())),
            value => Number::from_value(value)
                .map(Local::Number)
                .ok_or(DecodeError::Shape(
                    "a rate-limit value is a number or a string",
                )),
        }
    }
}
