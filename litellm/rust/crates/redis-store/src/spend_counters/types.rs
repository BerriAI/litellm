use litellm_python_compat::Value;

use super::keys::SpendCounter;
use crate::{DecodeError, FromValue, Number, codec};

/// The `spend:*` key Python names a counter by.
impl FromValue for SpendCounter {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        let Value::Str(key) = value else {
            return Err(DecodeError::Shape("a spend counter is a spend:* key"));
        };
        Self::parse(key).map_err(|_| DecodeError::Shape("a spend counter is a spend:* key"))
    }
}

/// Where a read was answered, named the way the Python store reports it. Redis is cross-pod truth,
/// including a clean miss; memory only holds this pod's writes and answers only when there is no
/// Redis or it failed.
#[derive(Clone, Copy, Debug, PartialEq, strum::IntoStaticStr)]
pub enum Reading {
    #[strum(serialize = "redis")]
    Redis(Option<Number>),
    #[strum(serialize = "memory")]
    Memory(Option<Number>),
}

impl Reading {
    pub fn value(self) -> Option<Number> {
        match self {
            Self::Redis(value) | Self::Memory(value) => value,
        }
    }
}

/// Who wins when a seed's `SET NX` lost and the winner's value is gone by the time it is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LostSeed {
    /// `SpendCounterReseed.coalesced`: answer with the seed without writing it.
    Answer,
    /// `SpendCounterReseed.coalesced_window`: `INCRBYFLOAT` the seed, keeping any TTL.
    Increment,
}

pub(super) fn decode(raw: Option<Vec<u8>>) -> Result<Option<Number>, DecodeError> {
    raw.as_deref()
        .map(|raw| codec::number(raw, "a spend counter is a number"))
        .transpose()
}
