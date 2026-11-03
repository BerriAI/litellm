use std::time::Duration;

use litellm_python_compat::repr::float_repr;
use redis::{Cmd, RedisWrite, ToRedisArgs, ToSingleRedisArg};

/// How an increment treats the key's TTL, the two ways `RedisCache` increments with one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expiry {
    /// `async_increment(refresh_ttl=True)` and every pipelined increment: `EXPIRE` after each write.
    Refresh(Duration),
    /// `async_increment(refresh_ttl=False)`: read the `TTL` and `EXPIRE` only a key that has none.
    KeepExisting(Duration),
}

/// A number written the way redis-py encodes it: an `int` with `str()`, a `float` with `repr()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Amount {
    Int(i64),
    Float(f64),
}

impl ToRedisArgs for Amount {
    fn write_redis_args<W: ?Sized + RedisWrite>(&self, out: &mut W) {
        match self {
            Self::Int(value) => out.write_arg(value.to_string().as_bytes()),
            Self::Float(value) => out.write_arg(float_repr(*value).as_bytes()),
        }
    }
}

impl ToSingleRedisArg for Amount {}

pub(crate) fn ping() -> Cmd {
    redis::cmd("PING")
}

pub(crate) fn publish(channel: &str, payload: &[u8]) -> Cmd {
    redis::cmd("PUBLISH").arg(channel).arg(payload).to_owned()
}

pub(crate) fn get(key: String) -> Cmd {
    redis::cmd("GET").arg(key).to_owned()
}

pub(crate) fn mget(keys: Vec<String>) -> Cmd {
    redis::cmd("MGET").arg(keys).to_owned()
}

// redis-py's `set(ex=...)` takes whole seconds, as `int(ttl)` or a `timedelta`'s `int(total_seconds())`
pub(crate) fn set(key: String, value: impl ToSingleRedisArg, ttl: Duration) -> Cmd {
    redis::cmd("SET")
        .arg(key)
        .arg(value)
        .arg("EX")
        .arg(ttl.as_secs())
        .to_owned()
}

pub(crate) fn set_nx(key: String, value: impl ToSingleRedisArg, ttl: Duration) -> Cmd {
    redis::cmd("SET")
        .arg(key)
        .arg(value)
        .arg("EX")
        .arg(ttl.as_secs())
        .arg("NX")
        .to_owned()
}

pub(crate) fn del(keys: Vec<String>) -> Cmd {
    redis::cmd("DEL").arg(keys).to_owned()
}

pub(crate) fn expire(key: String, ttl: Duration) -> Cmd {
    redis::cmd("EXPIRE").arg(key).arg(ttl.as_secs()).to_owned()
}

pub(crate) fn ttl(key: String) -> Cmd {
    redis::cmd("TTL").arg(key).to_owned()
}

pub(crate) fn incr_by(key: String, delta: i64) -> Cmd {
    redis::cmd("INCRBY").arg(key).arg(delta).to_owned()
}

pub(crate) fn incr_by_float(key: String, amount: Amount) -> Cmd {
    redis::cmd("INCRBYFLOAT").arg(key).arg(amount).to_owned()
}
