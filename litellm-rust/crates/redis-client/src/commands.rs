use std::time::Duration;

use redis::{Cmd, ToRedisArgs, ToSingleRedisArg};

pub(crate) fn ping() -> Cmd {
    redis::cmd("PING")
}

pub(crate) fn get(key: String) -> Cmd {
    redis::cmd("GET").arg(key).to_owned()
}

pub(crate) fn mget(keys: Vec<String>) -> Cmd {
    redis::cmd("MGET").arg(keys).to_owned()
}

pub(crate) fn set(key: String, value: impl ToSingleRedisArg, ttl: Duration) -> Cmd {
    redis::cmd("SET")
        .arg(key)
        .arg(value)
        .arg("PX")
        .arg(millis(ttl))
        .to_owned()
}

pub(crate) fn set_nx(key: String, value: impl ToSingleRedisArg, ttl: Duration) -> Cmd {
    redis::cmd("SET")
        .arg(key)
        .arg(value)
        .arg("NX")
        .arg("PX")
        .arg(millis(ttl))
        .to_owned()
}

pub(crate) fn del(keys: Vec<String>) -> Cmd {
    redis::cmd("DEL").arg(keys).to_owned()
}

pub(crate) fn expire(key: String, ttl: Duration) -> Cmd {
    redis::cmd("PEXPIRE").arg(key).arg(millis(ttl)).to_owned()
}

pub(crate) fn ttl(key: String) -> Cmd {
    redis::cmd("PTTL").arg(key).to_owned()
}

pub(crate) fn incr_by(key: String, delta: i64) -> Cmd {
    redis::cmd("INCRBY").arg(key).arg(delta).to_owned()
}

pub(crate) fn incr_by_float(key: String, delta: f64) -> Cmd {
    redis::cmd("INCRBYFLOAT").arg(key).arg(delta).to_owned()
}

pub(crate) fn rpush(key: String, values: impl ToRedisArgs) -> Cmd {
    redis::cmd("RPUSH").arg(key).arg(values).to_owned()
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}
