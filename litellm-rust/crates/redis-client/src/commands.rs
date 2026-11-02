use std::{sync::LazyLock, time::Duration};

use redis::{Cmd, ToRedisArgs, ToSingleRedisArg};

use crate::Script;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expiry {
    Refresh(Duration),
    KeepExisting(Duration),
}

pub(crate) struct ScriptCall {
    pub(crate) script: &'static Script,
    pub(crate) keys: Vec<String>,
    pub(crate) args: Vec<Vec<u8>>,
}

macro_rules! with_expiry {
    ($write:literal) => {
        with_expiry!($write, "value")
    };
    ($write:literal, $reply:literal) => {
        concat!(
            "local value = ",
            $write,
            "\nlocal reply = ",
            $reply,
            "\nif ARGV[1] == 'refresh' or redis.call('PTTL', KEYS[1]) == -1 then\n",
            "  redis.call('PEXPIRE', KEYS[1], ARGV[2])\n",
            "end\n",
            "return reply"
        )
    };
}

static INCR_BY: LazyLock<Script> = LazyLock::new(|| {
    Script::new(concat!(
        "redis.call('GET', KEYS[1])\n",
        with_expiry!(
            "redis.call('INCRBY', KEYS[1], ARGV[3])",
            "redis.call('GET', KEYS[1])"
        )
    ))
});
static INCR_BY_FLOAT: LazyLock<Script> =
    LazyLock::new(|| Script::new(with_expiry!("redis.call('INCRBYFLOAT', KEYS[1], ARGV[3])")));
static RPUSH: LazyLock<Script> = LazyLock::new(|| {
    Script::new(with_expiry!(
        "redis.call('RPUSH', KEYS[1], unpack(ARGV, 3))"
    ))
});

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

pub(crate) fn incr_by(key: String, delta: i64, expiry: Expiry) -> ScriptCall {
    with_expiry_call(&INCR_BY, key, expiry, delta)
}

pub(crate) fn incr_by_float(key: String, delta: f64, expiry: Expiry) -> ScriptCall {
    with_expiry_call(&INCR_BY_FLOAT, key, expiry, delta)
}

pub(crate) fn rpush(key: String, values: impl ToRedisArgs, expiry: Expiry) -> ScriptCall {
    with_expiry_call(&RPUSH, key, expiry, values)
}

fn with_expiry_call(
    script: &'static Script,
    key: String,
    expiry: Expiry,
    payload: impl ToRedisArgs,
) -> ScriptCall {
    let (mode, ttl) = match expiry {
        Expiry::Refresh(ttl) => ("refresh", ttl),
        Expiry::KeepExisting(ttl) => ("keep", ttl),
    };
    let args = [
        mode.to_redis_args(),
        millis(ttl).to_redis_args(),
        payload.to_redis_args(),
    ]
    .into_iter()
    .flatten()
    .collect();
    ScriptCall {
        script,
        keys: vec![key],
        args,
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}
