use std::{sync::LazyLock, time::Duration};

use litellm_redis_client::{RedisClient, Script};

use crate::{Error, as_strs};

static ACQUIRE: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
for i = 1, #KEYS do
    local limit = tonumber(ARGV[(i - 1) * 3 + 1])
    local slot_ttl = tonumber(ARGV[(i - 1) * 3 + 2])
    redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', now - slot_ttl)
    local in_flight = redis.call('ZCARD', KEYS[i])
    if in_flight + 1 > limit then
        return { 1, i, in_flight, limit }
    end
end
local results = { 0 }
for i = 1, #KEYS do
    local slot_ttl = tonumber(ARGV[(i - 1) * 3 + 2])
    local slot_id = ARGV[(i - 1) * 3 + 3]
    redis.call('ZADD', KEYS[i], now, slot_id)
    redis.call('EXPIRE', KEYS[i], slot_ttl)
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
",
    )
});

static RELEASE: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
local results = {}
for i = 1, #KEYS do
    redis.call('ZREM', KEYS[i], ARGV[i])
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
",
    )
});

static COUNT: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
local results = {}
for i = 1, #KEYS do
    redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', now - tonumber(ARGV[i]))
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
",
    )
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gauge {
    pub descriptor: Descriptor,
    pub limit: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Acquire {
    Admitted { in_flight: Vec<u64> },
    Rejected { index: usize, in_flight: u64, limit: u64 },
}

fn key(descriptor: &Descriptor) -> String {
    format!(
        "{{{}:{}}}:max_parallel_requests",
        descriptor.key, descriptor.value
    )
}

pub async fn acquire(
    client: &RedisClient,
    gauges: &[Gauge],
    slot_id: &str,
    slot_ttl: Duration,
) -> Result<Acquire, Error> {
    let keys: Vec<String> = gauges.iter().map(|gauge| key(&gauge.descriptor)).collect();
    let slot_ttl = slot_ttl.as_secs().to_string();
    let args: Vec<String> = gauges
        .iter()
        .flat_map(|gauge| [gauge.limit.to_string(), slot_ttl.clone(), slot_id.to_owned()])
        .collect();
    let reply: Vec<i64> = client.eval(&ACQUIRE, &as_strs(&keys), args).await?;
    match reply.as_slice() {
        [0, in_flight @ ..] => Ok(Acquire::Admitted {
            in_flight: counts(in_flight)?,
        }),
        [1, index, in_flight, limit] => Ok(Acquire::Rejected {
            index: usize::try_from(index - 1).map_err(Error::format)?,
            in_flight: count(*in_flight)?,
            limit: count(*limit)?,
        }),
        _ => Err(Error::unexpected(reply)),
    }
}

pub async fn release(
    client: &RedisClient,
    descriptors: &[Descriptor],
    slot_id: &str,
) -> Result<Vec<u64>, Error> {
    let keys: Vec<String> = descriptors.iter().map(key).collect();
    let args = vec![slot_id; descriptors.len()];
    let reply: Vec<i64> = client.eval(&RELEASE, &as_strs(&keys), args).await?;
    counts(&reply)
}

pub async fn count_in_flight(
    client: &RedisClient,
    descriptors: &[Descriptor],
    slot_ttl: Duration,
) -> Result<Vec<u64>, Error> {
    let keys: Vec<String> = descriptors.iter().map(key).collect();
    let args = vec![slot_ttl.as_secs(); descriptors.len()];
    let reply: Vec<i64> = client.eval(&COUNT, &as_strs(&keys), args).await?;
    counts(&reply)
}

fn count(value: i64) -> Result<u64, Error> {
    u64::try_from(value).map_err(Error::format)
}

fn counts(values: &[i64]) -> Result<Vec<u64>, Error> {
    values.iter().copied().map(count).collect()
}
