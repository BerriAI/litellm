use std::{sync::LazyLock, time::Duration};

use litellm_python_compat::json::dumps_str;
use litellm_redis_client::{RedisClient, Script};

use crate::Error;

static RELEASE: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"#,
    )
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Acquire {
    Acquired,
    HeldBy(String),
    Vacant,
}

fn key(cronjob_id: &str) -> String {
    format!("cronjob_lock:{cronjob_id}")
}

pub async fn acquire(
    client: &RedisClient,
    cronjob_id: &str,
    owner: &str,
    ttl: Duration,
) -> Result<Acquire, Error> {
    if client.set_nx(&key(cronjob_id), dumps_str(owner), ttl).await? {
        return Ok(Acquire::Acquired);
    }
    Ok(holder(client, cronjob_id)
        .await?
        .map_or(Acquire::Vacant, Acquire::HeldBy))
}

pub async fn holder(client: &RedisClient, cronjob_id: &str) -> Result<Option<String>, Error> {
    let Some(raw) = client.get::<Vec<u8>>(&key(cronjob_id)).await? else {
        return Ok(None);
    };
    serde_json::from_slice(&raw).map(Some).map_err(Error::format)
}

pub async fn release(client: &RedisClient, cronjob_id: &str, owner: &str) -> Result<bool, Error> {
    let deleted: i64 = client
        .eval(&RELEASE, &[&key(cronjob_id)], dumps_str(owner))
        .await?;
    Ok(deleted == 1)
}

pub async fn remove(client: &RedisClient, cronjob_id: &str) -> Result<bool, Error> {
    Ok(client.del(&[&key(cronjob_id)]).await? == 1)
}
