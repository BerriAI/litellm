use std::time::Duration;

use litellm_redis_client::commands;
use litellm_redis_store::{
    FromValue, IntoValue,
    auth_cache::{AuthCache, AuthObject, Settings},
};
use litellm_redis_testing::{RedisDeployment, Topology};

use super::{connect, remote};

const TTL: Duration = Duration::from_secs(60);

/// Writes `entry` under `read` as Python's writer does, reads it through the auth cache as `E`
/// and writes that back under `rewritten`. Returns the typed entry and the bytes Redis then holds
/// under `rewritten`.
pub async fn round_trip<E: FromValue + IntoValue + Clone>(
    topology: Topology,
    read: AuthObject,
    rewritten: AuthObject,
    entry: &str,
) -> (E, Option<String>) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let cache = AuthCache::new(Some(remote(connect(&deployment))), Settings::default());
    raw.send(commands::set(&read.key(), entry, TTL))
        .await
        .unwrap();

    let held = cache.get(None, &read).await.unwrap().unwrap();
    let typed = E::from_value(&held).unwrap();
    cache
        .set(None, &rewritten, typed.clone().into_value(), Some(TTL))
        .await
        .unwrap();

    let written = raw
        .send(commands::get::<String>(&rewritten.key()))
        .await
        .unwrap();
    (typed, written)
}
