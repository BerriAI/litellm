use std::sync::LazyLock;

use litellm_redis_connect::RedisHosts;

pub(crate) mod breaker;
pub(crate) mod client;
mod events;
pub(crate) mod store;

static HOSTS: LazyLock<RedisHosts> = LazyLock::new(RedisHosts::default);

/// The Redis servers this process talks to, shared by every `RedisCache` and native store in it.
pub(crate) fn hosts() -> &'static RedisHosts {
    &HOSTS
}
