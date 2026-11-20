const CHANNEL: &str = "litellm_proxy.auth_cache_invalidation";

/// `auth_cache_invalidation_channel`: `RedisCache.namespace` prefixes the channel whenever it is not
/// `None`, since `PUBLISH` and `SUBSCRIBE` send it as given.
pub fn channel(namespace: Option<&str>) -> String {
    match namespace {
        Some(namespace) => format!("{namespace}:{CHANNEL}"),
        None => CHANNEL.to_owned(),
    }
}
