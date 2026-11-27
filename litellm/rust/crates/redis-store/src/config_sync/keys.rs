const CHANNEL: &str = "litellm_proxy.config_change";

/// `config_sync_channel`: `RedisCache.namespace` prefixes the channel whenever it is not `None`,
/// since `PUBLISH` and `SUBSCRIBE` send it as given.
pub fn channel(namespace: Option<&str>) -> String {
    match namespace {
        Some(namespace) => format!("{namespace}:{CHANNEL}"),
        None => CHANNEL.to_owned(),
    }
}
