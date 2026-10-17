//! State shared with the gateway's other replicas, and with Python proxies, through the
//! coordination Redis the Python proxy uses.
//!
//! Nothing here sits on the path of a call that succeeds: cooldowns are written when a
//! deployment cools down and read by a background poll, `allowed_fails` counts are only
//! touched by a failed attempt, and key cache invalidations arrive by subscription. While
//! Redis is unreachable each replica keeps its own state, as it does without Redis.

use std::{
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use litellm_config::{Config, Object, Value};
use litellm_gateway_auth::CachedKeyLookup;
use litellm_gateway_inference::ModelRouter;
use litellm_router::{CountFuture, Failure, SharedHealth};
use litellm_secrets::source::SecretSource;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use tokio_util::sync::CancellationToken;

use crate::Error;

/// The Python proxy's channel for auth cache invalidations (`auth_cache_invalidation_pubsub`).
const AUTH_CACHE_INVALIDATION_CHANNEL: &str = "litellm_proxy.auth_cache_invalidation";
/// How often cooldowns written by other replicas are read, as Python's
/// `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS`.
const DEFAULT_COOLDOWN_READ_INTERVAL: Duration = Duration::from_secs(1);
/// Upper bound on the `allowed_fails` round trip a failed attempt waits for before it counts
/// this replica's failures alone.
const COUNT_TIMEOUT: Duration = Duration::from_millis(250);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const RECONNECT_MIN: Duration = Duration::from_secs(1);
const RECONNECT_MAX: Duration = Duration::from_secs(30);
const MGET_CHUNK: usize = 500;

/// Where the coordination Redis is, resolved as the Python proxy resolves it.
#[derive(Clone, PartialEq, Eq)]
pub struct RedisSettings {
    url: String,
    namespace: Option<String>,
    cooldown_read_interval: Duration,
}

impl std::fmt::Debug for RedisSettings {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RedisSettings")
            .field("namespace", &self.namespace)
            .finish_non_exhaustive()
    }
}

impl RedisSettings {
    /// A Redis at `url`, with keys and channels under `namespace` as the proxy's `namespace`
    /// cache param puts them.
    pub fn new(url: impl Into<String>, namespace: Option<String>) -> Self {
        Self {
            url: url.into(),
            namespace,
            cooldown_read_interval: read_interval(),
        }
    }

    /// How often cooldowns other replicas wrote are read; one second by default, or
    /// `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS` as for the Python router.
    pub fn with_cooldown_read_interval(self, interval: Duration) -> Self {
        Self {
            cooldown_read_interval: interval,
            ..self
        }
    }

    /// `general_settings.coordination_redis`, else the response cache's Redis
    /// (`litellm_settings.cache_params` with `type: redis` when `cache` is on), else the
    /// `REDIS_*` environment variables. `None` when none names a Redis.
    pub async fn resolve(
        config: &Config,
        secrets: &dyn SecretSource,
    ) -> Result<Option<Self>, Error> {
        if let Some(block) = &config.general_settings.coordination_redis {
            let settings = Self::from_params(block, "general_settings.coordination_redis", secrets)
                .await?
                .ok_or(Error::Redis(
                    "general_settings.coordination_redis needs a host or url".into(),
                ))?;
            return Ok(Some(settings));
        }
        let litellm = &config.litellm_settings;
        if litellm.cache == Some(true)
            && let Some(params) = &litellm.cache_params
            && params.get("type").and_then(Value::as_str) == Some("redis")
            && let Some(settings) =
                Self::from_params(params, "litellm_settings.cache_params", secrets).await?
        {
            return Ok(Some(settings));
        }
        Self::from_environment(secrets).await
    }

    async fn from_params(
        params: &Object,
        setting: &str,
        secrets: &dyn SecretSource,
    ) -> Result<Option<Self>, Error> {
        for unsupported in [
            "startup_nodes",
            "redis_startup_nodes",
            "sentinel_nodes",
            "service_name",
        ] {
            if params
                .get(unsupported)
                .is_some_and(|value| !value.is_null())
            {
                return Err(Error::Redis(format!(
                    "{setting}.{unsupported}: Redis Cluster and Sentinel are not supported by this gateway yet"
                )));
            }
        }
        let mut fields = Fields::default();
        for (name, slot) in [
            ("url", &mut fields.url),
            ("host", &mut fields.host),
            ("port", &mut fields.port),
            ("username", &mut fields.username),
            ("password", &mut fields.password),
            ("ssl", &mut fields.ssl),
            ("db", &mut fields.db),
        ] {
            *slot = match params.get(name) {
                Some(value) => scalar(value, secrets).await?,
                None => None,
            };
        }
        let namespace = match params.get("namespace") {
            Some(value) => scalar(value, secrets).await?,
            None => None,
        };
        Ok(fields.url().map(|url| Self::new(url, namespace)))
    }

    async fn from_environment(secrets: &dyn SecretSource) -> Result<Option<Self>, Error> {
        for unsupported in ["REDIS_CLUSTER_NODES", "REDIS_SENTINEL_NODES"] {
            if secrets.get_secret_str(unsupported).await?.is_some() {
                return Err(Error::Redis(format!(
                    "{unsupported}: Redis Cluster and Sentinel are not supported by this gateway yet"
                )));
            }
        }
        let read = async |name: &str| -> Result<Option<String>, Error> {
            Ok(secrets
                .get_secret_str(name)
                .await?
                .map(|value| value.expose().to_owned())
                .filter(|value| !value.trim().is_empty()))
        };
        let fields = Fields {
            url: read("REDIS_URL").await?,
            host: read("REDIS_HOST").await?,
            port: read("REDIS_PORT").await?,
            username: read("REDIS_USERNAME").await?,
            password: read("REDIS_PASSWORD").await?,
            ssl: read("REDIS_SSL").await?,
            db: read("REDIS_DB").await?,
        };
        Ok(fields.url().map(|url| Self::new(url, None)))
    }

    fn key(&self, key: &str) -> String {
        match &self.namespace {
            Some(namespace) => format!("{namespace}:{key}"),
            None => key.to_owned(),
        }
    }
}

#[derive(Default)]
struct Fields {
    url: Option<String>,
    host: Option<String>,
    port: Option<String>,
    username: Option<String>,
    password: Option<String>,
    ssl: Option<String>,
    db: Option<String>,
}

impl Fields {
    /// `url` when given, else one built from `host`; `None` when neither is set.
    fn url(self) -> Option<String> {
        if let Some(url) = self.url {
            return Some(url);
        }
        let host = self.host?;
        let scheme = match self.ssl.as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("true" | "1" | "yes") => "rediss",
            _ => "redis",
        };
        let credentials = match (self.username, self.password) {
            (None, None) => String::new(),
            (username, password) => format!(
                "{}:{}@",
                encode(username.as_deref().unwrap_or_default()),
                encode(password.as_deref().unwrap_or_default())
            ),
        };
        let port = self.port.unwrap_or_else(|| "6379".into());
        let db = self.db.map(|db| format!("/{db}")).unwrap_or_default();
        Some(format!("{scheme}://{credentials}{host}:{port}{db}"))
    }
}

/// Percent-encodes a URL userinfo part.
fn encode(part: &str) -> String {
    part.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                char::from(byte).to_string()
            }
            byte => format!("%{byte:02X}"),
        })
        .collect()
}

/// A config scalar as text, resolving an `os.environ/` reference.
async fn scalar(value: &Value, secrets: &dyn SecretSource) -> Result<Option<String>, Error> {
    let text = match value {
        Value::Null => return Ok(None),
        Value::String(text) => text.clone(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        _ => return Err(Error::Redis("Redis settings must be scalars".into())),
    };
    let text = match text.strip_prefix("os.environ/") {
        Some(name) => match secrets.get_secret_str(name).await? {
            Some(value) => value.expose().to_owned(),
            None => return Err(Error::Redis(format!("{name} is not set"))),
        },
        None => text,
    };
    Ok(Some(text).filter(|text| !text.trim().is_empty()))
}

/// The coordination Redis: a client and, once it has connected, a multiplexed connection that
/// reconnects on its own.
pub struct Coordination {
    settings: RedisSettings,
    client: redis::Client,
    connection: OnceLock<ConnectionManager>,
    subscribed: AtomicBool,
}

impl Coordination {
    pub fn new(settings: RedisSettings) -> Result<Arc<Self>, Error> {
        let client = redis::Client::open(settings.url.as_str())
            .map_err(|error| Error::Redis(format!("invalid coordination Redis: {error}")))?;
        Ok(Arc::new(Self {
            settings,
            client,
            connection: OnceLock::new(),
            subscribed: AtomicBool::new(false),
        }))
    }

    /// The shared connection, once Redis has answered.
    pub(crate) fn connection(&self) -> Option<ConnectionManager> {
        self.connection.get().cloned()
    }

    /// `key` under the configured namespace, as the proxy prefixes its keys.
    pub(crate) fn namespaced(&self, key: &str) -> String {
        self.settings.key(key)
    }

    /// Whether the connection shared state is written and read through is up yet.
    pub fn connected(&self) -> bool {
        self.connection.get().is_some()
    }

    /// Whether key cache invalidations are being followed right now.
    pub fn subscribed(&self) -> bool {
        self.subscribed.load(Ordering::Relaxed)
    }

    /// Connects in the background, retrying until Redis answers, then starts the cooldown poll
    /// and, with `keys`, the key cache invalidation subscriber. Returns at once.
    pub fn start(
        self: &Arc<Self>,
        router: Arc<dyn RouterHandle>,
        keys: Option<Arc<CachedKeyLookup>>,
        shutdown: CancellationToken,
    ) {
        let coordination = self.clone();
        let cooldowns = shutdown.clone();
        tokio::spawn(async move {
            let connected = tokio::select! {
                connected = coordination.connect() => connected,
                () = cooldowns.cancelled() => return,
            };
            let _ = coordination.connection.set(connected);
            tracing::info!("coordination Redis connected: sharing deployment cooldowns");
            tokio::select! {
                () = coordination.poll_cooldowns(router) => (),
                () = cooldowns.cancelled() => (),
            }
        });
        if let Some(keys) = keys {
            let coordination = self.clone();
            tokio::spawn(async move {
                tokio::select! {
                    () = coordination.subscribe(&keys) => (),
                    () = shutdown.cancelled() => (),
                }
            });
        }
    }

    async fn connect(&self) -> ConnectionManager {
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(COMMAND_TIMEOUT))
            .set_response_timeout(Some(COMMAND_TIMEOUT));
        let mut backoff = RECONNECT_MIN;
        loop {
            match self
                .client
                .get_connection_manager_with_config(config.clone())
                .await
            {
                Ok(connection) => return connection,
                Err(error) => {
                    tracing::warn!(%error, retry_secs = backoff.as_secs(), "coordination Redis unreachable; replicas keep their own router health until it answers");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(RECONNECT_MAX);
                }
            }
        }
    }

    /// Reads every deployment's cooldown key each interval and applies the active ones.
    async fn poll_cooldowns(&self, router: Arc<dyn RouterHandle>) {
        let router = router.router();
        let keys: Vec<String> = router
            .deployment_ids()
            .map(|id| self.settings.key(&cooldown_key(id)))
            .collect();
        let mut interval = tokio::time::interval(self.settings.cooldown_read_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let Some(mut connection) = self.connection() else {
                continue;
            };
            for (chunk_index, chunk) in keys.chunks(MGET_CHUNK).enumerate() {
                let values: Vec<Option<String>> = match redis::cmd("MGET")
                    .arg(chunk)
                    .query_async(&mut connection)
                    .await
                {
                    Ok(values) => values,
                    Err(error) => {
                        tracing::debug!(%error, "reading shared cooldowns failed");
                        break;
                    }
                };
                let now = epoch_seconds();
                for (offset, value) in values.into_iter().enumerate() {
                    if let Some(remaining) =
                        value.as_deref().and_then(|value| remaining(value, now))
                    {
                        router.cool_down(chunk_index * MGET_CHUNK + offset, remaining);
                    }
                }
            }
        }
    }

    /// Follows the proxy's auth cache invalidations, dropping what each names from `keys`.
    /// Every (re)subscription first drops all fresh answers, since invalidations published
    /// while unsubscribed were missed.
    async fn subscribe(&self, keys: &CachedKeyLookup) {
        let channel = self.settings.key(AUTH_CACHE_INVALIDATION_CHANNEL);
        let mut backoff = RECONNECT_MIN;
        loop {
            match self.client.get_async_pubsub().await {
                Ok(mut pubsub) => match pubsub.subscribe(&channel).await {
                    Ok(()) => {
                        backoff = RECONNECT_MIN;
                        keys.invalidate_all();
                        self.subscribed.store(true, Ordering::Relaxed);
                        tracing::info!(%channel, "following proxy virtual key invalidations");
                        let mut messages = pubsub.into_on_message();
                        while let Some(message) = messages.next().await {
                            if let Some(cache_key) = invalidated(message.get_payload_bytes()) {
                                keys.invalidate(&cache_key).await;
                            }
                        }
                        self.subscribed.store(false, Ordering::Relaxed);
                        tracing::warn!(
                            "virtual key invalidation subscription dropped; resubscribing"
                        );
                    }
                    Err(error) => {
                        tracing::warn!(%error, "subscribing to virtual key invalidations failed")
                    }
                },
                Err(error) => {
                    tracing::warn!(%error, retry_secs = backoff.as_secs(), "coordination Redis unreachable; virtual keys refresh on their TTL until it answers")
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(RECONNECT_MAX);
        }
    }
}

/// What the cooldown poll needs from the gateway: its router.
pub trait RouterHandle: Send + Sync + 'static {
    fn router(&self) -> &ModelRouter;
}

impl RouterHandle for litellm_gateway_inference::Gateway {
    fn router(&self) -> &ModelRouter {
        &self.models
    }
}

/// Shares cooldowns and `allowed_fails` counts through the coordination Redis, under the keys
/// and in the format the Python router uses, so replicas and Python routers that name
/// deployments by the same `model_info.id` see each other's.
pub struct RedisHealth(pub Arc<Coordination>);

impl SharedHealth for RedisHealth {
    fn cooled(&self, deployment: &str, cooldown: Duration, failure: Failure) {
        let Some(mut connection) = self.0.connection() else {
            return;
        };
        let key = self.0.settings.key(&cooldown_key(deployment));
        let value = serde_json::json!({
            "exception_received": format!("litellm-gateway: {failure:?}"),
            "status_code": match failure {
                Failure::RateLimited => "429",
                Failure::Rejected => "401",
                Failure::Unavailable => "503",
            },
            "timestamp": epoch_seconds(),
            "cooldown_time": cooldown.as_secs_f64(),
        })
        .to_string();
        let milliseconds = u64::try_from(cooldown.as_millis())
            .unwrap_or(u64::MAX)
            .max(1);
        tokio::spawn(async move {
            let result: redis::RedisResult<()> = redis::cmd("SET")
                .arg(&key)
                .arg(value)
                .arg("PX")
                .arg(milliseconds)
                .query_async(&mut connection)
                .await;
            if let Err(error) = result {
                tracing::warn!(%error, "sharing a deployment cooldown failed");
            }
        });
    }

    fn count_failure<'a>(&'a self, deployment: &'a str, window: Duration) -> CountFuture<'a> {
        Box::pin(async move {
            let mut connection = self.0.connection()?;
            let key = self
                .0
                .settings
                .key(&format!("deployment:{deployment}:allowed_fails"));
            let milliseconds = u64::try_from(window.as_millis()).unwrap_or(u64::MAX).max(1);
            let mut pipeline = redis::pipe();
            pipeline
                .cmd("SET")
                .arg(&key)
                .arg(0)
                .arg("PX")
                .arg(milliseconds)
                .arg("NX")
                .ignore()
                .cmd("INCR")
                .arg(&key);
            let count = pipeline.query_async::<(u64,)>(&mut connection);
            match tokio::time::timeout(COUNT_TIMEOUT, count).await {
                Ok(Ok((count,))) => Some(u32::try_from(count).unwrap_or(u32::MAX)),
                Ok(Err(error)) => {
                    tracing::debug!(%error, "shared allowed_fails count failed");
                    None
                }
                Err(_) => None,
            }
        })
    }
}

/// Python's `CooldownCache.get_cooldown_cache_key`.
fn cooldown_key(deployment: &str) -> String {
    format!("deployment:{deployment}:cooldown")
}

fn read_interval() -> Duration {
    std::env::var("DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS")
        .ok()
        .and_then(|seconds| seconds.trim().parse::<f64>().ok())
        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
        .filter(|interval| !interval.is_zero())
        .unwrap_or(DEFAULT_COOLDOWN_READ_INTERVAL)
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// How much of a stored cooldown is left at `now`, or `None` when it is over or unreadable.
/// Reads both this gateway's JSON and the Python dict repr the Python router stores.
fn remaining(value: &str, now: f64) -> Option<Duration> {
    let timestamp = number_after(value, "timestamp")?;
    let cooldown_time = number_after(value, "cooldown_time")?;
    Duration::try_from_secs_f64(timestamp + cooldown_time - now)
        .ok()
        .filter(|remaining| !remaining.is_zero())
}

/// The number following the quoted `field` and its colon in a flat JSON or Python dict.
fn number_after(text: &str, field: &str) -> Option<f64> {
    ["\"", "'"].into_iter().find_map(|quote| {
        let start = text.find(&format!("{quote}{field}{quote}"))? + field.len() + 2;
        let rest = text[start..].trim_start().strip_prefix(':')?.trim_start();
        let end = rest
            .find(|character: char| !(character.is_ascii_digit() || "+-.eE".contains(character)))
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    })
}

/// The cache key an invalidation message drops; `None` for a malformed message or one that
/// sets a value (spend counters), which keys do not cache.
fn invalidated(payload: &[u8]) -> Option<String> {
    let message: serde_json::Value = serde_json::from_slice(payload).ok()?;
    if message
        .get("new_value")
        .is_some_and(|value| !value.is_null())
    {
        return None;
    }
    message.get("cache_key")?.as_str().map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use futures_util::future::BoxFuture;
    use litellm_auth_types::SecretValue;
    use rstest::rstest;

    use super::*;

    struct Secrets(HashMap<&'static str, &'static str>);

    impl SecretSource for Secrets {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
            Box::pin(async move { Ok(self.0.get(name).map(|value| SecretValue::new(*value))) })
        }
    }

    async fn resolve(
        yaml: &str,
        environment: &[(&'static str, &'static str)],
    ) -> Result<Option<(String, Option<String>)>, String> {
        let config = Config::from_yaml(&format!("model_list: []\n{yaml}")).unwrap();
        RedisSettings::resolve(&config, &Secrets(environment.iter().copied().collect()))
            .await
            .map(|settings| settings.map(|settings| (settings.url, settings.namespace)))
            .map_err(|error| error.to_string())
    }

    #[rstest]
    #[case::none("", &[], Ok(None))]
    #[case::environment_host(
        "",
        &[("REDIS_HOST", "redis.internal"), ("REDIS_PASSWORD", "p@ss")],
        Ok(Some(("redis://:p%40ss@redis.internal:6379".into(), None)))
    )]
    #[case::environment_url("", &[("REDIS_URL", "rediss://r:6380")], Ok(Some(("rediss://r:6380".into(), None))))]
    #[case::coordination_block_wins(
        "general_settings:\n  coordination_redis: {host: coord, port: 6390, ssl: true, namespace: prod}\n",
        &[("REDIS_HOST", "env")],
        Ok(Some(("rediss://coord:6390".into(), Some("prod".into()))))
    )]
    #[case::coordination_reference(
        "general_settings:\n  coordination_redis: {host: os.environ/COORD_HOST, password: os.environ/COORD_PASSWORD}\n",
        &[("COORD_HOST", "coord"), ("COORD_PASSWORD", "secret")],
        Ok(Some(("redis://:secret@coord:6379".into(), None)))
    )]
    #[case::response_cache_redis(
        "litellm_settings:\n  cache: true\n  cache_params: {type: redis, host: cache, port: '6381', namespace: ns}\n",
        &[],
        Ok(Some(("redis://cache:6381".into(), Some("ns".into()))))
    )]
    #[case::response_cache_off(
        "litellm_settings:\n  cache_params: {type: redis, host: cache}\n",
        &[("REDIS_HOST", "env")],
        Ok(Some(("redis://env:6379".into(), None)))
    )]
    #[case::cluster_refused(
        "",
        &[("REDIS_CLUSTER_NODES", "[]")],
        Err("coordination Redis: REDIS_CLUSTER_NODES: Redis Cluster and Sentinel are not supported by this gateway yet".into())
    )]
    #[case::block_without_target(
        "general_settings:\n  coordination_redis: {port: 6379}\n",
        &[],
        Err("coordination Redis: general_settings.coordination_redis needs a host or url".into())
    )]
    #[tokio::test]
    async fn resolves_the_coordination_redis_like_the_proxy(
        #[case] yaml: &str,
        #[case] environment: &[(&'static str, &'static str)],
        #[case] expected: Result<Option<(String, Option<String>)>, String>,
    ) {
        assert_eq!(resolve(yaml, environment).await, expected);
    }

    #[rstest]
    #[case::gateway_json(r#"{"exception_received": "x", "status_code": "429", "timestamp": 100.0, "cooldown_time": 5.0}"#, 102.0, Some(3_000))]
    #[case::python_repr(
        "{'exception_received': 'RateLimitError', 'status_code': '429', 'timestamp': 1.0e2, 'cooldown_time': 5}",
        104.5,
        Some(500)
    )]
    #[case::over(r#"{"timestamp": 100.0, "cooldown_time": 5.0}"#, 105.0, None)]
    #[case::unreadable("garbage", 0.0, None)]
    fn reads_cooldowns_written_by_either_gateway(
        #[case] value: &str,
        #[case] now: f64,
        #[case] expected_ms: Option<u128>,
    ) {
        assert_eq!(
            remaining(value, now).map(|remaining| remaining.as_millis()),
            expected_ms
        );
    }

    #[rstest]
    #[case::key(br#"{"cache_key": "abc"}"#, Some("abc"))]
    #[case::team(br#"{"cache_key": "team_id:t"}"#, Some("team_id:t"))]
    #[case::spend_counter(
        br#"{"cache_key": "spend:key:abc", "new_value": 1.5, "ttl": 60}"#,
        None
    )]
    #[case::malformed(b"not json", None)]
    fn reads_proxy_invalidation_messages(#[case] payload: &[u8], #[case] expected: Option<&str>) {
        assert_eq!(invalidated(payload).as_deref(), expected);
    }
}
