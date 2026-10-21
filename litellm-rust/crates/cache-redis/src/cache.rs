use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use litellm_cache::{BatchEntry, CacheCodec, Error};

use crate::{
    connection::{ConnectionRef, Connections},
    topology::RedisTopology,
};

const DEFAULT_TTL: Duration = Duration::from_secs(600);

pub struct RedisCache<S, C = redis::Connection> {
    pub(crate) connections: Arc<Connections<C>>,
    pub(crate) default_ttl: Duration,
    pub(crate) codec: S,
    pub(crate) namespace: Option<String>,
    pub(crate) topology: RedisTopology,
    /// The server's major version, read from `INFO` once, like Python's `redis_version`.
    pub(crate) major_version: Arc<OnceLock<u32>>,
}

impl<S: CacheCodec> RedisCache<S> {
    pub fn new(url: &str, default_ttl: Option<Duration>, codec: S) -> Result<Self, Error> {
        Self::connect(url, &RedisTopology::Standalone, default_ttl, codec)
    }

    pub fn connect(
        url: &str,
        topology: &RedisTopology,
        default_ttl: Option<Duration>,
        codec: S,
    ) -> Result<Self, Error> {
        let connections = Connections::open(url, topology)?;
        Ok(Self {
            connections: Arc::new(connections),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            codec,
            namespace: None,
            topology: topology.clone(),
            major_version: Arc::default(),
        })
    }
}

impl<S, C> RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn with_connection(connection: C, default_ttl: Option<Duration>, codec: S) -> Self {
        Self {
            connections: Arc::new(Connections::fixed(connection)),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            codec,
            namespace: None,
            topology: RedisTopology::Standalone,
            major_version: Arc::default(),
        }
    }

    pub fn with_namespace(self, namespace: Option<String>) -> Self {
        Self {
            namespace: namespace.filter(|value| !value.is_empty()),
            ..self
        }
    }

    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    pub fn topology(&self) -> &RedisTopology {
        &self.topology
    }

    pub(crate) fn namespaced_key(&self, key: &str) -> String {
        namespaced_key(self.namespace.as_deref(), key)
    }

    pub(crate) fn namespaced_keys(&self, keys: &[String]) -> Vec<String> {
        keys.iter().map(|key| self.namespaced_key(key)).collect()
    }

    /// Whole seconds for `ttl`, falling back to the default TTL like Python's `get_ttl`.
    pub(crate) fn ttl_or_default(&self, ttl: Option<Duration>) -> u64 {
        ttl_seconds(ttl.unwrap_or(self.default_ttl))
    }

    pub(crate) fn execute<T>(
        &self,
        operation: impl FnOnce(&mut ConnectionRef<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.connections.execute(operation)
    }

    /// `_parse_redis_major_version`: the major version from `INFO`, or
    /// `DEFAULT_REDIS_MAJOR_VERSION` when `INFO` fails or its version does not parse. The first
    /// answer is kept, as Python reads `redis_version` once at construction.
    pub(crate) async fn major_version(&self) -> u32 {
        if let Some(version) = self.major_version.get() {
            return *version;
        }
        let info = self
            .run(|connection| connection.node_text(&redis::cmd("INFO")))
            .await;
        let version = info
            .ok()
            .and_then(|info| parse_major_version(&info))
            .unwrap_or_else(default_major_version);
        *self.major_version.get_or_init(|| version)
    }

    pub(crate) async fn run<T, F>(&self, operation: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&mut ConnectionRef<'_>) -> Result<T, Error> + Send + 'static,
    {
        Connections::run_blocking(Arc::clone(&self.connections), operation).await
    }

    pub(crate) fn namespaced_pattern(&self) -> Result<String, Error> {
        let namespace = self.namespace.as_ref().ok_or(Error::UnscopedFlush)?;
        let escaped: String = namespace
            .chars()
            .flat_map(|ch| {
                if matches!(ch, '*' | '?' | '[' | ']' | '\\') {
                    vec!['\\', ch]
                } else {
                    vec![ch]
                }
            })
            .collect();
        Ok(format!("{escaped}:*"))
    }

    pub(crate) fn decode_response(&self, value: redis::Value) -> Result<Option<S::Value>, Error> {
        match value {
            redis::Value::Nil => Ok(None),
            redis::Value::BulkString(bytes) => self.codec.decode(&bytes).map(Some),
            redis::Value::SimpleString(text) => self.codec.decode(text.as_bytes()).map(Some),
            _ => Err(Error::InvalidEntry),
        }
    }

    pub(crate) fn decode_batch_response(
        &self,
        value: redis::Value,
    ) -> Result<BatchEntry<S::Value>, Error> {
        match self.decode_response(value) {
            Ok(Some(value)) => Ok(BatchEntry::Hit(value)),
            Ok(None) => Ok(BatchEntry::Miss),
            Err(Error::InvalidEntry) => Ok(BatchEntry::Invalid),
            Err(error) => Err(error),
        }
    }
}

pub(crate) fn namespaced_key(namespace: Option<&str>, key: &str) -> String {
    match namespace {
        Some(namespace) if !key.starts_with(&format!("{namespace}:")) => {
            format!("{namespace}:{key}")
        }
        _ => key.into(),
    }
}

pub(crate) fn ttl_seconds(ttl: Duration) -> u64 {
    ttl.as_secs()
        .saturating_add(u64::from(ttl.subsec_nanos() > 0))
        .max(1)
}

fn parse_major_version(info: &str) -> Option<u32> {
    let version = info
        .lines()
        .find_map(|line| line.trim().strip_prefix("redis_version:"))?
        .trim();
    match version.split_once('.') {
        Some((major, _)) => major.parse().ok(),
        None => version.parse::<f64>().ok().map(|major| major as u32),
    }
}

fn default_major_version() -> u32 {
    std::env::var("DEFAULT_REDIS_MAJOR_VERSION")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(7)
}
