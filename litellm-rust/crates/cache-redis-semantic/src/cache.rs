use std::{
    future::Future,
    sync::{Arc, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{
    BaseCache, CacheCodec, CacheConnectionResult, CacheConnectionStatus, Error,
    SemanticCacheContext,
};
use litellm_cache_redis::{
    RedisTopology,
    connection::{ConnectionRef, Connections},
};
use litellm_cache_response::{CacheEntry, ResponseCacheCodec};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::prompt::prompt_from_context;

const CACHE_KEY_FIELD: &str = "litellm_cache_key";
const VECTOR_FIELD: &str = "prompt_vector";

pub trait Embedder: Send + Sync + 'static {
    fn embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error>;

    fn async_embed(
        &self,
        prompt: &str,
        metadata: Option<&Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send;
}

#[derive(Clone, Debug)]
pub struct RedisSemanticConfig {
    pub index_name: String,
    pub similarity_threshold: f32,
}

struct Inner {
    index_name: String,
    distance_threshold: f64,
    resolved_index: OnceLock<String>,
    codec: ResponseCacheCodec,
    clock: fn() -> f64,
}

impl Inner {
    fn new(config: RedisSemanticConfig) -> Self {
        Self {
            index_name: config.index_name,
            distance_threshold: 1.0 - f64::from(config.similarity_threshold),
            resolved_index: OnceLock::new(),
            codec: ResponseCacheCodec,
            clock: timestamp,
        }
    }

    fn ensure_index(
        &self,
        connection: &mut ConnectionRef<'_>,
        dims: usize,
    ) -> Result<String, Error> {
        if let Some(name) = self.resolved_index.get() {
            return Ok(name.clone());
        }
        let name = match index_compatible(connection, &self.index_name, dims)? {
            Some(true) => self.index_name.clone(),
            Some(false) => self.isolated_index(connection, dims)?,
            None => match create_index(connection, &self.index_name, dims) {
                Ok(()) => self.index_name.clone(),
                Err(_) => match index_compatible(connection, &self.index_name, dims)? {
                    Some(true) => self.index_name.clone(),
                    Some(false) => self.isolated_index(connection, dims)?,
                    None => return Err(Error::Unavailable),
                },
            },
        };
        let _ = self.resolved_index.set(name.clone());
        Ok(name)
    }

    fn isolated_index(
        &self,
        connection: &mut ConnectionRef<'_>,
        dims: usize,
    ) -> Result<String, Error> {
        let name = format!("{}_isolated", self.index_name);
        match index_compatible(connection, &name, dims)? {
            Some(true) => Ok(name),
            Some(false) => {
                redis::cmd("FT.DROPINDEX")
                    .arg(&name)
                    .query::<()>(connection)
                    .map_err(|_| Error::Unavailable)?;
                create_index(connection, &name, dims)?;
                Ok(name)
            }
            None => {
                create_index(connection, &name, dims)?;
                Ok(name)
            }
        }
    }

    fn store(
        &self,
        connection: &mut ConnectionRef<'_>,
        tag: &str,
        value: &CacheEntry,
        prompt: &str,
        vector: &[f32],
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let index = self.ensure_index(connection, vector.len())?;
        let entry_id = entry_id(prompt, tag);
        let hash_key = format!("{index}:{entry_id}");
        let response = self.codec.encode(value)?;
        redis::cmd("HSET")
            .arg(&hash_key)
            .arg("entry_id")
            .arg(&entry_id)
            .arg("prompt")
            .arg(prompt)
            .arg("response")
            .arg(response)
            .arg(VECTOR_FIELD)
            .arg(vector_buffer(vector))
            .arg("inserted_at")
            .arg(format!("{}", (self.clock)()))
            .arg("updated_at")
            .arg(format!("{}", (self.clock)()))
            .arg(CACHE_KEY_FIELD)
            .arg(tag)
            .query::<()>(connection)
            .map_err(|_| Error::Unavailable)?;
        if let Some(ttl) = ttl {
            redis::cmd("EXPIRE")
                .arg(&hash_key)
                .arg(ttl_seconds(ttl))
                .query::<()>(connection)
                .map_err(|_| Error::Unavailable)?;
        }
        Ok(())
    }

    fn lookup(
        &self,
        connection: &mut ConnectionRef<'_>,
        tag: &str,
        vector: &[f32],
    ) -> Result<Option<CacheEntry>, Error> {
        let index = self.ensure_index(connection, vector.len())?;
        let query = format!(
            "(@{CACHE_KEY_FIELD}:{{{}}})=>[KNN 1 @{VECTOR_FIELD} $vector AS vector_distance]",
            escape_tag(tag)
        );
        let result = redis::cmd("FT.SEARCH")
            .arg(&index)
            .arg(query)
            .arg("RETURN")
            .arg(8)
            .arg("entry_id")
            .arg("prompt")
            .arg("response")
            .arg("inserted_at")
            .arg("updated_at")
            .arg("metadata")
            .arg(CACHE_KEY_FIELD)
            .arg("vector_distance")
            .arg("SORTBY")
            .arg("vector_distance")
            .arg("ASC")
            .arg("DIALECT")
            .arg(2)
            .arg("LIMIT")
            .arg(0)
            .arg(1)
            .arg("PARAMS")
            .arg(2)
            .arg("vector")
            .arg(vector_buffer(vector))
            .query::<redis::Value>(connection)
            .map_err(|_| Error::Unavailable)?;
        let Some(fields) = first_document(&result) else {
            return Ok(None);
        };
        if string_field(fields, CACHE_KEY_FIELD).as_deref() != Some(tag) {
            return Ok(None);
        }
        if number_field(fields, "vector_distance")
            .is_none_or(|distance| distance > self.distance_threshold)
        {
            return Ok(None);
        }
        let Some(response) = bytes_field(fields, "response") else {
            return Ok(None);
        };
        self.codec.decode(&response).map(Some)
    }
}

pub struct RedisSemanticCache<E: Embedder, C = redis::Connection> {
    connections: Arc<Connections<C>>,
    embedder: E,
    inner: Arc<Inner>,
}

impl<E: Embedder> RedisSemanticCache<E> {
    pub fn new(url: &str, embedder: E, config: RedisSemanticConfig) -> Result<Self, Error> {
        Ok(Self {
            connections: Arc::new(Connections::open(url, &RedisTopology::Standalone)?),
            embedder,
            inner: Arc::new(Inner::new(config)),
        })
    }
}

impl<E: Embedder, C: redis::ConnectionLike + Send + 'static> RedisSemanticCache<E, C> {
    pub fn with_connection(connection: C, embedder: E, config: RedisSemanticConfig) -> Self {
        Self {
            connections: Arc::new(Connections::fixed(connection)),
            embedder,
            inner: Arc::new(Inner::new(config)),
        }
    }

    pub fn with_clock(self, clock: fn() -> f64) -> Self {
        Self {
            inner: Arc::new(Inner {
                index_name: self.inner.index_name.clone(),
                distance_threshold: self.inner.distance_threshold,
                resolved_index: OnceLock::new(),
                codec: self.inner.codec,
                clock,
            }),
            ..self
        }
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    pub fn index_name(&self) -> &str {
        &self.inner.index_name
    }

    pub fn similarity_threshold(&self) -> f32 {
        (1.0 - self.inner.distance_threshold) as f32
    }

    fn tag<'a>(key: &'a str, context: &'a SemanticCacheContext) -> &'a str {
        context.scope.as_deref().unwrap_or(key)
    }
}

impl<E: Embedder, C: redis::ConnectionLike + Send + 'static> BaseCache
    for RedisSemanticCache<E, C>
{
    type Value = CacheEntry;
    type Context = SemanticCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &Self::Context,
    ) -> Result<(), Error> {
        let Some(prompt) = prompt_from_context(context) else {
            return Ok(());
        };
        let vector = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        let tag = Self::tag(key, context).to_string();
        self.connections.execute(|connection| {
            self.inner
                .store(connection, &tag, &value, &prompt, &vector, context.ttl)
        })
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        let Some(prompt) = prompt_from_context(context) else {
            return Ok(None);
        };
        let vector = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        let tag = Self::tag(key, context).to_string();
        self.connections
            .execute(|connection| self.inner.lookup(connection, &tag, &vector))
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: Self::Context,
    ) -> Result<(), Error> {
        let Some(prompt) = prompt_from_context(&context) else {
            return Ok(());
        };
        let vector = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let tag = Self::tag(key, &context).to_string();
        let inner = Arc::clone(&self.inner);
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            inner.store(connection, &tag, &value, &prompt, &vector, context.ttl)
        })
        .await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<Option<Self::Value>, Error> {
        let Some(prompt) = prompt_from_context(context) else {
            return Ok(None);
        };
        let vector = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let tag = Self::tag(key, context).to_string();
        let inner = Arc::clone(&self.inner);
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            inner.lookup(connection, &tag, &vector)
        })
        .await
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match Connections::run_blocking(Arc::clone(&self.connections), |connection| {
            Ok(match redis::cmd("PING").query::<String>(connection) {
                Ok(_) => CacheConnectionResult {
                    status: CacheConnectionStatus::Success,
                    message: "Redis cache connection test successful".into(),
                    error: None,
                },
                Err(error) => CacheConnectionResult {
                    status: CacheConnectionStatus::Failed,
                    message: format!("Redis connection failed: {error}"),
                    error: Some(error.to_string()),
                },
            })
        })
        .await
        {
            Ok(result) => Ok(result),
            Err(error) => Ok(CacheConnectionResult {
                status: CacheConnectionStatus::Failed,
                message: format!("Redis connection failed: {error}"),
                error: Some(error.to_string()),
            }),
        }
    }
}

fn timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

fn entry_id(prompt: &str, tag: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(prompt.as_bytes());
    digest.update(CACHE_KEY_FIELD.as_bytes());
    digest.update(tag.as_bytes());
    format!("{:x}", digest.finalize())
}

fn vector_buffer(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

fn escape_tag(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| {
            if matches!(
                ch,
                ',' | '.'
                    | '<'
                    | '>'
                    | '{'
                    | '}'
                    | '['
                    | ']'
                    | '\\'
                    | '"'
                    | '\''
                    | ':'
                    | ';'
                    | '!'
                    | '@'
                    | '#'
                    | '$'
                    | '%'
                    | '^'
                    | '&'
                    | '*'
                    | '('
                    | ')'
                    | '-'
                    | '+'
                    | '='
                    | '~'
                    | '|'
                    | '/'
                    | ' '
                    | '?'
            ) {
                vec!['\\', ch]
            } else {
                vec![ch]
            }
        })
        .collect()
}

fn create_index(connection: &mut ConnectionRef<'_>, name: &str, dims: usize) -> Result<(), Error> {
    redis::cmd("FT.CREATE")
        .arg(name)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(name)
        .arg("SCORE")
        .arg(1.0)
        .arg("SCHEMA")
        .arg("prompt")
        .arg("TEXT")
        .arg("WEIGHT")
        .arg(1)
        .arg("response")
        .arg("TEXT")
        .arg("WEIGHT")
        .arg(1)
        .arg("inserted_at")
        .arg("NUMERIC")
        .arg("updated_at")
        .arg("NUMERIC")
        .arg(VECTOR_FIELD)
        .arg("VECTOR")
        .arg("FLAT")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dims)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .arg(CACHE_KEY_FIELD)
        .arg("TAG")
        .arg("SEPARATOR")
        .arg(",")
        .query::<()>(connection)
        .map_err(|_| Error::Unavailable)
}

fn index_compatible(
    connection: &mut ConnectionRef<'_>,
    name: &str,
    dims: usize,
) -> Result<Option<bool>, Error> {
    let info = match redis::cmd("FT.INFO")
        .arg(name)
        .query::<redis::Value>(connection)
    {
        Ok(info) => info,
        Err(error) if unknown_index(&error) => return Ok(None),
        Err(_) => return Err(Error::Unavailable),
    };
    Ok(Some(schema_compatible(&info, dims)))
}

fn unknown_index(error: &redis::RedisError) -> bool {
    let message = error.to_string().to_lowercase();
    message.contains("unknown") && message.contains("index")
}

fn schema_compatible(info: &redis::Value, dims: usize) -> bool {
    let redis::Value::Array(entries) = info else {
        return false;
    };
    let attributes = entries
        .as_chunks::<2>()
        .0
        .iter()
        .find(|pair| string_value(&pair[0]).as_deref() == Some("attributes"))
        .map(|pair| &pair[1]);
    let Some(redis::Value::Array(attributes)) = attributes else {
        return false;
    };
    let fields = attributes
        .iter()
        .map(|attribute| {
            let redis::Value::Array(attribute) = attribute else {
                return (None, None, None, None, None);
            };
            let mut name = None;
            let mut field_type = None;
            let mut dim = None;
            let mut data_type = None;
            let mut distance_metric = None;
            for pair in attribute.as_chunks::<2>().0 {
                match string_value(&pair[0]).as_deref() {
                    Some("identifier") => name = string_value(&pair[1]),
                    Some("type") => field_type = string_value(&pair[1]),
                    Some("dim") => dim = number_value(&pair[1]),
                    Some("data_type") => data_type = string_value(&pair[1]),
                    Some("distance_metric") => distance_metric = string_value(&pair[1]),
                    _ => {}
                }
            }
            (name, field_type, dim, data_type, distance_metric)
        })
        .collect::<Vec<_>>();
    let has_field = |name: &str, field_type: &str| {
        fields
            .iter()
            .any(|(n, t, ..)| n.as_deref() == Some(name) && t.as_deref() == Some(field_type))
    };
    has_field("prompt", "TEXT")
        && has_field("response", "TEXT")
        && has_field("inserted_at", "NUMERIC")
        && has_field("updated_at", "NUMERIC")
        && has_field(CACHE_KEY_FIELD, "TAG")
        && fields.iter().any(|(n, t, d, data, metric)| {
            n.as_deref() == Some(VECTOR_FIELD)
                && t.as_deref() == Some("VECTOR")
                && *d == Some(dims as f64)
                && data
                    .as_deref()
                    .is_some_and(|data| data.eq_ignore_ascii_case("float32"))
                && metric
                    .as_deref()
                    .is_some_and(|metric| metric.eq_ignore_ascii_case("cosine"))
        })
}

fn string_value(value: &redis::Value) -> Option<String> {
    match value {
        redis::Value::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
        redis::Value::SimpleString(text) => Some(text.clone()),
        redis::Value::VerbatimString { text, .. } => Some(text.clone()),
        _ => None,
    }
}

fn number_value(value: &redis::Value) -> Option<f64> {
    match value {
        redis::Value::Int(number) => Some(*number as f64),
        redis::Value::Double(number) => Some(*number),
        _ => string_value(value).and_then(|text| text.parse().ok()),
    }
}

fn first_document(result: &redis::Value) -> Option<&[redis::Value]> {
    let redis::Value::Array(items) = result else {
        return None;
    };
    let [count, _document_id, fields, ..] = items.as_slice() else {
        return None;
    };
    if !matches!(count, redis::Value::Int(count) if *count > 0) {
        return None;
    }
    match fields {
        redis::Value::Array(fields) => Some(fields.as_slice()),
        _ => None,
    }
}

fn field_value<'a>(fields: &'a [redis::Value], name: &str) -> Option<&'a redis::Value> {
    fields
        .as_chunks::<2>()
        .0
        .iter()
        .find(|pair| string_value(&pair[0]).as_deref() == Some(name))
        .map(|pair| &pair[1])
}

fn string_field(fields: &[redis::Value], name: &str) -> Option<String> {
    field_value(fields, name).and_then(string_value)
}

fn number_field(fields: &[redis::Value], name: &str) -> Option<f64> {
    field_value(fields, name).and_then(number_value)
}

fn bytes_field(fields: &[redis::Value], name: &str) -> Option<Vec<u8>> {
    match field_value(fields, name)? {
        redis::Value::BulkString(bytes) => Some(bytes.clone()),
        redis::Value::SimpleString(text) => Some(text.clone().into_bytes()),
        _ => None,
    }
}

fn ttl_seconds(ttl: Duration) -> u64 {
    ttl.as_secs()
        .saturating_add(u64::from(ttl.subsec_nanos() > 0))
        .max(1)
}
