use std::{
    future::Future,
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{BaseCache, CacheCodec, CacheConnectionResult, Error, SemanticCacheContext};
use litellm_cache_response::CacheEntry;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub trait Embedder: Send + Sync + 'static {
    fn embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error>;

    fn async_embed(
        &self,
        prompt: &str,
        metadata: Option<&Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValkeySemanticConfig {
    pub similarity_threshold: f64,
    pub index_name: String,
}

pub const DEFAULT_INDEX_NAME: &str = "litellm_semantic_cache_index";

struct PooledConnection {
    connection: redis::Connection,
    failed: bool,
}

struct ConnectionManager(redis::Client);

impl r2d2::ManageConnection for ConnectionManager {
    type Connection = PooledConnection;
    type Error = redis::RedisError;

    fn connect(&self) -> Result<Self::Connection, Self::Error> {
        let connection = self.0.get_connection()?;
        Ok(PooledConnection {
            connection,
            failed: false,
        })
    }

    fn is_valid(&self, connection: &mut Self::Connection) -> Result<(), Self::Error> {
        redis::cmd("PING").query::<String>(&mut connection.connection)?;
        Ok(())
    }

    fn has_broken(&self, connection: &mut Self::Connection) -> bool {
        connection.failed || !redis::ConnectionLike::is_open(&connection.connection)
    }
}

enum Connections<C> {
    Pool(r2d2::Pool<ConnectionManager>),
    Fixed(Mutex<C>),
}

struct ConnectionRef<'a>(&'a mut dyn redis::ConnectionLike);

impl redis::ConnectionLike for ConnectionRef<'_> {
    fn req_packed_command(&mut self, cmd: &[u8]) -> redis::RedisResult<redis::Value> {
        self.0.req_packed_command(cmd)
    }

    fn req_packed_commands(
        &mut self,
        cmd: &[u8],
        offset: usize,
        count: usize,
    ) -> redis::RedisResult<Vec<redis::Value>> {
        self.0.req_packed_commands(cmd, offset, count)
    }

    fn get_db(&self) -> i64 {
        self.0.get_db()
    }

    fn supports_pipelining(&self) -> bool {
        self.0.supports_pipelining()
    }

    fn check_connection(&mut self) -> bool {
        self.0.check_connection()
    }

    fn is_open(&self) -> bool {
        self.0.is_open()
    }
}

impl<C> Connections<C>
where
    C: redis::ConnectionLike + Send + 'static,
{
    fn execute<T>(
        &self,
        operation: impl FnOnce(&mut ConnectionRef<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        match self {
            Self::Pool(pool) => {
                let mut pooled = pool.get().map_err(|_| Error::Unavailable)?;
                let result = operation(&mut ConnectionRef(&mut pooled.connection));
                pooled.failed = matches!(result, Err(Error::Unavailable));
                result
            }
            Self::Fixed(connection) => {
                let mut connection = connection.lock().map_err(|_| Error::Unavailable)?;
                operation(&mut ConnectionRef(&mut *connection))
            }
        }
    }
}

pub struct ValkeySemanticCache<
    E: Embedder,
    S: CacheCodec<Value = CacheEntry>,
    C = redis::Connection,
> {
    connections: Arc<Connections<C>>,
    embedder: E,
    codec: S,
    config: ValkeySemanticConfig,
    index_dimension: Arc<Mutex<Option<usize>>>,
}

impl<E, S> ValkeySemanticCache<E, S>
where
    E: Embedder,
    S: CacheCodec<Value = CacheEntry>,
{
    pub fn new(
        url: &str,
        embedder: E,
        codec: S,
        config: ValkeySemanticConfig,
    ) -> Result<Self, Error> {
        let client = redis::Client::open(url).map_err(|_| Error::Unavailable)?;
        let pool = r2d2::Pool::builder()
            .max_size(16)
            .min_idle(Some(0))
            .test_on_check_out(false)
            .build(ConnectionManager(client))
            .map_err(|_| Error::Unavailable)?;
        Ok(Self {
            connections: Arc::new(Connections::Pool(pool)),
            embedder,
            codec,
            config,
            index_dimension: Arc::new(Mutex::new(None)),
        })
    }
}

impl<E, S, C> ValkeySemanticCache<E, S, C>
where
    E: Embedder,
    S: CacheCodec<Value = CacheEntry>,
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn with_connection(
        connection: C,
        embedder: E,
        codec: S,
        config: ValkeySemanticConfig,
    ) -> Self {
        Self {
            connections: Arc::new(Connections::Fixed(Mutex::new(connection))),
            embedder,
            codec,
            config,
            index_dimension: Arc::new(Mutex::new(None)),
        }
    }

    pub fn similarity_threshold(&self) -> f64 {
        self.config.similarity_threshold
    }

    pub fn index_name(&self) -> &str {
        &self.config.index_name
    }

    fn key_prefix(&self) -> String {
        format!("{}:", self.config.index_name)
    }

    fn ensure_index(&self, dimension: usize) -> Result<(), Error> {
        ensure_index(
            &self.connections,
            &self.config.index_name,
            &self.key_prefix(),
            &self.index_dimension,
            dimension,
        )
    }
}

impl<E, S, C> BaseCache for ValkeySemanticCache<E, S, C>
where
    E: Embedder,
    S: CacheCodec<Value = CacheEntry>,
    C: redis::ConnectionLike + Send + 'static,
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
        let embedding = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        self.ensure_index(embedding.len())?;
        let scope = scope_tag(key);
        let document = format!("{}{}:{}", self.key_prefix(), scope, Uuid::new_v4());
        let response = self.codec.encode(&value)?;
        let vector = embedding_bytes(&embedding);
        let ttl = self.get_ttl(context);
        self.connections.execute(|connection| {
            let mut pipeline = redis::pipe();
            pipeline
                .cmd("HSET")
                .arg(&document)
                .arg("litellm_cache_key")
                .arg(&scope)
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(response)
                .arg("embedding")
                .arg(vector)
                .ignore();
            if let Some(ttl) = ttl {
                pipeline
                    .cmd("EXPIRE")
                    .arg(&document)
                    .arg(ttl.as_secs())
                    .ignore();
            }
            pipeline
                .query::<()>(connection)
                .map_err(|_| Error::Unavailable)
        })
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        let Some(prompt) = prompt_from_context(context) else {
            return Ok(None);
        };
        let embedding = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        self.ensure_index(embedding.len())?;
        let scope = scope_tag(key);
        let query =
            format!("(@litellm_cache_key:{{{scope}}})=>[KNN 1 @embedding $vec AS vector_distance]");
        let vector = embedding_bytes(&embedding);
        let response = self.connections.execute(|connection| {
            redis::cmd("FT.SEARCH")
                .arg(&self.config.index_name)
                .arg(query)
                .arg("PARAMS")
                .arg(2)
                .arg("vec")
                .arg(vector)
                .arg("RETURN")
                .arg(2)
                .arg("response")
                .arg("vector_distance")
                .arg("DIALECT")
                .arg(2)
                .query::<redis::Value>(connection)
                .map_err(|_| Error::Unavailable)
        })?;
        let Some(fields) = search_fields(response)? else {
            return Ok(None);
        };
        let response = fields
            .iter()
            .find_map(|(name, value)| (name == "response").then(|| value.clone()))
            .ok_or(Error::InvalidEntry)?;
        let distance = fields
            .iter()
            .find_map(|(name, value)| (name == "vector_distance").then(|| value.clone()))
            .ok_or(Error::InvalidEntry)?;
        let distance = parse_f64(&distance)?;
        if 1.0 - distance < self.config.similarity_threshold {
            return Ok(None);
        }
        self.codec.decode(&response).map(Some)
    }

    fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: Self::Context,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        let key = key.to_owned();
        let prompt = prompt_from_context(&context);
        let metadata = context.metadata.clone();
        async move {
            let Some(prompt) = prompt else {
                return Ok(());
            };
            let embedding = self
                .embedder
                .async_embed(&prompt, metadata.as_ref())
                .await?;
            let connections = Arc::clone(&self.connections);
            let config = self.config.clone();
            let index_dimension = Arc::clone(&self.index_dimension);
            let response = self.codec.encode(&value)?;
            let vector = embedding_bytes(&embedding);
            let prefix = format!("{}:", config.index_name);
            let scope = scope_tag(&key);
            let document = format!("{prefix}{scope}:{}", Uuid::new_v4());
            let ttl = context.ttl;
            tokio::task::spawn_blocking(move || {
                ensure_index(
                    &connections,
                    &config.index_name,
                    &prefix,
                    &index_dimension,
                    embedding.len(),
                )?;
                connections.execute(|connection| {
                    let mut pipeline = redis::pipe();
                    pipeline
                        .cmd("HSET")
                        .arg(&document)
                        .arg("litellm_cache_key")
                        .arg(&scope)
                        .arg("prompt")
                        .arg(prompt)
                        .arg("response")
                        .arg(response)
                        .arg("embedding")
                        .arg(vector)
                        .ignore();
                    if let Some(ttl) = ttl {
                        pipeline
                            .cmd("EXPIRE")
                            .arg(&document)
                            .arg(ttl.as_secs())
                            .ignore();
                    }
                    pipeline
                        .query::<()>(connection)
                        .map_err(|_| Error::Unavailable)
                })
            })
            .await
            .map_err(|_| Error::Unavailable)?
        }
    }

    fn async_get_cache(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> impl Future<Output = Result<Option<Self::Value>, Error>> + Send {
        let key = key.to_owned();
        let prompt = prompt_from_context(context);
        let metadata = context.metadata.clone();
        async move {
            let Some(prompt) = prompt else {
                return Ok(None);
            };
            let embedding = self
                .embedder
                .async_embed(&prompt, metadata.as_ref())
                .await?;
            let connections = Arc::clone(&self.connections);
            let config = self.config.clone();
            let index_dimension = Arc::clone(&self.index_dimension);
            let threshold = config.similarity_threshold;
            tokio::task::spawn_blocking(move || {
                let prefix = format!("{}:", config.index_name);
                ensure_index(
                    &connections,
                    &config.index_name,
                    &prefix,
                    &index_dimension,
                    embedding.len(),
                )?;
                let scope = scope_tag(&key);
                let query = format!(
                    "(@litellm_cache_key:{{{scope}}})=>[KNN 1 @embedding $vec AS vector_distance]"
                );
                let vector = embedding_bytes(&embedding);
                let response = connections.execute(|connection| {
                    redis::cmd("FT.SEARCH")
                        .arg(&config.index_name)
                        .arg(query)
                        .arg("PARAMS")
                        .arg(2)
                        .arg("vec")
                        .arg(vector)
                        .arg("RETURN")
                        .arg(2)
                        .arg("response")
                        .arg("vector_distance")
                        .arg("DIALECT")
                        .arg(2)
                        .query::<redis::Value>(connection)
                        .map_err(|_| Error::Unavailable)
                })?;
                let Some(fields) = search_fields(response)? else {
                    return Ok(None);
                };
                let response = fields
                    .iter()
                    .find_map(|(name, value)| (name == "response").then(|| value.clone()))
                    .ok_or(Error::InvalidEntry)?;
                let distance = fields
                    .iter()
                    .find_map(|(name, value)| (name == "vector_distance").then(|| value.clone()))
                    .ok_or(Error::InvalidEntry)?;
                let distance = parse_f64(&distance)?;
                if 1.0 - distance < threshold {
                    return Ok(None);
                }
                Ok(Some(response))
            })
            .await
            .map_err(|_| Error::Unavailable)?
            .and_then(|response| response.map(|bytes| self.codec.decode(&bytes)).transpose())
        }
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        Err(Error::UnsupportedOperation)
    }
}

pub fn prompt_from_context(context: &SemanticCacheContext) -> Option<String> {
    if let Some(Value::Array(messages)) = context.messages.as_ref()
        && !messages.is_empty()
    {
        return Some(
            messages
                .iter()
                .filter_map(Value::as_object)
                .map(message_text)
                .collect(),
        );
    }
    let input = context.input.as_ref()?;
    let mut parts = Vec::new();
    collect_input_text(input, &mut parts);
    let prompt = parts.join("\n").trim().to_owned();
    (!prompt.is_empty()).then_some(prompt)
}

fn message_text(message: &serde_json::Map<String, Value>) -> String {
    let content = match message.get("content") {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(Value::as_object)
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .filter(|text| !text.is_empty())
            .collect(),
        _ => String::new(),
    };
    format!(
        "{content}{}",
        search_results_text(message.get("search_results"))
    )
}

fn search_results_text(value: Option<&Value>) -> String {
    let Some(Value::Array(results)) = value else {
        return String::new();
    };
    results
        .iter()
        .filter_map(Value::as_object)
        .map(|result| {
            let source = result.get("source").and_then(Value::as_str).unwrap_or("");
            let title = result.get("title").and_then(Value::as_str).unwrap_or("");
            let content = result
                .get("content")
                .and_then(Value::as_array)
                .map(|blocks| {
                    blocks
                        .iter()
                        .filter_map(Value::as_object)
                        .filter_map(|block| block.get("text").and_then(Value::as_str))
                        .collect::<String>()
                })
                .unwrap_or_default();
            let citations = result
                .get("citations")
                .filter(|value| !value.is_null())
                .and_then(|value| serde_json::to_string(value).ok())
                .unwrap_or_default();
            format!("{source}{title}{content}{citations}")
        })
        .collect()
}

fn collect_input_text(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(value) => {
            let value = value.trim();
            if !value.is_empty() {
                parts.push(value.to_owned());
            }
        }
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_input_text(value, parts)),
        Value::Object(object) => {
            if let Some(content) = object.get("content").filter(|value| !value.is_null()) {
                collect_input_text(content, parts);
                return;
            }
            for key in ["text", "output", "input_text", "output_text"] {
                if let Some(Value::String(value)) = object.get(key) {
                    let value = value.trim();
                    if !value.is_empty() {
                        parts.push(value.to_owned());
                        return;
                    }
                }
            }
        }
        _ => {}
    }
}

fn scope_tag(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn embedding_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn ensure_index<C>(
    connections: &Connections<C>,
    index_name: &str,
    prefix: &str,
    index_dimension: &Mutex<Option<usize>>,
    dimension: usize,
) -> Result<(), Error>
where
    C: redis::ConnectionLike + Send + 'static,
{
    if index_dimension
        .lock()
        .map_err(|_| Error::Unavailable)?
        .is_some_and(|existing| existing == dimension)
    {
        return Ok(());
    }
    let create = connections.execute(|connection| {
        Ok(redis::cmd("FT.CREATE")
            .arg(index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg(1)
            .arg(prefix)
            .arg("SCHEMA")
            .arg("litellm_cache_key")
            .arg("TAG")
            .arg("embedding")
            .arg("VECTOR")
            .arg("HNSW")
            .arg(6)
            .arg("TYPE")
            .arg("FLOAT32")
            .arg("DIM")
            .arg(dimension)
            .arg("DISTANCE_METRIC")
            .arg("COSINE")
            .query::<String>(connection)
            .map(|_| ())
            .map_err(|error| error.to_string()))
    })?;
    if let Err(message) = create {
        if !message.to_ascii_lowercase().contains("already exists") {
            return Err(Error::Unavailable);
        }
        let info = connections.execute(|connection| {
            redis::cmd("FT.INFO")
                .arg(index_name)
                .query::<redis::Value>(connection)
                .map_err(|_| Error::Unavailable)
        })?;
        let existing = index_dimension_from_info(&info).ok_or(Error::Unavailable)?;
        if existing != dimension {
            return Err(Error::Unavailable);
        }
    }
    *index_dimension.lock().map_err(|_| Error::Unavailable)? = Some(dimension);
    Ok(())
}

fn index_dimension_from_info(value: &redis::Value) -> Option<usize> {
    let redis::Value::Array(values) = value else {
        return None;
    };
    let attributes = values.windows(2).find_map(|pair| {
        (value_text(&pair[0]).as_deref() == Some("attributes")).then_some(&pair[1])
    })?;
    let redis::Value::Array(fields) = attributes else {
        return None;
    };
    fields.iter().find_map(|field| {
        let redis::Value::Array(values) = field else {
            return None;
        };
        let flattened = values.iter().flat_map(|value| match value {
            redis::Value::Array(values) => values.as_slice(),
            _ => std::slice::from_ref(value),
        });
        let values = flattened.collect::<Vec<_>>();
        values.windows(2).find_map(|pair| {
            if value_text(pair[0]).as_deref() == Some("dimensions") {
                return value_text(pair[1]).and_then(|value| value.parse().ok());
            }
            None
        })
    })
}

type SearchFields = Vec<(String, Vec<u8>)>;

fn search_fields(value: redis::Value) -> Result<Option<SearchFields>, Error> {
    let redis::Value::Array(values) = value else {
        return Err(Error::InvalidEntry);
    };
    let total = parse_i64(values.first().ok_or(Error::InvalidEntry)?)?;
    if total <= 0 || values.len() < 3 {
        return Ok(None);
    }
    let redis::Value::Array(fields) = &values[2] else {
        return Err(Error::InvalidEntry);
    };
    let (pairs, remainder) = fields.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(Error::InvalidEntry);
    }
    let pairs = pairs
        .iter()
        .map(|pair| {
            Ok((
                value_text(&pair[0]).ok_or(Error::InvalidEntry)?,
                value_bytes(&pair[1])?,
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(Some(pairs))
}

fn parse_i64(value: &redis::Value) -> Result<i64, Error> {
    value_text(value)
        .ok_or(Error::InvalidEntry)?
        .parse()
        .map_err(|_| Error::InvalidEntry)
}

fn parse_f64(value: &[u8]) -> Result<f64, Error> {
    std::str::from_utf8(value)
        .map_err(|_| Error::InvalidEntry)?
        .parse()
        .map_err(|_| Error::InvalidEntry)
}

fn value_text(value: &redis::Value) -> Option<String> {
    match value {
        redis::Value::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
        redis::Value::SimpleString(value) => Some(value.clone()),
        redis::Value::Int(value) => Some(value.to_string()),
        _ => None,
    }
}

fn value_bytes(value: &redis::Value) -> Result<Vec<u8>, Error> {
    match value {
        redis::Value::BulkString(bytes) => Ok(bytes.clone()),
        redis::Value::SimpleString(value) => Ok(value.as_bytes().to_vec()),
        redis::Value::Int(value) => Ok(value.to_string().into_bytes()),
        _ => Err(Error::InvalidEntry),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use litellm_cache::BaseCache;
    use litellm_cache_response::ResponseCacheCodec;
    use redis_test::MockRedisConnection;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{
        Embedder, ValkeySemanticCache, ValkeySemanticConfig, index_dimension_from_info,
        prompt_from_context, scope_tag,
    };

    #[derive(Clone)]
    struct FixedEmbedder {
        vector: Vec<f32>,
        calls: EmbedderCalls,
    }

    type EmbedderCalls = Arc<Mutex<Vec<(String, Option<Value>)>>>;

    impl Embedder for FixedEmbedder {
        fn embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, super::Error> {
            self.calls
                .lock()
                .unwrap()
                .push((prompt.into(), metadata.cloned()));
            Ok(self.vector.clone())
        }

        async fn async_embed(
            &self,
            prompt: &str,
            metadata: Option<&Value>,
        ) -> Result<Vec<f32>, super::Error> {
            self.embed(prompt, metadata)
        }
    }

    fn context(
        messages: Option<Value>,
        input: Option<Value>,
    ) -> litellm_cache::SemanticCacheContext {
        litellm_cache::SemanticCacheContext {
            messages,
            input,
            ..Default::default()
        }
    }

    #[rstest]
    #[case(json!([{"content": "hello"}]), None, Some("hello"))]
    #[case(json!([{"content": [{"text": "hello"}, {"text": " world"}]}]), None, Some("hello world"))]
    #[case(json!([{"search_results": [{"source": "s", "title": "t", "content": [{"text": "c"}], "citations": ["x"]}]}]), None, Some(r#"stc["x"]"#))]
    #[case(Value::Array(vec![]), Some(json!(" hello ")), Some("hello"))]
    #[case(Value::Array(vec![]), Some(json!([{"content": "first"}, {"text": "second"}])), Some("first\nsecond"))]
    #[case(Value::Array(vec![]), Some(json!("   ")), None)]
    fn prompt_shapes(
        #[case] messages: Value,
        #[case] input: Option<Value>,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            prompt_from_context(&context(Some(messages), input)),
            expected.map(str::to_owned)
        );
    }

    #[test]
    fn scope_tags_are_lowercase_sha256() {
        assert_eq!(
            scope_tag("key"),
            "2c70e12b7a0646f92279f427c7b38e7334d8e5389cff167a1dc30e73f826b683"
        );
    }

    #[test]
    fn existing_index_dimension_is_read_from_attributes() {
        let info = redis::Value::Array(vec![
            redis::Value::SimpleString("attributes".into()),
            redis::Value::Array(vec![redis::Value::Array(vec![
                redis::Value::SimpleString("identifier".into()),
                redis::Value::SimpleString("embedding".into()),
                redis::Value::Array(vec![
                    redis::Value::SimpleString("dimensions".into()),
                    redis::Value::SimpleString("2".into()),
                ]),
            ])]),
        ]);
        assert_eq!(index_dimension_from_info(&info), Some(2));
    }

    #[tokio::test]
    async fn unsupported_connection_test_is_reported() {
        let cache = ValkeySemanticCache::with_connection(
            MockRedisConnection::new([]).assert_all_commands_consumed(),
            FixedEmbedder {
                vector: vec![1.0, 0.0],
                calls: Arc::default(),
            },
            ResponseCacheCodec,
            ValkeySemanticConfig {
                similarity_threshold: 0.8,
                index_name: "test".into(),
            },
        );
        assert_eq!(
            cache.test_connection().await,
            Err(super::Error::UnsupportedOperation)
        );
    }

    #[test]
    fn missing_prompt_does_not_touch_redis() {
        let cache = ValkeySemanticCache::with_connection(
            MockRedisConnection::new([]).assert_all_commands_consumed(),
            FixedEmbedder {
                vector: vec![1.0, 0.0],
                calls: Arc::default(),
            },
            ResponseCacheCodec,
            ValkeySemanticConfig {
                similarity_threshold: 0.8,
                index_name: "test".into(),
            },
        );
        assert_eq!(cache.get_cache("key", &context(None, None)).unwrap(), None);
        assert_eq!(cache.get_ttl(&context(None, None)), None);
    }
}
