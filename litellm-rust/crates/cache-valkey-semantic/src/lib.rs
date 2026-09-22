use std::{
    future::Future,
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{BaseCache, CacheCodec, CacheConnectionResult, Error, SemanticCacheContext};
use litellm_cache_redis::{
    RedisTopology,
    connection::{ConnectionRef, Connections},
};
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

pub struct PreparedEmbedding(pub Vec<f32>);

impl Embedder for PreparedEmbedding {
    fn embed(&self, _prompt: &str, _metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        Ok(self.0.clone())
    }

    async fn async_embed(
        &self,
        _prompt: &str,
        _metadata: Option<&Value>,
    ) -> Result<Vec<f32>, Error> {
        Ok(self.0.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValkeySemanticConfig {
    pub similarity_threshold: f64,
    pub index_name: String,
}

pub const DEFAULT_INDEX_NAME: &str = "litellm_semantic_cache_index";

#[derive(Clone)]
struct IndexState {
    name: String,
    prefix: String,
    dimension: Arc<Mutex<Option<usize>>>,
    similarity_threshold: f64,
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
        Ok(Self {
            connections: Arc::new(Connections::open(url, &RedisTopology::Standalone)?),
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
            connections: Arc::new(Connections::fixed(connection)),
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

    fn index_state(&self) -> IndexState {
        IndexState {
            name: self.config.index_name.clone(),
            prefix: format!("{}:", self.config.index_name),
            dimension: Arc::clone(&self.index_dimension),
            similarity_threshold: self.config.similarity_threshold,
        }
    }
}

impl<E, S, C> ValkeySemanticCache<E, S, C>
where
    E: Embedder,
    S: CacheCodec<Value = CacheEntry> + Clone,
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn with_embedder<E2: Embedder>(&self, embedder: E2) -> ValkeySemanticCache<E2, S, C> {
        ValkeySemanticCache {
            connections: Arc::clone(&self.connections),
            embedder,
            codec: self.codec.clone(),
            config: self.config.clone(),
            index_dimension: Arc::clone(&self.index_dimension),
        }
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
        let scope = scope_tag(key);
        let response = self.codec.encode(&value)?;
        let vector = embedding_bytes(&embedding);
        let index = self.index_state();
        self.connections.execute(|connection| {
            write_document(
                connection,
                &index,
                &scope,
                &prompt,
                response,
                vector,
                self.get_ttl(context),
            )
        })
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        let Some(prompt) = prompt_from_context(context) else {
            return Ok(None);
        };
        let embedding = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        let scope = scope_tag(key);
        let vector = embedding_bytes(&embedding);
        let index = self.index_state();
        let response = self.connections.execute(|connection| {
            search_document(connection, &index, &scope, vector, embedding.len())
        })?;
        let Some(response) = response else {
            return Ok(None);
        };
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
            let index = self.index_state();
            let response = self.codec.encode(&value)?;
            let vector = embedding_bytes(&embedding);
            let scope = scope_tag(&key);
            let ttl = context.ttl;
            Connections::run_blocking(connections, move |connection| {
                write_document(connection, &index, &scope, &prompt, response, vector, ttl)
            })
            .await
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
            let index = self.index_state();
            Connections::run_blocking(connections, move |connection| {
                let scope = scope_tag(&key);
                let vector = embedding_bytes(&embedding);
                search_document(connection, &index, &scope, vector, embedding.len())
            })
            .await
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
        return messages
            .iter()
            .filter_map(Value::as_object)
            .map(message_text)
            .collect();
    }
    let input = context.input.as_ref()?;
    let mut parts = Vec::new();
    collect_input_text(input, &mut parts);
    let prompt = parts.join("\n").trim().to_owned();
    (!prompt.is_empty()).then_some(prompt)
}

fn message_text(message: &serde_json::Map<String, Value>) -> Option<String> {
    let content = match message.get("content") {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Array(parts)) => {
            let mut content = String::new();
            for part in parts {
                let part = part.as_object()?;
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    content.push_str(text);
                }
            }
            content
        }
        _ => String::new(),
    };
    Some(format!(
        "{content}{}",
        search_results_text(message.get("search_results"))
    ))
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

fn write_document(
    connection: &mut ConnectionRef<'_>,
    index: &IndexState,
    scope: &str,
    prompt: &str,
    response: Vec<u8>,
    vector: Vec<u8>,
    ttl: Option<Duration>,
) -> Result<(), Error> {
    let dimension = vector.len() / std::mem::size_of::<f32>();
    ensure_index(
        connection,
        &index.name,
        &index.prefix,
        &index.dimension,
        dimension,
    )?;
    let document = format!("{}{scope}:{}", index.prefix, Uuid::new_v4());
    let mut pipeline = redis::pipe();
    pipeline
        .cmd("HSET")
        .arg(&document)
        .arg("litellm_cache_key")
        .arg(scope)
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
}

fn search_document(
    connection: &mut ConnectionRef<'_>,
    index: &IndexState,
    scope: &str,
    vector: Vec<u8>,
    dimension: usize,
) -> Result<Option<Vec<u8>>, Error> {
    ensure_index(
        connection,
        &index.name,
        &index.prefix,
        &index.dimension,
        dimension,
    )?;
    let query =
        format!("(@litellm_cache_key:{{{scope}}})=>[KNN 1 @embedding $vec AS vector_distance]");
    let response = redis::cmd("FT.SEARCH")
        .arg(&index.name)
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
        .map_err(|_| Error::Unavailable)?;
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
    if 1.0 - distance < index.similarity_threshold {
        return Ok(None);
    }
    Ok(Some(response))
}

fn ensure_index(
    connection: &mut ConnectionRef<'_>,
    index_name: &str,
    prefix: &str,
    index_dimension: &Mutex<Option<usize>>,
    dimension: usize,
) -> Result<(), Error> {
    if index_dimension
        .lock()
        .map_err(|_| Error::Unavailable)?
        .is_some_and(|existing| existing == dimension)
    {
        return Ok(());
    }
    let create = redis::cmd("FT.CREATE")
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
        .map_err(|error| error.to_string());
    if let Err(message) = create {
        if !message.to_ascii_lowercase().contains("already exists") {
            return Err(Error::Unavailable);
        }
        let info = redis::cmd("FT.INFO")
            .arg(index_name)
            .query::<redis::Value>(connection)
            .map_err(|_| Error::Unavailable)?;
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
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use litellm_cache::{BaseCache, CacheCodec};
    use litellm_cache_response::{
        CacheEntry, CacheKeyInput, ResponseCache, ResponseCacheCodec, ResponseCacheRequest,
    };
    use redis_test::MockRedisConnection;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{
        Embedder, PreparedEmbedding, ValkeySemanticCache, ValkeySemanticConfig,
        index_dimension_from_info, prompt_from_context, scope_tag,
    };

    #[derive(Clone)]
    struct FixedEmbedder {
        vector: Vec<f32>,
        calls: EmbedderCalls,
    }

    type EmbedderCalls = Arc<Mutex<Vec<(String, Option<Value>)>>>;
    type RecordingCache =
        ValkeySemanticCache<FixedEmbedder, ResponseCacheCodec, RecordingConnection>;
    type RecordingSetup = (RecordingCache, Arc<Mutex<Vec<Vec<u8>>>>, EmbedderCalls);

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

    struct RecordingConnection {
        requests: Arc<Mutex<Vec<Vec<u8>>>>,
        replies: Mutex<VecDeque<redis::RedisResult<redis::Value>>>,
    }

    impl RecordingConnection {
        fn new(replies: impl IntoIterator<Item = redis::RedisResult<redis::Value>>) -> Self {
            Self {
                requests: Arc::default(),
                replies: Mutex::new(replies.into_iter().collect()),
            }
        }

        fn requests(&self) -> Arc<Mutex<Vec<Vec<u8>>>> {
            Arc::clone(&self.requests)
        }

        fn reply(&self) -> redis::RedisResult<redis::Value> {
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Ok(redis::Value::SimpleString("OK".into())))
        }
    }

    impl redis::ConnectionLike for RecordingConnection {
        fn req_packed_command(&mut self, command: &[u8]) -> redis::RedisResult<redis::Value> {
            self.requests.lock().unwrap().push(command.to_vec());
            self.reply()
        }

        fn req_packed_commands(
            &mut self,
            command: &[u8],
            _offset: usize,
            count: usize,
        ) -> redis::RedisResult<Vec<redis::Value>> {
            self.requests.lock().unwrap().push(command.to_vec());
            (0..count).map(|_| self.reply()).collect()
        }

        fn get_db(&self) -> i64 {
            0
        }

        fn check_connection(&mut self) -> bool {
            true
        }

        fn is_open(&self) -> bool {
            true
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
    #[case(json!([{"content": ["raw", {"text": "hello"}]}]), None, None)]
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

    #[tokio::test]
    async fn prepared_embedding_returns_its_vector_for_any_prompt() {
        let embedding = PreparedEmbedding(vec![1.0, 2.0]);
        assert_eq!(
            embedding
                .async_embed("different prompt", None)
                .await
                .unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn with_embedder_shares_index_state_and_connections() {
        let entry = CacheEntry {
            timestamp: Some(1.0),
            response: json!({"answer": "ok"}),
        };
        let encoded = ResponseCacheCodec.encode(&entry).unwrap();
        let cache = ValkeySemanticCache::with_connection(
            RecordingConnection::new([ok(), ok(), Ok(search_hit(encoded, "0.1"))]),
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
        cache
            .set_cache("key", entry.clone(), &semantic_context(None))
            .unwrap();
        let prepared = cache.with_embedder(PreparedEmbedding(vec![1.0, 0.0]));
        assert_eq!(
            prepared.get_cache("key", &semantic_context(None)).unwrap(),
            Some(entry)
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

    fn semantic_context(ttl: Option<Duration>) -> litellm_cache::SemanticCacheContext {
        litellm_cache::SemanticCacheContext {
            messages: Some(json!([{"role": "user", "content": "hello"}])),
            metadata: Some(json!({"source": "test"})),
            ttl,
            ..Default::default()
        }
    }

    fn cache_with_recording(
        replies: impl IntoIterator<Item = redis::RedisResult<redis::Value>>,
        vector: Vec<f32>,
        threshold: f64,
    ) -> RecordingSetup {
        let connection = RecordingConnection::new(replies);
        let requests = connection.requests();
        let calls: EmbedderCalls = Arc::default();
        let cache = ValkeySemanticCache::with_connection(
            connection,
            FixedEmbedder {
                vector,
                calls: Arc::clone(&calls),
            },
            ResponseCacheCodec,
            ValkeySemanticConfig {
                similarity_threshold: threshold,
                index_name: "test".into(),
            },
        );
        (cache, requests, calls)
    }

    fn ok() -> redis::RedisResult<redis::Value> {
        Ok(redis::Value::SimpleString("OK".into()))
    }

    fn already_exists() -> redis::RedisResult<redis::Value> {
        Err(redis::RedisError::from((
            redis::ErrorKind::Io,
            "already exists",
        )))
    }

    fn info_dimension(dimension: usize) -> redis::Value {
        redis::Value::Array(vec![
            redis::Value::SimpleString("attributes".into()),
            redis::Value::Array(vec![redis::Value::Array(vec![
                redis::Value::SimpleString("embedding".into()),
                redis::Value::Array(vec![
                    redis::Value::SimpleString("dimensions".into()),
                    redis::Value::Int(dimension as i64),
                ]),
            ])]),
        ])
    }

    fn search_hit(response: Vec<u8>, distance: &str) -> redis::Value {
        redis::Value::Array(vec![
            redis::Value::Int(1),
            redis::Value::BulkString(b"test:document".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"response".to_vec()),
                redis::Value::BulkString(response),
                redis::Value::BulkString(b"vector_distance".to_vec()),
                redis::Value::BulkString(distance.as_bytes().to_vec()),
            ]),
        ])
    }

    fn requests_text(requests: &Arc<Mutex<Vec<Vec<u8>>>>) -> String {
        requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| String::from_utf8_lossy(request))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn set_without_ttl_writes_hset_without_expire() {
        let (cache, requests, calls) = cache_with_recording([ok()], vec![1.0, 0.0], 0.8);
        cache
            .set_cache(
                "key",
                CacheEntry {
                    timestamp: None,
                    response: json!({"answer": "ok"}),
                },
                &semantic_context(None),
            )
            .unwrap();
        let text = requests_text(&requests);
        assert!(text.contains("FT.CREATE"));
        assert!(text.contains("HSET"));
        assert!(
            text.contains("test:2c70e12b7a0646f92279f427c7b38e7334d8e5389cff167a1dc30e73f826b683:")
        );
        assert!(!text.contains("EXPIRE"));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![("hello".into(), Some(json!({"source": "test"})))]
        );
    }

    #[test]
    fn set_with_ttl_truncates_expire_seconds() {
        let (cache, requests, _) = cache_with_recording([ok()], vec![1.0, 0.0], 0.8);
        cache
            .set_cache(
                "key",
                CacheEntry {
                    timestamp: None,
                    response: json!({"answer": "ok"}),
                },
                &semantic_context(Some(Duration::from_millis(1900))),
            )
            .unwrap();
        let text = requests_text(&requests);
        assert!(text.contains("EXPIRE"));
        assert!(text.contains("\r\n$1\r\n1\r\n"));
    }

    #[test]
    fn second_set_skips_create_after_dimension_is_cached() {
        let (cache, requests, _) = cache_with_recording([ok()], vec![1.0, 0.0], 0.8);
        let context = semantic_context(None);
        let entry = CacheEntry {
            timestamp: None,
            response: json!({"answer": "ok"}),
        };
        cache.set_cache("key", entry.clone(), &context).unwrap();
        cache.set_cache("key", entry, &context).unwrap();
        let text = requests_text(&requests);
        assert_eq!(text.matches("FT.CREATE").count(), 1);
        assert_eq!(text.matches("HSET").count(), 2);
    }

    #[test]
    fn existing_index_dimension_must_match_embedding() {
        let (cache, _, _) = cache_with_recording(
            [already_exists(), Ok(info_dimension(2))],
            vec![1.0, 0.0],
            0.8,
        );
        cache
            .set_cache(
                "key",
                CacheEntry {
                    timestamp: None,
                    response: json!({"answer": "ok"}),
                },
                &semantic_context(None),
            )
            .unwrap();

        let (cache, _, _) = cache_with_recording(
            [already_exists(), Ok(info_dimension(3))],
            vec![1.0, 0.0],
            0.8,
        );
        assert_eq!(
            cache.set_cache(
                "key",
                CacheEntry {
                    timestamp: None,
                    response: json!({"answer": "ok"}),
                },
                &semantic_context(None),
            ),
            Err(super::Error::Unavailable)
        );
    }

    #[test]
    fn get_applies_threshold_and_decodes_entry() {
        let entry = CacheEntry {
            timestamp: Some(1.0),
            response: json!({"answer": "ok"}),
        };
        let encoded = ResponseCacheCodec.encode(&entry).unwrap();
        let (cache, _, _) = cache_with_recording(
            [ok(), Ok(search_hit(encoded.clone(), "0.1"))],
            vec![1.0, 0.0],
            0.8,
        );
        assert_eq!(
            cache.get_cache("key", &semantic_context(None)).unwrap(),
            Some(entry)
        );

        let (cache, _, _) =
            cache_with_recording([ok(), Ok(search_hit(encoded, "0.5"))], vec![1.0, 0.0], 0.8);
        assert_eq!(
            cache.get_cache("key", &semantic_context(None)).unwrap(),
            None
        );
    }

    #[test]
    fn get_zero_docs_is_a_miss() {
        let (cache, _, _) = cache_with_recording(
            [ok(), Ok(redis::Value::Array(vec![redis::Value::Int(0)]))],
            vec![1.0, 0.0],
            0.8,
        );
        assert_eq!(
            cache.get_cache("key", &semantic_context(None)).unwrap(),
            None
        );
    }

    #[rstest]
    #[case(redis::Value::Array(vec![
        redis::Value::Int(1),
        redis::Value::BulkString(b"document".to_vec()),
        redis::Value::Array(vec![
            redis::Value::BulkString(b"vector_distance".to_vec()),
            redis::Value::BulkString(b"0.1".to_vec()),
        ]),
    ]))]
    #[case(redis::Value::Array(vec![
        redis::Value::Int(1),
        redis::Value::BulkString(b"document".to_vec()),
        redis::Value::Array(vec![
            redis::Value::BulkString(b"response".to_vec()),
            redis::Value::BulkString(b"not-json".to_vec()),
            redis::Value::BulkString(b"vector_distance".to_vec()),
            redis::Value::BulkString(b"abc".to_vec()),
        ]),
    ]))]
    fn malformed_entries_are_invalid(#[case] search: redis::Value) {
        let (cache, _, _) = cache_with_recording([ok(), Ok(search)], vec![1.0, 0.0], 0.8);
        assert_eq!(
            cache.get_cache("key", &semantic_context(None)),
            Err(super::Error::InvalidEntry)
        );
    }

    #[test]
    fn response_cache_turns_invalid_entries_into_misses() {
        let (cache, _, _) = cache_with_recording(
            [
                ok(),
                Ok(redis::Value::Array(vec![
                    redis::Value::Int(1),
                    redis::Value::BulkString(b"document".to_vec()),
                    redis::Value::Array(vec![
                        redis::Value::BulkString(b"response".to_vec()),
                        redis::Value::BulkString(b"not-json".to_vec()),
                        redis::Value::BulkString(b"vector_distance".to_vec()),
                        redis::Value::BulkString(b"0.1".to_vec()),
                    ]),
                ])),
            ],
            vec![1.0, 0.0],
            0.8,
        );
        let service = ResponseCache::new(Arc::new(cache));
        let request = ResponseCacheRequest {
            key: CacheKeyInput {
                preset: Some("key".into()),
                ..Default::default()
            },
            context: semantic_context(None),
            ..ResponseCacheRequest::new(CacheKeyInput::default())
        };
        assert_eq!(service.lookup(&request, Duration::ZERO).unwrap(), None);
    }

    #[tokio::test]
    async fn async_set_and_get_use_shared_document_helpers() {
        let entry = CacheEntry {
            timestamp: Some(1.0),
            response: json!({"answer": "ok"}),
        };
        let encoded = ResponseCacheCodec.encode(&entry).unwrap();
        let (cache, requests, calls) = cache_with_recording(
            [ok(), ok(), ok(), Ok(search_hit(encoded, "0.1"))],
            vec![1.0, 0.0],
            0.8,
        );
        let context = semantic_context(Some(Duration::from_millis(1900)));
        cache
            .async_set_cache("key", entry.clone(), context.clone())
            .await
            .unwrap();
        assert_eq!(
            cache.async_get_cache("key", &context).await.unwrap(),
            Some(entry)
        );
        let text = requests_text(&requests);
        assert!(text.contains("FT.CREATE"));
        assert!(text.contains("HSET"));
        assert!(text.contains("EXPIRE"));
        assert_eq!(calls.lock().unwrap().len(), 2);
    }
}
