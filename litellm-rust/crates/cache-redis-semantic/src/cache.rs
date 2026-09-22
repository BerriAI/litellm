use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{
    BaseCache, CacheCodec, Error, SemanticCacheContext,
    semantic::{Embedder, SemanticCache, SemanticLookup, prompt_from_context},
};
use litellm_cache_redis::{
    RedisTopology,
    connection::{ConnectionRef, Connections},
};
use sha2::{Digest, Sha256};

use crate::{
    RedisSemanticConfig,
    index::{CACHE_KEY_FIELD, Index, VECTOR_FIELD},
    reply::{bytes_field, first_document, number_field, string_field},
};

struct Inner {
    index: Index,
    distance_threshold: f64,
    clock: fn() -> f64,
}

impl Inner {
    fn new(config: RedisSemanticConfig, clock: fn() -> f64) -> Self {
        Self {
            index: Index::new(config.index_name),
            distance_threshold: 1.0 - f64::from(config.similarity_threshold),
            clock,
        }
    }

    fn store(
        &self,
        connection: &mut ConnectionRef<'_>,
        tag: &str,
        response: Vec<u8>,
        prompt: &str,
        vector: &[f32],
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let index = self.index.ensure(connection, vector.len())?;
        let entry_id = entry_id(prompt, tag);
        let hash_key = format!("{index}:{entry_id}");
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
    ) -> Result<SemanticLookup<Vec<u8>>, Error> {
        let index = self.index.ensure(connection, vector.len())?;
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
            return Ok(SemanticLookup::miss(Some(0.0)));
        };
        if string_field(fields, CACHE_KEY_FIELD).as_deref() != Some(tag) {
            return Ok(SemanticLookup::miss(Some(0.0)));
        }
        // redisvl's range query only returns entries within the distance threshold, so a
        // farther hit reads as no result.
        let Some(distance) = number_field(fields, "vector_distance")
            .filter(|distance| *distance <= self.distance_threshold)
        else {
            return Ok(SemanticLookup::miss(Some(0.0)));
        };
        let Some(response) = bytes_field(fields, "response") else {
            return Ok(SemanticLookup::miss(Some(0.0)));
        };
        Ok(SemanticLookup {
            value: Some(response),
            similarity: Some(1.0 - distance),
        })
    }
}

/// `RedisSemanticCache`: a redisvl-compatible semantic index on Redis Stack. Values go through
/// the injected codec, so the response layer decides what a cached entry is.
pub struct RedisSemanticCache<E, S, C = redis::Connection> {
    connections: Arc<Connections<C>>,
    embedder: E,
    codec: S,
    inner: Arc<Inner>,
}

impl<E: Embedder, S: CacheCodec> RedisSemanticCache<E, S> {
    pub fn new(
        url: &str,
        embedder: E,
        codec: S,
        config: RedisSemanticConfig,
    ) -> Result<Self, Error> {
        Ok(Self {
            connections: Arc::new(Connections::open(url, &RedisTopology::Standalone)?),
            embedder,
            codec,
            inner: Arc::new(Inner::new(config, timestamp)),
        })
    }
}

impl<E, S, C> RedisSemanticCache<E, S, C>
where
    E: Embedder,
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn with_connection(
        connection: C,
        embedder: E,
        codec: S,
        config: RedisSemanticConfig,
    ) -> Self {
        Self {
            connections: Arc::new(Connections::fixed(connection)),
            embedder,
            codec,
            inner: Arc::new(Inner::new(config, timestamp)),
        }
    }

    pub fn with_clock(self, clock: fn() -> f64) -> Self {
        let config = RedisSemanticConfig {
            index_name: self.index_name().to_owned(),
            similarity_threshold: self.similarity_threshold(),
        };
        Self {
            inner: Arc::new(Inner::new(config, clock)),
            ..self
        }
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    pub fn index_name(&self) -> &str {
        self.inner.index.name()
    }

    pub fn similarity_threshold(&self) -> f32 {
        (1.0 - self.inner.distance_threshold) as f32
    }

    fn tag<'a>(key: &'a str, context: &'a SemanticCacheContext) -> &'a str {
        context.scope.as_deref().unwrap_or(key)
    }

    fn decode(&self, lookup: SemanticLookup<Vec<u8>>) -> Result<SemanticLookup<S::Value>, Error> {
        Ok(SemanticLookup {
            value: lookup
                .value
                .map(|bytes| self.codec.decode(&bytes))
                .transpose()?,
            similarity: lookup.similarity,
        })
    }
}

impl<E, S, C> BaseCache for RedisSemanticCache<E, S, C>
where
    E: Embedder,
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type Value = S::Value;
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
        let response = self.codec.encode(&value)?;
        let vector = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        let tag = Self::tag(key, context);
        self.connections.execute(|connection| {
            self.inner
                .store(connection, tag, response, &prompt, &vector, context.ttl)
        })
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        self.get_cache_with_similarity(key, context)
            .map(|lookup| lookup.value)
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
        let response = self.codec.encode(&value)?;
        let vector = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let tag = Self::tag(key, &context).to_owned();
        let inner = Arc::clone(&self.inner);
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            inner.store(connection, &tag, response, &prompt, &vector, context.ttl)
        })
        .await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<Option<Self::Value>, Error> {
        self.async_get_cache_with_similarity(key, context)
            .await
            .map(|lookup| lookup.value)
    }
}

/// Python stamps a similarity of `0.0` when there is no prompt or no hit in the key's scope.
impl<E, S, C> SemanticCache for RedisSemanticCache<E, S, C>
where
    E: Embedder,
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        let Some(prompt) = prompt_from_context(context) else {
            return Ok(SemanticLookup::miss(Some(0.0)));
        };
        let vector = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        let tag = Self::tag(key, context);
        let lookup = self
            .connections
            .execute(|connection| self.inner.lookup(connection, tag, &vector))?;
        self.decode(lookup)
    }

    async fn async_get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        let Some(prompt) = prompt_from_context(context) else {
            return Ok(SemanticLookup::miss(Some(0.0)));
        };
        let vector = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let tag = Self::tag(key, context).to_owned();
        let inner = Arc::clone(&self.inner);
        let lookup = Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            inner.lookup(connection, &tag, &vector)
        })
        .await?;
        self.decode(lookup)
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
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
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
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

fn ttl_seconds(ttl: Duration) -> u64 {
    ttl.as_secs()
        .saturating_add(u64::from(ttl.subsec_nanos() > 0))
        .max(1)
}
