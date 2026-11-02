use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, CacheCodec, Error, SemanticCacheContext,
    semantic::{Embedder, SemanticCache, SemanticLookup, prompt_from_context},
};
use litellm_cache_redis::{RedisTopology, connection::Connections};

use crate::{
    ValkeySemanticConfig,
    index::IndexState,
    search::{embedding_bytes, scope_tag, search_document, write_document},
};

/// `ValkeySemanticCache`: a semantic cache on valkey-search's TAG + VECTOR index. Values go
/// through the injected codec, so the response layer decides what a cached entry is.
pub struct ValkeySemanticCache<E, S, C = redis::Connection> {
    connections: Arc<Connections<C>>,
    embedder: E,
    codec: S,
    config: ValkeySemanticConfig,
    index_dimension: Arc<Mutex<Option<usize>>>,
}

impl<E: Embedder, S: CacheCodec> ValkeySemanticCache<E, S> {
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
    S: CacheCodec,
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

impl<E, S, C> ValkeySemanticCache<E, S, C>
where
    E: Embedder,
    S: CacheCodec + Clone,
    C: redis::ConnectionLike + Send + 'static,
{
    /// The same index and connections behind a different embedder.
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
        let embedding = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        let response = self.codec.encode(&value)?;
        let index = self.index_state();
        self.connections.execute(|connection| {
            write_document(
                connection,
                &index,
                &scope_tag(key),
                &prompt,
                response,
                embedding_bytes(&embedding),
                self.get_ttl(context),
            )
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
        let embedding = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let response = self.codec.encode(&value)?;
        let index = self.index_state();
        let scope = scope_tag(key);
        Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            write_document(
                connection,
                &index,
                &scope,
                &prompt,
                response,
                embedding_bytes(&embedding),
                context.ttl,
            )
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

/// Python stamps a similarity of `0.0` when there is no prompt or no document in the key's
/// scope, and the closest document's similarity even when it misses the threshold.
impl<E, S, C> SemanticCache for ValkeySemanticCache<E, S, C>
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
        let embedding = self.embedder.embed(&prompt, context.metadata.as_ref())?;
        let index = self.index_state();
        let lookup = self.connections.execute(|connection| {
            search_document(
                connection,
                &index,
                &scope_tag(key),
                embedding_bytes(&embedding),
            )
        })?;
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
        let embedding = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let index = self.index_state();
        let scope = scope_tag(key);
        let lookup = Connections::run_blocking(Arc::clone(&self.connections), move |connection| {
            search_document(connection, &index, &scope, embedding_bytes(&embedding))
        })
        .await?;
        self.decode(lookup)
    }
}
