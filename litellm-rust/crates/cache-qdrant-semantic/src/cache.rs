use futures_util::future::try_join_all;
use litellm_cache::{
    BaseCache, CacheCodec, Error, SemanticCacheContext,
    semantic::{Embedder, SemanticCache, SemanticLookup, prompt_from_messages},
};
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        BinaryQuantizationBuilder, CompressionRatio, Condition, CreateCollectionBuilder,
        CreateFieldIndexCollectionBuilder, Distance, FieldType, Filter, PointStruct,
        ProductQuantizationBuilder, QuantizationSearchParamsBuilder, ScalarQuantizationBuilder,
        SearchParamsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
};
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::{QdrantSemanticConfig, Quantization};

pub struct QdrantSemanticCache<E: Embedder, C: CacheCodec> {
    client: Qdrant,
    embedder: E,
    codec: C,
    config: QdrantSemanticConfig,
    runtime: tokio::runtime::Handle,
}

impl<E: Embedder, C: CacheCodec> QdrantSemanticCache<E, C> {
    pub async fn connect(
        client: Qdrant,
        embedder: E,
        codec: C,
        config: QdrantSemanticConfig,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, Error> {
        let exists = client
            .collection_exists(config.collection_name.clone())
            .await
            .map_err(|_| Error::Unavailable)?;
        if !exists {
            client
                .create_collection(
                    CreateCollectionBuilder::new(config.collection_name.clone())
                        .vectors_config(VectorParamsBuilder::new(
                            config.vector_size,
                            Distance::Cosine,
                        ))
                        .quantization_config(quantization(&config.quantization)),
                )
                .await
                .map_err(|_| Error::Unavailable)?;
        }
        let _ = client
            .create_field_index(CreateFieldIndexCollectionBuilder::new(
                config.collection_name.clone(),
                "litellm_cache_key".to_owned(),
                FieldType::Keyword,
            ))
            .await;
        Ok(Self {
            client,
            embedder,
            codec,
            config,
            runtime,
        })
    }

    pub fn collection_name(&self) -> &str {
        &self.config.collection_name
    }

    pub fn similarity_threshold(&self) -> f64 {
        self.config.similarity_threshold
    }

    pub fn vector_size(&self) -> u64 {
        self.config.vector_size
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Python reads `kwargs["messages"]` unguarded, so a request without messages fails.
    fn prompt(context: &SemanticCacheContext) -> Result<String, Error> {
        prompt_from_messages(context).ok_or(Error::MissingPrompt)
    }

    async fn set(
        &self,
        key: &str,
        value: C::Value,
        context: &SemanticCacheContext,
    ) -> Result<(), Error> {
        let prompt = Self::prompt(context)?;
        let vector = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let response =
            String::from_utf8(self.codec.encode(&value)?).map_err(|_| Error::InvalidEntry)?;
        let payload = Payload::try_from(json!({
            "litellm_cache_key": key,
            "text": prompt,
            "response": response,
        }))
        .map_err(|_| Error::InvalidEntry)?;
        self.client
            .upsert_points(
                UpsertPointsBuilder::new(
                    self.collection_name(),
                    vec![PointStruct::new(
                        Uuid::new_v4().to_string(),
                        vector,
                        payload,
                    )],
                )
                .wait(true),
            )
            .await
            .map_err(|_| Error::Unavailable)?;
        Ok(())
    }

    async fn get(
        &self,
        key: &str,
        context: &SemanticCacheContext,
    ) -> Result<SemanticLookup<C::Value>, Error> {
        let prompt = Self::prompt(context)?;
        let vector = self
            .embedder
            .async_embed(&prompt, context.metadata.as_ref())
            .await?;
        let result = self
            .client
            .search_points(
                SearchPointsBuilder::new(self.collection_name(), vector, 1)
                    .with_payload(true)
                    .filter(Filter::must([Condition::matches(
                        "litellm_cache_key",
                        key.to_owned(),
                    )]))
                    .params(
                        SearchParamsBuilder::default().quantization(
                            QuantizationSearchParamsBuilder::default()
                                .ignore(false)
                                .rescore(true)
                                .oversampling(3.0),
                        ),
                    ),
            )
            .await
            .map_err(|_| Error::Unavailable)?;
        let Some(point) = result.result.into_iter().next() else {
            return Ok(SemanticLookup::miss(Some(0.0)));
        };
        let payload: Map<String, Value> = Payload::from(point.payload).into();
        if !payload
            .get("litellm_cache_key")
            .is_some_and(|cached| python_str(cached).as_deref() == Some(key))
        {
            return Ok(SemanticLookup::miss(Some(0.0)));
        }
        let similarity = f64::from(point.score);
        if similarity < self.config.similarity_threshold {
            return Ok(SemanticLookup::miss(Some(similarity)));
        }
        let response = payload
            .get("response")
            .and_then(Value::as_str)
            .ok_or(Error::InvalidEntry)?;
        Ok(SemanticLookup {
            value: Some(self.codec.decode(response.as_bytes())?),
            similarity: Some(similarity),
        })
    }
}

fn quantization(value: &Quantization) -> qdrant_client::qdrant::quantization_config::Quantization {
    match value {
        Quantization::Binary => BinaryQuantizationBuilder::new(false).into(),
        Quantization::Scalar => ScalarQuantizationBuilder::default()
            .quantile(0.99)
            .always_ram(false)
            .into(),
        Quantization::Product => ProductQuantizationBuilder::new(CompressionRatio::X16.into())
            .always_ram(false)
            .into(),
    }
}

impl<E: Embedder, C: CacheCodec> BaseCache for QdrantSemanticCache<E, C> {
    type Value = C::Value;
    type Context = SemanticCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<std::time::Duration> {
        None
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &Self::Context,
    ) -> Result<(), Error> {
        self.runtime.block_on(self.set(key, value, context))
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
        self.set(key, value, &context).await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<Option<Self::Value>, Error> {
        self.get(key, context).await.map(|lookup| lookup.value)
    }

    async fn async_set_cache_pipeline(
        &self,
        entries: Vec<(String, Self::Value)>,
        context: Self::Context,
    ) -> Result<(), Error> {
        try_join_all(entries.into_iter().map(|(key, value)| {
            let context = context.clone();
            async move { self.async_set_cache(&key, value, context).await }
        }))
        .await
        .map(|_| ())
    }
}

/// Python stamps the top point's score, even below the threshold, and `0.0` when there is no
/// point or it belongs to another key. A request without messages fails before any search.
impl<E: Embedder, C: CacheCodec> SemanticCache for QdrantSemanticCache<E, C> {
    fn get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        self.runtime.block_on(self.get(key, context))
    }

    async fn async_get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        self.get(key, context).await
    }
}

/// `str(value)` for the scalar payload values `_payload_matches_cache_key` compares; `None` for
/// null (a pre-isolation point without a key) and for containers, which never equal a key.
fn python_str(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(true) => Some("True".into()),
        Value::Bool(false) => Some("False".into()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}
