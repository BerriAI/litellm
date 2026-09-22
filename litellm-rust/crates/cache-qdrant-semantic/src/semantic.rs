use std::future::Future;

use futures_util::future::try_join_all;
use litellm_cache::{BaseCache, CacheCodec, CacheConnectionResult, Error, SemanticCacheContext};
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

use crate::prompt_from_messages;

pub trait Embedder: Send + Sync + 'static {
    fn model(&self) -> &str;
    fn embed(&self, input: &str) -> impl Future<Output = Result<Vec<f32>, Error>> + Send;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Quantization {
    Binary,
    Scalar,
    Product,
}

pub struct QdrantSemanticConfig {
    pub collection_name: String,
    pub similarity_threshold: f64,
    pub vector_size: u64,
    pub quantization: Quantization,
}

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

    fn prompt(context: &SemanticCacheContext) -> Result<String, Error> {
        let Some(messages) = context.messages.as_ref().and_then(Value::as_array) else {
            return Err(Error::MissingPrompt);
        };
        if messages.is_empty() {
            return Err(Error::MissingPrompt);
        }
        Ok(prompt_from_messages(messages))
    }

    async fn set(
        &self,
        key: &str,
        value: C::Value,
        context: &SemanticCacheContext,
    ) -> Result<(), Error> {
        let prompt = Self::prompt(context)?;
        let vector = self.embedder.embed(&prompt).await?;
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
    ) -> Result<Option<C::Value>, Error> {
        let prompt = Self::prompt(context)?;
        let vector = self.embedder.embed(&prompt).await?;
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
            return Ok(None);
        };
        let payload: Map<String, Value> = Payload::from(point.payload).into();
        if payload.get("litellm_cache_key").and_then(Value::as_str) != Some(key) {
            return Ok(None);
        }
        if f64::from(point.score) < self.config.similarity_threshold {
            return Ok(None);
        }
        let response = payload
            .get("response")
            .and_then(Value::as_str)
            .ok_or(Error::InvalidEntry)?;
        self.codec.decode(response.as_bytes()).map(Some)
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
        self.runtime.block_on(self.get(key, context))
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
        self.get(key, context).await
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

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        Err(Error::UnsupportedOperation)
    }
}
