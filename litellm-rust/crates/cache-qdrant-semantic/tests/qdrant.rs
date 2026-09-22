#[path = "support/mod.rs"]
mod support;

use std::{collections::HashMap, sync::Arc, time::Duration};

use litellm_cache::{BaseCache, CacheCodec, CacheContext, Error, SemanticCacheContext};
use litellm_cache_qdrant_semantic::{
    Embedder, QdrantSemanticCache, QdrantSemanticConfig, Quantization,
};
use litellm_cache_response::{
    CacheEntry, CacheKeyInput, ResponseCache, ResponseCacheCodec, ResponseCacheRequest,
};
use qdrant_client::Payload;
use qdrant_client::{
    Qdrant,
    qdrant::{self, CompressionRatio, Distance, PointId, QuantizationType, Value, VectorParams},
};
use serde_json::{Value as JsonValue, json};

use support::{FakeQdrant, FakeState, StoredPoint};

#[derive(Clone)]
struct FixedEmbedder {
    vectors: Arc<HashMap<String, Vec<f32>>>,
}

impl FixedEmbedder {
    fn new(vectors: impl IntoIterator<Item = (&'static str, Vec<f32>)>) -> Self {
        Self {
            vectors: Arc::new(
                vectors
                    .into_iter()
                    .map(|(prompt, vector)| (prompt.to_owned(), vector))
                    .collect(),
            ),
        }
    }
}

impl Embedder for FixedEmbedder {
    fn model(&self) -> &str {
        "fixed"
    }

    async fn embed(&self, input: &str) -> Result<Vec<f32>, Error> {
        self.vectors.get(input).cloned().ok_or(Error::Unavailable)
    }
}

fn config(quantization: Quantization) -> QdrantSemanticConfig {
    QdrantSemanticConfig {
        collection_name: "semantic".to_owned(),
        similarity_threshold: 0.9,
        vector_size: 2,
        quantization,
    }
}

fn context(prompt: &str) -> SemanticCacheContext {
    SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": prompt}])),
        ..Default::default()
    }
}

fn value(response: JsonValue) -> CacheEntry {
    CacheEntry {
        timestamp: Some(1.0),
        response,
    }
}

async fn connect(
    server: &FakeQdrant,
    vectors: impl IntoIterator<Item = (&'static str, Vec<f32>)>,
) -> QdrantSemanticCache<FixedEmbedder, ResponseCacheCodec> {
    let client = Qdrant::from_url(&server.url()).build().unwrap();
    QdrantSemanticCache::connect(
        client,
        FixedEmbedder::new(vectors),
        ResponseCacheCodec,
        config(Quantization::Binary),
        tokio::runtime::Handle::current(),
    )
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
#[expect(
    deprecated,
    reason = "the test verifies Qdrant's legacy always_ram quantization contract"
)]
async fn connect_sets_collection_quantization_and_index() {
    for (quantization, expected) in [
        (Quantization::Binary, 0),
        (Quantization::Scalar, 1),
        (Quantization::Product, 2),
    ] {
        let server = FakeQdrant::start(FakeState::default()).await;
        let client = Qdrant::from_url(&server.url()).build().unwrap();
        QdrantSemanticCache::connect(
            client,
            FixedEmbedder::new([]),
            ResponseCacheCodec,
            config(quantization),
            tokio::runtime::Handle::current(),
        )
        .await
        .unwrap();
        let state = server.state.lock().unwrap();
        let request = &state.created_collections[0];
        let Some(qdrant::vectors_config::Config::Params(VectorParams { size, distance, .. })) =
            request
                .vectors_config
                .as_ref()
                .and_then(|config| config.config.clone())
        else {
            panic!("missing vector params");
        };
        assert_eq!(size, 2);
        assert_eq!(distance, Distance::Cosine as i32);
        let quantization_config = request
            .quantization_config
            .as_ref()
            .unwrap()
            .quantization
            .unwrap();
        match (expected, quantization_config) {
            (0, qdrant::quantization_config::Quantization::Binary(binary)) => {
                assert_eq!(binary.always_ram, Some(false));
            }
            (1, qdrant::quantization_config::Quantization::Scalar(scalar)) => {
                assert_eq!(scalar.r#type, QuantizationType::Int8 as i32);
                assert_eq!(scalar.quantile, Some(0.99));
                assert_eq!(scalar.always_ram, Some(false));
            }
            (2, qdrant::quantization_config::Quantization::Product(product)) => {
                assert_eq!(product.compression, CompressionRatio::X16 as i32);
                assert_eq!(product.always_ram, Some(false));
            }
            _ => panic!("unexpected quantization"),
        }
        assert!(state.index_creations >= 1);
        assert_eq!(state.field_indexes[0].collection_name, "semantic");
        assert_eq!(state.field_indexes[0].field_name, "litellm_cache_key");
        assert_eq!(
            state.field_indexes[0].field_type,
            Some(qdrant::FieldType::Keyword as i32)
        );
        server.stop();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn existing_collection_skips_create_and_index_failure_is_non_fatal() {
    let server = FakeQdrant::start(FakeState {
        collections: ["semantic".to_owned()].into_iter().collect(),
        fail_field_index: true,
        ..Default::default()
    })
    .await;
    let _cache = connect(&server, [("hello", vec![1.0, 0.0])]).await;
    let state = server.state.lock().unwrap();
    assert!(state.created_collections.is_empty());
    assert!(state.index_creations >= 1);
    server.stop();
}

#[tokio::test(flavor = "multi_thread")]
async fn async_and_sync_set_get_store_exact_payload() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = Arc::new(connect(&server, [("hello", vec![1.0, 0.0])]).await);
    let ctx = context("hello");
    let entry = value(json!({"answer": 42}));
    cache
        .async_set_cache("key", entry.clone(), ctx.clone())
        .await
        .unwrap();
    assert_eq!(
        cache.async_get_cache("key", &ctx).await.unwrap().as_ref(),
        Some(&entry)
    );
    {
        let state = server.state.lock().unwrap();
        let payload = &state.points[0].payload;
        let mut payload_keys = payload.keys().cloned().collect::<Vec<_>>();
        payload_keys.sort();
        assert_eq!(payload_keys, ["litellm_cache_key", "response", "text"]);
        assert_eq!(payload["litellm_cache_key"], Value::from("key"));
        assert_eq!(
            payload["response"],
            Value::from(String::from_utf8(ResponseCacheCodec.encode(&entry).unwrap()).unwrap())
        );
    }
    let sync_entry = entry.clone();
    let sync_cache = cache.clone();
    let sync_ctx = ctx.clone();
    tokio::task::spawn_blocking(move || {
        sync_cache
            .set_cache("sync", sync_entry.clone(), &sync_ctx)
            .unwrap();
        assert_eq!(
            sync_cache.get_cache("sync", &sync_ctx).unwrap(),
            Some(sync_entry)
        );
    })
    .await
    .unwrap();
    server.stop();
}

#[tokio::test(flavor = "multi_thread")]
async fn misses_and_payload_validation_are_safe() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(
        &server,
        [("hello", vec![1.0, 0.0]), ("near", vec![0.7, 0.71414286])],
    )
    .await;
    let entry = value(json!({"answer": 1}));
    cache
        .async_set_cache("key", entry, context("hello"))
        .await
        .unwrap();
    assert_eq!(
        cache
            .async_get_cache("other", &context("hello"))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        cache
            .async_get_cache("key", &context("near"))
            .await
            .unwrap(),
        None
    );
    server.insert_point(StoredPoint {
        id: Some(PointId::from(99_u64)),
        vector: vec![1.0, 0.0],
        payload: Payload::try_from(json!({
            "litellm_cache_key": 99,
            "response": "{}",
        }))
        .unwrap()
        .into(),
    });
    assert_eq!(
        cache
            .async_get_cache("99", &context("hello"))
            .await
            .unwrap(),
        None
    );
    server.stop();
}

#[tokio::test(flavor = "multi_thread")]
async fn decoding_errors_missing_prompt_pipeline_and_ttl_behave_as_required() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("one", vec![1.0, 0.0]), ("two", vec![0.0, 1.0])]).await;
    let empty = SemanticCacheContext::default();
    assert_eq!(
        cache
            .async_set_cache("key", value(json!({})), empty.clone())
            .await,
        Err(Error::MissingPrompt)
    );
    assert_eq!(
        cache.async_get_cache("key", &empty).await,
        Err(Error::MissingPrompt)
    );
    assert_eq!(
        cache.async_get_cache("key", &context("unknown")).await,
        Err(Error::Unavailable)
    );
    cache
        .async_set_cache(
            "ttl",
            value(json!({"ttl": true})),
            context("one").with_ttl(Some(Duration::from_secs(1))),
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    assert!(
        cache
            .async_get_cache(
                "ttl",
                &context("one").with_ttl(Some(Duration::from_secs(1))),
            )
            .await
            .unwrap()
            .is_some()
    );
    cache
        .async_set_cache_pipeline(
            vec![
                ("one".to_owned(), value(json!({"n": 1}))),
                ("two".to_owned(), value(json!({"n": 2}))),
            ],
            context("one"),
        )
        .await
        .unwrap();
    assert!(
        cache
            .async_get_cache("one", &context("one"))
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        cache
            .async_get_cache("two", &context("one"))
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        server.state.lock().unwrap().upsert_waits,
        vec![Some(true), Some(true), Some(true)]
    );
    assert_eq!(cache.get_ttl(&context("one")), None);
    assert_eq!(
        cache.test_connection().await,
        Err(Error::UnsupportedOperation)
    );
    server.stop();
}

#[tokio::test(flavor = "multi_thread")]
async fn response_payloads_decode_and_invalid_entries_fail() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("hello", vec![1.0, 0.0])]).await;
    for (key, response) in [
        ("python", json!("{'timestamp': 1.0, 'response': {'a': 1}}")),
        ("garbage", json!("not json")),
        ("missing", json!("unused")),
    ] {
        let mut payload = serde_json::Map::new();
        payload.insert("litellm_cache_key".to_owned(), json!(key));
        if key != "missing" {
            payload.insert("response".to_owned(), response);
        }
        server.insert_point(StoredPoint {
            id: Some(PointId::from(key.len() as u64)),
            vector: vec![1.0, 0.0],
            payload: Payload::try_from(JsonValue::Object(payload))
                .unwrap()
                .into(),
        });
    }
    assert_eq!(
        cache
            .async_get_cache("python", &context("hello"))
            .await
            .unwrap(),
        Some(value(json!({"a": 1})))
    );
    assert_eq!(
        cache.async_get_cache("garbage", &context("hello")).await,
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.async_get_cache("missing", &context("hello")).await,
        Err(Error::InvalidEntry)
    );
    server.stop();
}

#[tokio::test(flavor = "multi_thread")]
async fn response_cache_facade_turns_invalid_entry_into_miss() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = Arc::new(connect(&server, [("hello", vec![1.0, 0.0])]).await);
    let request = ResponseCacheRequest::<SemanticCacheContext>::new(CacheKeyInput {
        preset: Some("key".to_owned()),
        ..Default::default()
    })
    .with_context(context("hello"));
    let response = json!({"answer": 42});
    let facade = ResponseCache::new(cache.clone());
    facade
        .async_store(&request, response.clone(), Duration::from_secs(1))
        .await
        .unwrap();
    assert_eq!(
        facade
            .async_lookup(&request, Duration::from_secs(1))
            .await
            .unwrap(),
        Some(response)
    );
    {
        let mut state = server.state.lock().unwrap();
        state.points[0]
            .payload
            .insert("response".to_owned(), Value::from("not json"));
    }
    assert_eq!(
        facade
            .async_lookup(&request, Duration::from_secs(1))
            .await
            .unwrap(),
        None
    );
    server.stop();
}

#[tokio::test(flavor = "multi_thread")]
async fn stopped_qdrant_server_maps_to_unavailable() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("hello", vec![1.0, 0.0])]).await;
    server.stop();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        cache.async_get_cache("key", &context("hello")).await,
        Err(Error::Unavailable)
    );
}
