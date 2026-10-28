mod support;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, CacheContext, Error, JsonCodec, SemanticCacheContext,
    semantic::{Embedder, SemanticCache, SemanticLookup},
};
use litellm_cache_qdrant_semantic::{QdrantSemanticCache, QdrantSemanticConfig, Quantization};
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{self, CompressionRatio, Distance, PointId, QuantizationType, Value, VectorParams},
};
use rstest::{fixture, rstest};
use serde_json::{Value as JsonValue, json};
use support::{FakeQdrant, FakeState, StoredPoint};

type Calls = Arc<Mutex<Vec<(String, Option<JsonValue>)>>>;
type Cache = QdrantSemanticCache<FixedEmbedder, JsonCodec<JsonValue>>;

/// Embeds known prompts, fails on anything else, and records every call.
#[derive(Clone)]
struct FixedEmbedder {
    vectors: Arc<HashMap<String, Vec<f32>>>,
    calls: Calls,
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
            calls: Calls::default(),
        }
    }
}

impl Embedder for FixedEmbedder {
    async fn async_embed(
        &self,
        input: &str,
        metadata: Option<&JsonValue>,
    ) -> Result<Vec<f32>, Error> {
        self.calls
            .lock()
            .unwrap()
            .push((input.to_owned(), metadata.cloned()));
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

#[fixture]
fn entry() -> JsonValue {
    json!({"timestamp": 1.0, "response": {"answer": 42}})
}

async fn connect(
    server: &FakeQdrant,
    vectors: impl IntoIterator<Item = (&'static str, Vec<f32>)>,
) -> Cache {
    let client = Qdrant::from_url(&server.url()).build().unwrap();
    QdrantSemanticCache::connect(
        client,
        FixedEmbedder::new(vectors),
        JsonCodec::new(),
        config(Quantization::Binary),
        tokio::runtime::Handle::current(),
    )
    .await
    .unwrap()
}

#[rstest]
#[case::binary(Quantization::Binary)]
#[case::scalar(Quantization::Scalar)]
#[case::product(Quantization::Product)]
#[tokio::test(flavor = "multi_thread")]
async fn connect_sets_collection_quantization_and_index(#[case] quantization: Quantization) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let client = Qdrant::from_url(&server.url()).build().unwrap();
    QdrantSemanticCache::connect(
        client,
        FixedEmbedder::new([]),
        JsonCodec::<JsonValue>::new(),
        config(quantization.clone()),
        tokio::runtime::Handle::current(),
    )
    .await
    .unwrap();
    let state = server.state.lock().unwrap();
    let request = &state.created_collections[0];
    let Some(qdrant::vectors_config::Config::Params(VectorParams { size, distance, .. })) = request
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
    #[expect(
        deprecated,
        reason = "the test verifies Qdrant's legacy always_ram quantization contract"
    )]
    match (quantization, quantization_config) {
        (Quantization::Binary, qdrant::quantization_config::Quantization::Binary(binary)) => {
            assert_eq!(binary.always_ram, Some(false));
        }
        (Quantization::Scalar, qdrant::quantization_config::Quantization::Scalar(scalar)) => {
            assert_eq!(scalar.r#type, QuantizationType::Int8 as i32);
            assert_eq!(scalar.quantile, Some(0.99));
            assert_eq!(scalar.always_ram, Some(false));
        }
        (Quantization::Product, qdrant::quantization_config::Quantization::Product(product)) => {
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

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn existing_collection_skips_create_and_index_failure_is_non_fatal() {
    let server = FakeQdrant::start(FakeState {
        collections: ["semantic".to_owned()].into_iter().collect(),
        fail_field_index: true,
        ..Default::default()
    })
    .await;
    let cache = connect(&server, [("hello", vec![1.0, 0.0])]).await;
    assert_eq!(cache.collection_name(), "semantic");
    assert_eq!(cache.similarity_threshold(), 0.9);
    assert_eq!(cache.vector_size(), 2);
    let state = server.state.lock().unwrap();
    assert!(state.created_collections.is_empty());
    assert!(state.index_creations >= 1);
    server.stop();
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn async_and_sync_set_get_store_exact_payload(entry: JsonValue) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = Arc::new(connect(&server, [("hello", vec![1.0, 0.0])]).await);
    let ctx = SemanticCacheContext {
        metadata: Some(json!({"tenant": "team"})),
        ..context("hello")
    };
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
        assert_eq!(payload["text"], Value::from("hello"));
        assert_eq!(payload["response"], Value::from(entry.to_string()));
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
    assert_eq!(
        *cache.embedder().calls.lock().unwrap(),
        vec![("hello".to_owned(), ctx.metadata.clone()); 4]
    );
    server.stop();
}

#[rstest]
#[case::content_parts_skip_images(
    json!([
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "world"},
                {"type": "image_url", "image_url": {"url": "ignored"}},
                {"type": "text", "text": "!"},
            ],
        },
    ]),
    "helloworld!"
)]
#[case::search_results_and_compact_citations(
    json!([{
        "role": "tool",
        "content": null,
        "search_results": [{
            "source": "source",
            "title": "title",
            "content": [{"text": "body"}],
            "citations": {"page": 1, "section": "intro"},
        }],
    }]),
    r#"sourcetitlebody{"page":1,"section":"intro"}"#
)]
#[tokio::test(flavor = "multi_thread")]
async fn prompt_matches_python_message_rules(
    #[case] messages: JsonValue,
    #[case] prompt: &'static str,
    entry: JsonValue,
) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [(prompt, vec![1.0, 0.0])]).await;
    let context = SemanticCacheContext {
        messages: Some(messages),
        ..Default::default()
    };

    cache.async_set_cache("key", entry, context).await.unwrap();

    assert_eq!(cache.embedder().calls.lock().unwrap()[0].0, prompt);
    assert_eq!(
        server.state.lock().unwrap().points[0].payload["text"],
        Value::from(prompt)
    );
    server.stop();
}

#[rstest]
#[case::no_messages(SemanticCacheContext::default())]
#[case::empty_messages(SemanticCacheContext { messages: Some(json!([])), ..Default::default() })]
#[case::responses_input_is_not_read(SemanticCacheContext { input: Some(json!("hello")), ..Default::default() })]
#[tokio::test(flavor = "multi_thread")]
async fn requests_without_messages_are_missing_a_prompt(
    #[case] context: SemanticCacheContext,
    entry: JsonValue,
) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("hello", vec![1.0, 0.0])]).await;

    assert_eq!(
        cache.async_set_cache("key", entry, context.clone()).await,
        Err(Error::MissingPrompt)
    );
    assert_eq!(
        cache.async_get_cache("key", &context).await,
        Err(Error::MissingPrompt)
    );
    assert!(cache.embedder().calls.lock().unwrap().is_empty());
    server.stop();
}

#[rstest]
#[case::other_key("other", "hello", None)]
#[case::below_similarity_threshold("key", "near", None)]
#[tokio::test(flavor = "multi_thread")]
async fn misses_and_payload_validation_are_safe(
    #[case] key: &str,
    #[case] prompt: &str,
    #[case] numeric_key_point: Option<u64>,
    entry: JsonValue,
) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(
        &server,
        [("hello", vec![1.0, 0.0]), ("near", vec![0.7, 0.71414286])],
    )
    .await;
    cache
        .async_set_cache("key", entry, context("hello"))
        .await
        .unwrap();
    if let Some(id) = numeric_key_point {
        server.insert_point(StoredPoint {
            id: Some(PointId::from(id)),
            vector: vec![1.0, 0.0],
            payload: Payload::try_from(json!({
                "litellm_cache_key": id,
                "response": "{}",
            }))
            .unwrap()
            .into(),
        });
    }

    assert_eq!(
        cache.async_get_cache(key, &context(prompt)).await.unwrap(),
        None
    );
    server.stop();
}

#[rstest]
#[case::hit("key", context("hello"), Ok((true, Some(1.0))))]
#[case::below_similarity_threshold("key", context("near"), Ok((false, Some(0.7))))]
#[case::no_results("other", context("hello"), Ok((false, Some(0.0))))]
#[case::no_prompt("key", SemanticCacheContext::default(), Err(Error::MissingPrompt))]
#[tokio::test(flavor = "multi_thread")]
async fn lookup_reports_python_semantic_similarity(
    #[case] key: &'static str,
    #[case] context: SemanticCacheContext,
    #[case] expected: Result<(bool, Option<f64>), Error>,
    #[values(false, true)] use_async: bool,
    entry: JsonValue,
) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = Arc::new(
        connect(
            &server,
            [("hello", vec![1.0, 0.0]), ("near", vec![0.7, 0.71414286])],
        )
        .await,
    );
    cache
        .async_set_cache("key", entry.clone(), self::context("hello"))
        .await
        .unwrap();
    server.insert_point(StoredPoint {
        id: Some(PointId::from(99_u64)),
        vector: vec![1.0, 0.0],
        payload: Payload::try_from(json!({"litellm_cache_key": 99, "response": "{}"}))
            .unwrap()
            .into(),
    });

    let lookup = if use_async {
        cache.async_get_cache_with_similarity(key, &context).await
    } else {
        let cache = Arc::clone(&cache);
        tokio::task::spawn_blocking(move || cache.get_cache_with_similarity(key, &context))
            .await
            .unwrap()
    };

    match (lookup, expected) {
        (Ok(SemanticLookup { value, similarity }), Ok((hit, expected))) => {
            assert_eq!(value, hit.then_some(entry));
            assert_eq!(similarity.is_some(), expected.is_some());
            if let (Some(similarity), Some(expected)) = (similarity, expected) {
                assert!((similarity - expected).abs() < 1e-6, "{similarity}");
            }
        }
        (lookup, expected) => assert_eq!(lookup.map(|_| ()), expected.map(|_| ())),
    }
    server.stop();
}

#[rstest]
#[case::codec_decodes_the_payload(Some(json!("{\"a\":1}")), Ok(Some(json!({"a": 1}))))]
#[case::undecodable_response(Some(json!("not json")), Err(Error::InvalidEntry))]
#[case::non_string_response(Some(json!(1)), Err(Error::InvalidEntry))]
#[case::missing_response(None, Err(Error::InvalidEntry))]
#[tokio::test(flavor = "multi_thread")]
async fn stored_responses_go_through_the_codec(
    #[case] response: Option<JsonValue>,
    #[case] expected: Result<Option<JsonValue>, Error>,
) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("hello", vec![1.0, 0.0])]).await;
    let mut payload = serde_json::Map::new();
    payload.insert("litellm_cache_key".to_owned(), json!("key"));
    if let Some(response) = response {
        payload.insert("response".to_owned(), response);
    }
    server.insert_point(StoredPoint {
        id: Some(PointId::from(1_u64)),
        vector: vec![1.0, 0.0],
        payload: Payload::try_from(JsonValue::Object(payload))
            .unwrap()
            .into(),
    });

    assert_eq!(
        cache.async_get_cache("key", &context("hello")).await,
        expected
    );
    server.stop();
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn embedding_failures_propagate() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, []).await;

    assert_eq!(
        cache.async_get_cache("key", &context("unknown")).await,
        Err(Error::Unavailable)
    );
    server.stop();
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn ttl_is_ignored_and_entries_do_not_expire(entry: JsonValue) {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("one", vec![1.0, 0.0])]).await;
    let ctx = context("one").with_ttl(Some(Duration::from_secs(1)));

    assert_eq!(cache.get_ttl(&ctx), None);
    cache
        .async_set_cache("ttl", entry, ctx.clone())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    assert!(cache.async_get_cache("ttl", &ctx).await.unwrap().is_some());
    server.stop();
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn pipeline_upserts_each_entry_and_waits_for_indexing() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("one", vec![1.0, 0.0])]).await;

    cache
        .async_set_cache_pipeline(
            vec![
                ("one".to_owned(), json!({"n": 1})),
                ("two".to_owned(), json!({"n": 2})),
            ],
            context("one"),
        )
        .await
        .unwrap();

    for (key, value) in [("one", json!({"n": 1})), ("two", json!({"n": 2}))] {
        assert_eq!(
            cache.async_get_cache(key, &context("one")).await.unwrap(),
            Some(value)
        );
    }
    assert_eq!(
        server.state.lock().unwrap().upsert_waits,
        vec![Some(true), Some(true)]
    );
    server.stop();
}

#[rstest]
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

/// `_payload_matches_cache_key` compares `str(cached_key) == str(key)`, so a point whose stored
/// key is the number 99 answers a lookup for `"99"`.
#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn numeric_stored_cache_keys_match_like_python_str() {
    let server = FakeQdrant::start(FakeState::default()).await;
    let cache = connect(&server, [("hello", vec![1.0, 0.0])]).await;
    server.insert_point(StoredPoint {
        id: Some(PointId::from(99_u64)),
        vector: vec![1.0, 0.0],
        payload: Payload::try_from(json!({"litellm_cache_key": 99, "response": "{}"}))
            .unwrap()
            .into(),
    });

    let lookup = cache
        .async_get_cache_with_similarity("99", &context("hello"))
        .await
        .unwrap();

    assert_eq!(lookup.value, Some(json!({})));
    assert!((lookup.similarity.unwrap() - 1.0).abs() < 1e-6);
    server.stop();
}
