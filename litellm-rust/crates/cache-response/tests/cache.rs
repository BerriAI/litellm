mod support;

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::{BaseCache, ExactCacheContext};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheEntry, CacheKey, ResponseCache};
use redis_test::MockCmd;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::{DEFAULT, key, keyed, memory, redis};

type Memory = Arc<ResponseCache<InMemoryCache<CacheEntry>>>;

#[rstest]
#[tokio::test]
async fn sync_and_async_consumers_share_keys_ttls_and_freshness(key: CacheKey) {
    let clock = Arc::new(AtomicU64::new(100));
    let backend = Arc::new(InMemoryCache::with_clock(
        Some(8),
        Some(Duration::from_secs(600)),
        {
            let clock = clock.clone();
            move || Duration::from_secs(clock.load(Ordering::SeqCst))
        },
    ));
    let cache = ResponseCache::new(backend.clone());
    let context = ExactCacheContext {
        ttl: Some(Duration::from_secs(10)),
    };
    let max_age = Some(Duration::from_secs(5));
    cache
        .store(
            &key,
            &context,
            json!({"choices": [1], "usage": {"total_tokens": 7}}),
            Duration::from_secs(100),
        )
        .unwrap();
    assert_eq!(
        backend.expires_at("tenant:key").unwrap(),
        Some(Duration::from_secs(110))
    );
    assert!(
        cache
            .async_lookup(&key, &context, max_age, Duration::from_secs(105))
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        cache
            .lookup(&key, &context, max_age, Duration::from_secs(106))
            .unwrap(),
        None
    );
    assert_eq!(
        cache
            .lookup(&key, &context, None, Duration::from_secs(106))
            .unwrap()
            .unwrap()["usage"]["total_tokens"],
        7
    );
    clock.store(111, Ordering::SeqCst);
    assert_eq!(
        cache
            .async_lookup(&key, &context, None, Duration::from_secs(111))
            .await
            .unwrap(),
        None
    );
    cache
        .async_store(
            &key,
            context.clone(),
            json!({"choices": [2]}),
            Duration::from_secs(111),
        )
        .await
        .unwrap();
    assert_eq!(
        cache
            .lookup(&key, &context, None, Duration::from_secs(111))
            .unwrap(),
        Some(json!({"choices": [2]}))
    );
}

#[rstest]
#[case::python_sync_literal(
    br#"{'timestamp': 100.0, 'response': '{"ok": true, "text": "cached"}'}"#.as_slice()
)]
#[case::python_async_json(br#"{"timestamp":100.0,"response":{"ok":true,"text":"cached"}}"#.as_slice())]
#[tokio::test]
async fn redis_consumer_reads_python_sync_and_async_envelopes(
    key: CacheKey,
    #[case] stored: &[u8],
    #[values(false, true)] asynchronous: bool,
) {
    let cache = redis(
        vec![MockCmd::new(
            redis::cmd("GET").arg("tenant:key"),
            Ok(stored.to_vec()),
        )],
        Some("tenant"),
    );
    let now = Duration::from_secs(101);
    let found = if asynchronous {
        cache.async_lookup(&key, &DEFAULT, None, now).await.unwrap()
    } else {
        cache.lookup(&key, &DEFAULT, None, now).unwrap()
    };
    assert_eq!(found, Some(json!({"ok": true, "text": "cached"})));
}

#[rstest]
#[case::object(
    json!({"ok": true, "text": "cached"}),
    br#"{"timestamp":100.0,"response":{"ok":true,"text":"cached"}}"#.as_slice()
)]
#[case::array(json!([1, 2]), br#"{"timestamp":100.0,"response":"[1,2]"}"#.as_slice())]
#[tokio::test]
async fn redis_consumer_writes_python_compatible_json(
    key: CacheKey,
    #[case] response: Value,
    #[case] wire: &[u8],
) {
    let cache = redis(
        vec![MockCmd::new(
            redis::cmd("SETEX").arg("tenant:key").arg(600).arg(wire),
            Ok("OK"),
        )],
        Some("tenant"),
    );
    cache
        .async_store(&key, DEFAULT, response, Duration::from_secs(100))
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn invalid_entries_are_misses(key: CacheKey) {
    let cache = redis(
        vec![MockCmd::new(
            redis::cmd("GET").arg("tenant:key"),
            Ok(b"invalid".to_vec()),
        )],
        None,
    );
    assert_eq!(
        cache
            .async_lookup(&key, &DEFAULT, None, Duration::ZERO)
            .await
            .unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn captured_service_keeps_the_selected_backend_for_background_writes(
    #[from(memory)] original: Memory,
    #[from(memory)] replacement: Memory,
    key: CacheKey,
) {
    let captured = original.clone();
    let writer = tokio::spawn({
        let key = key.clone();
        async move {
            captured
                .async_store(
                    &key,
                    DEFAULT,
                    json!({"selected": "original"}),
                    Duration::from_secs(100),
                )
                .await
        }
    });
    writer.await.unwrap().unwrap();
    assert_eq!(
        original
            .lookup(&key, &DEFAULT, None, Duration::from_secs(100))
            .unwrap(),
        Some(json!({"selected":"original"}))
    );
    assert_eq!(
        replacement
            .lookup(&key, &DEFAULT, None, Duration::from_secs(100))
            .unwrap(),
        None
    );
}

#[rstest]
#[case::text(json!("hello world"))]
#[case::numeric_text(json!("123"))]
#[case::null_text(json!("null"))]
#[case::array(json!([1, 2]))]
fn non_object_responses_round_trip_through_a_typed_backend(
    memory: Memory,
    key: CacheKey,
    #[case] response: Value,
) {
    let now = Duration::from_secs(100);
    memory.store(&key, &DEFAULT, response.clone(), now).unwrap();
    assert_eq!(
        memory.lookup(&key, &DEFAULT, None, now).unwrap(),
        Some(response)
    );
}

#[rstest]
fn entries_without_timestamps_are_always_fresh(key: CacheKey) {
    let backend = Arc::new(InMemoryCache::default());
    BaseCache::set_cache(
        backend.as_ref(),
        "tenant:key",
        CacheEntry {
            timestamp: None,
            response: json!({"choices": [{"text": "legacy"}]}),
        },
        &Default::default(),
    )
    .unwrap();
    let cache = ResponseCache::new(backend);
    assert_eq!(
        cache
            .lookup(
                &key,
                &DEFAULT,
                Some(Duration::from_secs(1)),
                Duration::from_secs(100)
            )
            .unwrap(),
        Some(json!({"choices": [{"text": "legacy"}]}))
    );
}

#[rstest]
#[case::python_sync("{'timestamp': 100.0, 'response': '{\"answer\": 7}'}")]
#[case::python_async(r#"{"timestamp":100.0,"response":{"answer":7}}"#)]
#[case::bare_response(r#"{"answer":7}"#)]
#[tokio::test]
async fn gcs_reads_python_entries_and_writes_python_compatible_envelopes(
    #[case] encoded: &str,
    #[values(false, true)] asynchronous: bool,
    #[future(awt)] gcs: (wiremock::MockServer, Gcs),
) {
    use wiremock::{
        Mock, ResponseTemplate,
        matchers::{body_json, header, method, path, query_param},
    };

    let (server, cache) = gcs;
    let response = json!({"answer": 7});
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/cache%2Fpython"))
        .and(query_param("alt", "media"))
        .and(header("authorization", "Bearer token"))
        .respond_with(ResponseTemplate::new(200).set_body_string(encoded))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/upload/storage/v1/b/bucket/o"))
        .and(query_param("uploadType", "media"))
        .and(query_param("name", "cache/native"))
        .and(header("authorization", "Bearer token"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({"timestamp": 102.0, "response": response})))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    let lookup = if asynchronous {
        cache
            .async_lookup(&keyed("python"), &DEFAULT, None, Duration::from_secs(102))
            .await
    } else {
        cache.lookup(&keyed("python"), &DEFAULT, None, Duration::from_secs(102))
    };
    assert_eq!(lookup.unwrap(), Some(response.clone()));
    let context = ExactCacheContext {
        ttl: Some(Duration::from_secs(12)),
    };
    let native = keyed("native");
    let stored = if asynchronous {
        cache
            .async_store(&native, context, response, Duration::from_secs(102))
            .await
    } else {
        cache.store(&native, &context, response, Duration::from_secs(102))
    };
    assert_eq!(stored, Ok(()));
    let requests = server.received_requests().await.unwrap();
    let upload = requests
        .iter()
        .find(|request| request.method.as_str() == "POST")
        .unwrap();
    assert_eq!(
        upload.url.query(),
        Some("uploadType=media&name=cache%2Fnative")
    );
}

#[rstest]
#[tokio::test]
async fn gcs_batch_reads_preserve_order_and_treat_invalid_entries_as_misses(
    #[future(awt)] gcs: (wiremock::MockServer, Gcs),
) {
    use wiremock::{
        Mock, ResponseTemplate,
        matchers::{method, path},
    };

    let (server, cache) = gcs;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/cache%2Fhit"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"timestamp": 100.0, "response": {"answer":7}})),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/cache%2Finvalid"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not an entry"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/cache%2Fmissing"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let keys = [keyed("hit"), keyed("missing"), keyed("invalid")];
    let partial = cache
        .async_lookup_batch(&keys, &DEFAULT, None, Duration::from_secs(102))
        .await
        .unwrap();
    assert_eq!(partial.values, vec![Some(json!({"answer":7})), None, None]);
    assert_eq!(partial.missing_indices(), vec![1, 2]);
}

type Gcs = ResponseCache<litellm_cache_gcs::GcsCache<litellm_cache_response::ResponseCacheCodec>>;

#[fixture]
async fn gcs() -> (wiremock::MockServer, Gcs) {
    let server = wiremock::MockServer::start().await;
    let cache = ResponseCache::new(Arc::new(litellm_cache_gcs::GcsCache::with_token_source(
        litellm_cache_gcs::GcsConfig {
            bucket_name: "bucket".into(),
            gcs_path: Some("cache".into()),
            path_service_account: None,
            endpoint: server.uri(),
        },
        litellm_http::Client::plain_for_test(),
        litellm_cache_response::ResponseCacheCodec,
        Arc::new(litellm_cache_gcs::StaticTokenSource("token".into())),
    )));
    (server, cache)
}
