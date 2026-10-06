mod support;

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::BaseCache;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheAccess, CacheEntry, CacheKeyInput, CacheTarget, ResponseCache, ResponseCacheConfig,
    ResponseCacheRequest,
};
use redis_test::MockCmd;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::{keyed, memory, redis, request};

type Memory = Arc<ResponseCache<InMemoryCache<CacheEntry>>>;

#[rstest]
#[tokio::test]
async fn sync_and_async_consumers_share_keys_ttls_and_freshness(mut request: ResponseCacheRequest) {
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
    request.context.ttl = Some(Duration::from_secs(10));
    request.max_age = Some(Duration::from_secs(5));
    cache
        .store(
            &request,
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
            .async_lookup(&request, Duration::from_secs(105))
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        cache.lookup(&request, Duration::from_secs(106)).unwrap(),
        None
    );
    request.max_age = None;
    assert_eq!(
        cache
            .lookup(&request, Duration::from_secs(106))
            .unwrap()
            .unwrap()["usage"]["total_tokens"],
        7
    );
    clock.store(111, Ordering::SeqCst);
    assert_eq!(
        cache
            .async_lookup(&request, Duration::from_secs(111))
            .await
            .unwrap(),
        None
    );
    cache
        .async_store(&request, json!({"choices": [2]}), Duration::from_secs(111))
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&request, Duration::from_secs(111)).unwrap(),
        Some(json!({"choices": [2]}))
    );
}

const WRITE_ONLY: CacheAccess = CacheAccess {
    reads: false,
    writes: true,
};
const READ_ONLY: CacheAccess = CacheAccess {
    reads: true,
    writes: false,
};

#[rstest]
#[case::read_write(CacheAccess::READ_WRITE, CacheAccess::READ_WRITE, true)]
#[case::read_only_writer_skips_the_write(READ_ONLY, CacheAccess::READ_WRITE, false)]
#[case::read_only_reader_still_reads(CacheAccess::READ_WRITE, READ_ONLY, true)]
#[case::write_only_writer_still_writes(WRITE_ONLY, CacheAccess::READ_WRITE, true)]
#[case::write_only_reader_skips_the_read(CacheAccess::READ_WRITE, WRITE_ONLY, false)]
#[case::no_access_writer_skips_the_write(CacheAccess::NONE, CacheAccess::READ_WRITE, false)]
#[case::no_access_reader_skips_the_read(CacheAccess::READ_WRITE, CacheAccess::NONE, false)]
#[tokio::test]
async fn access_skips_io_and_keeps_reads_and_writes_independent(
    memory: Memory,
    request: ResponseCacheRequest,
    #[case] write: CacheAccess,
    #[case] read: CacheAccess,
    #[case] hit: bool,
    #[values(false, true)] asynchronous: bool,
) {
    let now = Duration::from_secs(100);
    let writer = ResponseCacheRequest {
        access: write,
        ..request.clone()
    };
    let reader = ResponseCacheRequest {
        access: read,
        ..request
    };

    if asynchronous {
        memory
            .async_store(&writer, json!({"v": 1}), now)
            .await
            .unwrap();
    } else {
        memory.store(&writer, json!({"v": 1}), now).unwrap();
    }
    let found = if asynchronous {
        memory.async_lookup(&reader, now).await.unwrap()
    } else {
        memory.lookup(&reader, now).unwrap()
    };

    assert_eq!(
        found,
        hit.then(|| json!({"v": 1})),
        "{write:?} then {read:?}"
    );
}

#[rstest]
#[case::python_sync_literal(
    br#"{'timestamp': 100.0, 'response': '{"ok": true, "text": "cached"}'}"#.as_slice()
)]
#[case::python_async_json(br#"{"timestamp":100.0,"response":{"ok":true,"text":"cached"}}"#.as_slice())]
#[tokio::test]
async fn redis_consumer_reads_python_sync_and_async_envelopes(
    request: ResponseCacheRequest,
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
        cache.async_lookup(&request, now).await.unwrap()
    } else {
        cache.lookup(&request, now).unwrap()
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
    request: ResponseCacheRequest,
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
        .async_store(&request, response, Duration::from_secs(100))
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn invalid_entries_are_misses_and_disabled_reads_do_not_touch_redis(
    mut request: ResponseCacheRequest,
) {
    let cache = redis(
        vec![MockCmd::new(
            redis::cmd("GET").arg("tenant:key"),
            Ok(b"invalid".to_vec()),
        )],
        None,
    );
    request.access.reads = false;
    assert_eq!(cache.lookup(&request, Duration::ZERO).unwrap(), None);
    request.access.reads = true;
    assert_eq!(
        cache.async_lookup(&request, Duration::ZERO).await.unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn captured_service_keeps_the_selected_backend_for_background_writes(
    #[from(memory)] original: Memory,
    #[from(memory)] replacement: Memory,
    request: ResponseCacheRequest,
) {
    let captured = original.clone();
    let writer = tokio::spawn({
        let request = request.clone();
        async move {
            captured
                .async_store(
                    &request,
                    json!({"selected": "original"}),
                    Duration::from_secs(100),
                )
                .await
        }
    });
    writer.await.unwrap().unwrap();
    assert_eq!(
        original.lookup(&request, Duration::from_secs(100)).unwrap(),
        Some(json!({"selected":"original"}))
    );
    assert_eq!(
        replacement
            .lookup(&request, Duration::from_secs(100))
            .unwrap(),
        None
    );
}

#[rstest]
#[case::with_namespace("tenant")]
#[case::without_namespace("")]
fn generated_keys_preserve_namespace_and_explicit_keys(#[case] namespace: &str) {
    let memory = ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default())).with_config(
        ResponseCacheConfig {
            namespace: namespace.into(),
            ..ResponseCacheConfig::default()
        },
    );
    let generated = ResponseCacheRequest::new(CacheKeyInput::request(
        CacheTarget::resolve(None, "a", None, None),
        json!({}),
    ));
    let explicit = keyed(memory.key(&generated).as_str());
    memory
        .store(&generated, json!({"value": 7}), Duration::from_secs(100))
        .unwrap();
    assert_eq!(
        memory.lookup(&explicit, Duration::from_secs(100)).unwrap(),
        Some(json!({"value":7}))
    );
}

#[rstest]
#[case::text(json!("hello world"))]
#[case::numeric_text(json!("123"))]
#[case::null_text(json!("null"))]
#[case::array(json!([1, 2]))]
fn non_object_responses_round_trip_through_a_typed_backend(
    memory: Memory,
    request: ResponseCacheRequest,
    #[case] response: Value,
) {
    let now = Duration::from_secs(100);
    memory.store(&request, response.clone(), now).unwrap();
    assert_eq!(memory.lookup(&request, now).unwrap(), Some(response));
}

#[rstest]
fn entries_without_timestamps_are_always_fresh(request: ResponseCacheRequest) {
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
    let mut request = request;
    request.max_age = Some(Duration::from_secs(1));
    assert_eq!(
        cache.lookup(&request, Duration::from_secs(100)).unwrap(),
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
            .async_lookup(&keyed("python"), Duration::from_secs(102))
            .await
    } else {
        cache.lookup(&keyed("python"), Duration::from_secs(102))
    };
    assert_eq!(lookup.unwrap(), Some(response.clone()));
    let request = ResponseCacheRequest {
        context: litellm_cache::ExactCacheContext {
            ttl: Some(Duration::from_secs(12)),
        },
        ..keyed("native")
    };
    let stored = if asynchronous {
        cache
            .async_store(&request, response, Duration::from_secs(102))
            .await
    } else {
        cache.store(&request, response, Duration::from_secs(102))
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
    let requests = [keyed("hit"), keyed("missing"), keyed("invalid")];
    let partial = cache
        .async_lookup_batch(&requests, Duration::from_secs(102))
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
