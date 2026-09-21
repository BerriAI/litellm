use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::{BaseCache, CacheCodec, CacheEntry, CacheKeyField, CacheKeyInput, Error};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::RedisCache;
use litellm_cache_response::{
    NativeResponseCache, ResponseCache, ResponseCacheCodec, ResponseCacheRequest,
};
use redis_test::{MockCmd, MockRedisConnection};
use serde_json::json;

fn request() -> ResponseCacheRequest {
    ResponseCacheRequest::new(CacheKeyInput {
        preset: Some("tenant:key".into()),
        ..Default::default()
    })
}

#[tokio::test]
async fn sync_and_async_consumers_share_keys_ttls_and_freshness() {
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
    let mut request = request();
    request.kwargs.ttl = Some(Duration::from_secs(10));
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

#[tokio::test]
async fn directives_skip_io_and_keep_reads_and_writes_independent() {
    let cache = NativeResponseCache::memory(8, Duration::from_secs(600), 1024);
    let mut request = request();
    let now = Duration::from_secs(100);
    request.controls.no_store = true;
    cache
        .async_store(&request, json!({"v": 1}), now)
        .await
        .unwrap();
    assert_eq!(cache.lookup(&request, now).unwrap(), None);
    request.controls.no_store = false;
    request.controls.no_cache = true;
    cache.store(&request, json!({"v": 2}), now).unwrap();
    assert_eq!(cache.async_lookup(&request, now).await.unwrap(), None);
    request.controls.no_cache = false;
    assert_eq!(cache.lookup(&request, now).unwrap(), Some(json!({"v": 2})));
    request.controls.default_on = false;
    cache.store(&request, json!({"v": 3}), now).unwrap();
    assert_eq!(cache.lookup(&request, now).unwrap(), None);
    request.controls.use_cache = true;
    assert_eq!(cache.lookup(&request, now).unwrap(), Some(json!({"v": 2})));
    request.controls.supported_call_type = false;
    assert_eq!(cache.lookup(&request, now).unwrap(), None);
}

#[tokio::test]
async fn redis_enum_reads_python_sync_and_async_envelopes_and_writes_compatible_json() {
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("GET").arg("tenant:key"),
            Ok(br#"{'timestamp': 100.0, 'response': '{"ok": true, "text": "cached"}'}"#.to_vec()),
        ),
        MockCmd::new(
            redis::cmd("GET").arg("tenant:key"),
            Ok(br#"{"timestamp":100.0,"response":{"ok":true,"text":"cached"}}"#.to_vec()),
        ),
        MockCmd::new(
            redis::cmd("SETEX")
                .arg("tenant:key")
                .arg(600)
                .arg(br#"{"timestamp":100.0,"response":{"ok":true,"text":"cached"}}"#.as_slice()),
            Ok("OK"),
        ),
    ])
    .assert_all_commands_consumed();
    let backend = RedisCache::with_connection(connection, None, ResponseCacheCodec)
        .with_namespace(Some("tenant".into()));
    let cache = NativeResponseCache::Redis(Arc::new(ResponseCache::new(Arc::new(backend))));
    let request = request();
    let expected = json!({"ok": true, "text": "cached"});
    assert_eq!(
        cache.lookup(&request, Duration::from_secs(101)).unwrap(),
        Some(expected.clone())
    );
    assert_eq!(
        cache
            .async_lookup(&request, Duration::from_secs(101))
            .await
            .unwrap(),
        Some(expected.clone())
    );
    cache
        .async_store(&request, expected, Duration::from_secs(100))
        .await
        .unwrap();
}

#[tokio::test]
async fn captured_enum_keeps_the_selected_backend_for_background_writes() {
    let original = NativeResponseCache::memory(8, Duration::from_secs(600), 1024);
    let captured = original.clone();
    let replacement = NativeResponseCache::memory(8, Duration::from_secs(600), 1024);
    let request = request();
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

#[test]
fn generated_keys_preserve_namespace_and_explicit_keys() {
    let cache = NativeResponseCache::memory(8, Duration::from_secs(600), 1024);
    let key = CacheKeyInput {
        fields: vec![CacheKeyField {
            name: "model".into(),
            value: Some("a".into()),
            api_parameter: true,
            internal_parameter: false,
        }],
        namespace: Some("tenant".into()),
        ..Default::default()
    };
    let generated = ResponseCacheRequest::new(key.clone());
    let explicit = ResponseCacheRequest::new(CacheKeyInput {
        preset: Some(litellm_cache::cache_key(&key)),
        ..Default::default()
    });
    cache
        .store(&generated, json!({"value": 7}), Duration::from_secs(100))
        .unwrap();
    assert_eq!(
        cache.lookup(&explicit, Duration::from_secs(100)).unwrap(),
        Some(json!({"value":7}))
    );
}

#[test]
fn response_codec_accepts_python_literals_without_executing_code() {
    let bytes = br#"{'timestamp': 100.0, 'response': {'text': 'hello \\ world', 'flag': True, 'empty': None, 'list': [1, 2.5]}}"#;
    let entry = ResponseCacheCodec.decode(bytes).unwrap();
    assert_eq!(
        entry.response,
        json!({"text": "hello \\ world", "flag": true, "empty": null, "list": [1, 2.5]})
    );
    for bytes in [
        b"__import__('os').system('false')".as_slice(),
        b"{'timestamp': 'invalid', 'response': {}}",
        b"{'timestamp': 1e9999, 'response': {}}",
    ] {
        assert_eq!(
            ResponseCacheCodec.decode(bytes).unwrap_err(),
            Error::InvalidEntry
        );
    }
    let deep = format!("{}None{}", "[".repeat(1000), "]".repeat(1000));
    assert_eq!(
        ResponseCacheCodec.decode(deep.as_bytes()).unwrap_err(),
        Error::InvalidEntry
    );
    assert_eq!(
        ResponseCacheCodec
            .encode(&CacheEntry {
                timestamp: f64::NAN,
                response: json!({})
            })
            .unwrap_err(),
        Error::InvalidEntry
    );
}

#[tokio::test]
async fn backend_failures_remain_observable_and_disabled_reads_do_not_touch_redis() {
    let connection = MockRedisConnection::new([MockCmd::new(
        redis::cmd("GET").arg("tenant:key"),
        Ok(b"invalid".to_vec()),
    )])
    .assert_all_commands_consumed();
    let backend = RedisCache::with_connection(connection, None, ResponseCacheCodec);
    let cache = ResponseCache::new(Arc::new(backend));
    let mut request = request();
    request.controls.no_cache = true;
    assert_eq!(cache.lookup(&request, Duration::ZERO).unwrap(), None);
    request.controls.no_cache = false;
    assert_eq!(
        cache
            .async_lookup(&request, Duration::ZERO)
            .await
            .unwrap_err(),
        Error::InvalidEntry
    );
}

#[test]
fn malformed_memory_entries_are_rejected_by_the_response_consumer() {
    let backend = Arc::new(InMemoryCache::default());
    BaseCache::set_cache(
        backend.as_ref(),
        "tenant:key",
        CacheEntry {
            timestamp: 100.0,
            response: json!("not a serialized response"),
        },
        Default::default(),
    )
    .unwrap();
    let cache = ResponseCache::new(backend);
    assert_eq!(
        cache
            .lookup(&request(), Duration::from_secs(100))
            .unwrap_err(),
        Error::InvalidEntry
    );
}
