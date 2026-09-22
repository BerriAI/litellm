use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::{
    BaseCache, CacheCodec, CacheConnectionResult, CacheConnectionStatus, Error,
    SemanticCacheContext,
};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::RedisCache;
use litellm_cache_response::{
    CacheEntry, CacheKeyField, CacheKeyInput, ResponseCache, ResponseCacheCodec,
    ResponseCacheRequest, WriteBuffer,
};
use redis_test::{MockCmd, MockRedisConnection};
use serde_json::json;

fn memory() -> Arc<ResponseCache<InMemoryCache<CacheEntry>>> {
    Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
        Some(8),
        Some(Duration::from_secs(600)),
    ))))
}

fn request() -> ResponseCacheRequest {
    ResponseCacheRequest::new(CacheKeyInput {
        preset: Some("tenant:key".into()),
        ..Default::default()
    })
}

struct SemanticBackend {
    entries: Mutex<Vec<(String, CacheEntry)>>,
    contexts: Mutex<Vec<SemanticCacheContext>>,
}

impl BaseCache for SemanticBackend {
    type Value = CacheEntry;
    type Context = SemanticCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
        None
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &Self::Context,
    ) -> Result<(), Error> {
        self.contexts.lock().unwrap().push(context.clone());
        self.entries.lock().unwrap().push((key.to_owned(), value));
        Ok(())
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        self.contexts.lock().unwrap().push(context.clone());
        Ok(self
            .entries
            .lock()
            .unwrap()
            .iter()
            .find(|(entry_key, _)| entry_key == key)
            .map(|(_, entry)| entry.clone()))
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        Ok(CacheConnectionResult {
            status: CacheConnectionStatus::Success,
            message: "ok".into(),
            error: None,
        })
    }
}

#[test]
fn semantic_context_reaches_backend_for_store_and_lookup() {
    let backend = Arc::new(SemanticBackend {
        entries: Mutex::new(Vec::new()),
        contexts: Mutex::new(Vec::new()),
    });
    let cache = ResponseCache::new(backend.clone());
    let context = SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": "hello"}])),
        ..Default::default()
    };
    let request = request().with_context(context.clone());
    let response = json!({"answer": 42});

    cache
        .store(&request, response.clone(), Duration::from_secs(100))
        .unwrap();

    assert_eq!(
        cache.lookup(&request, Duration::from_secs(100)).unwrap(),
        Some(response)
    );
    assert_eq!(
        backend.contexts.lock().unwrap().as_slice(),
        &[context.clone(), context]
    );
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

#[tokio::test]
async fn directives_skip_io_and_keep_reads_and_writes_independent() {
    let cache = memory();
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
async fn redis_consumer_reads_python_sync_and_async_envelopes_and_writes_compatible_json() {
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
    let cache = ResponseCache::new(Arc::new(backend));
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
async fn captured_service_keeps_the_selected_backend_for_background_writes() {
    let original = memory();
    let captured = original.clone();
    let replacement = memory();
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
    let cache = memory();
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
        preset: Some(litellm_cache_response::cache_key(&key)),
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
                timestamp: Some(f64::NAN),
                response: json!({})
            })
            .unwrap_err(),
        Error::InvalidEntry
    );
}

#[tokio::test]
async fn invalid_entries_are_misses_and_disabled_reads_do_not_touch_redis() {
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
        cache.async_lookup(&request, Duration::ZERO).await.unwrap(),
        None
    );
}

#[test]
fn string_responses_round_trip_through_typed_and_wire_backends() {
    let cache = ResponseCache::new(Arc::new(InMemoryCache::default()));
    let now = Duration::from_secs(100);
    for response in [json!("hello world"), json!("123"), json!("null")] {
        cache.store(&request(), response.clone(), now).unwrap();
        assert_eq!(
            cache.lookup(&request(), now).unwrap(),
            Some(response.clone())
        );

        let wire = ResponseCacheCodec
            .encode(&CacheEntry {
                timestamp: Some(100.0),
                response: response.clone(),
            })
            .unwrap();
        assert_eq!(ResponseCacheCodec.decode(&wire).unwrap().response, response);
    }
}

#[test]
fn non_object_responses_are_written_as_python_readable_serialized_strings() {
    let wire = ResponseCacheCodec
        .encode(&CacheEntry {
            timestamp: Some(100.0),
            response: json!([1, 2]),
        })
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&wire).unwrap(),
        json!({"timestamp": 100.0, "response": "[1,2]"})
    );
    assert_eq!(
        ResponseCacheCodec.decode(&wire).unwrap().response,
        json!([1, 2])
    );
    assert_eq!(
        ResponseCacheCodec.decode(br#"{"timestamp": 100.0, "response": "not serialized"}"#),
        Err(Error::InvalidEntry)
    );
}

#[test]
fn response_entries_preserve_the_existing_json_representation() {
    let codec = ResponseCacheCodec;
    let entry = CacheEntry {
        timestamp: Some(123.0),
        response: json!({"choices": [{"text": "cached"}]}),
    };
    let bytes = codec.encode(&entry).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&entry).unwrap());
    assert_eq!(codec.decode(&bytes).unwrap(), entry);
}

#[test]
fn response_codec_preserves_values_without_timestamps() {
    let codec = ResponseCacheCodec;
    let raw = json!({"choices": [{"text": "legacy"}]});
    let entry = codec.decode(&serde_json::to_vec(&raw).unwrap()).unwrap();
    assert_eq!(entry.timestamp, None);
    assert_eq!(entry.response, raw);

    let backend = Arc::new(InMemoryCache::default());
    BaseCache::set_cache(backend.as_ref(), "tenant:key", entry, &Default::default()).unwrap();
    let cache = ResponseCache::new(backend);
    assert_eq!(
        cache.lookup(&request(), Duration::from_secs(100)).unwrap(),
        Some(json!({"choices": [{"text": "legacy"}]}))
    );
}

#[tokio::test]
async fn batch_lookup_reports_partial_hits_and_batch_store_populates_misses() {
    let cache = memory();
    let requests = ["hit", "miss", "disabled"].map(|key| {
        ResponseCacheRequest::new(CacheKeyInput {
            preset: Some(key.into()),
            ..Default::default()
        })
    });
    cache
        .store(&requests[0], json!({"value": 1}), Duration::from_secs(100))
        .unwrap();
    let mut requests = requests.to_vec();
    requests[2].controls.caching = Some(false);

    let partial = cache
        .async_lookup_batch(&requests, Duration::from_secs(100))
        .await
        .unwrap();
    assert_eq!(partial.values, vec![Some(json!({"value": 1})), None, None]);
    assert_eq!(partial.missing_indices, vec![1, 2]);

    cache
        .async_store_batch(
            vec![
                (requests[1].clone(), json!({"value": 2})),
                (requests[2].clone(), json!({"value": 3})),
            ],
            Duration::from_secs(100),
        )
        .await
        .unwrap();
    assert_eq!(
        cache
            .lookup(&requests[1], Duration::from_secs(100))
            .unwrap(),
        Some(json!({"value": 2}))
    );
    requests[2].controls.caching = None;
    assert_eq!(
        cache
            .lookup(&requests[2], Duration::from_secs(100))
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn deferred_entries_keep_the_time_they_were_produced() {
    let cache = ResponseCache::new(Arc::new(InMemoryCache::default()));
    let mut request = request();
    request.max_age = Some(Duration::from_secs(10));
    cache
        .async_store_entries(vec![(
            request.clone(),
            json!({"answer": 7}),
            Duration::from_secs(100),
        )])
        .await
        .unwrap();

    assert_eq!(
        cache.lookup(&request, Duration::from_secs(110)).unwrap(),
        Some(json!({"answer": 7}))
    );
    assert_eq!(
        cache.lookup(&request, Duration::from_secs(111)).unwrap(),
        None
    );
}

#[tokio::test]
async fn write_buffer_flushes_at_its_size_and_keeps_each_produced_time() {
    let cache = ResponseCache::new(Arc::new(InMemoryCache::default()));
    let buffer = WriteBuffer::new(2);
    let mut first = request();
    first.max_age = Some(Duration::from_secs(10));
    let mut second = request();
    second.key.preset = Some("tenant:other".into());

    buffer
        .async_store(
            &cache,
            &first,
            json!({"answer": 7}),
            Duration::from_secs(100),
        )
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&first, Duration::from_secs(100)).unwrap(),
        None
    );

    buffer
        .async_store(
            &cache,
            &second,
            json!({"answer": 8}),
            Duration::from_secs(200),
        )
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&first, Duration::from_secs(110)).unwrap(),
        Some(json!({"answer": 7}))
    );
    assert_eq!(
        cache.lookup(&first, Duration::from_secs(111)).unwrap(),
        None
    );
    assert_eq!(
        cache.lookup(&second, Duration::from_secs(200)).unwrap(),
        Some(json!({"answer": 8}))
    );
}

#[tokio::test]
async fn write_buffer_clear_drops_pending_entries() {
    let cache = ResponseCache::new(Arc::new(InMemoryCache::default()));
    let buffer = WriteBuffer::new(2);
    let mut other = request();
    other.key.preset = Some("tenant:other".into());
    let now = Duration::from_secs(100);

    buffer
        .async_store(&cache, &request(), json!({"answer": 7}), now)
        .await
        .unwrap();
    buffer.clear().unwrap();
    buffer
        .async_store(&cache, &other, json!({"answer": 8}), now)
        .await
        .unwrap();

    assert_eq!(cache.lookup(&request(), now).unwrap(), None);
    assert_eq!(cache.lookup(&other, now).unwrap(), None);
}
