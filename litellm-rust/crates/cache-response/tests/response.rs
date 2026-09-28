mod support;

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::{
    BaseCache, Error, SemanticCacheContext,
    semantic::{SemanticCache, SemanticLookup},
};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheControls, CacheEntry, CacheKeyField, CacheKeyInput, ResponseCache, ResponseCacheRequest,
    WriteBuffer, cache_key,
};
use redis_test::MockCmd;
use rstest::rstest;
use serde_json::{Value, json};
use support::{keyed, memory, redis, request};

type Memory = Arc<ResponseCache<InMemoryCache<CacheEntry>>>;

#[derive(Default)]
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
}

#[rstest]
#[tokio::test]
async fn semantic_context_reaches_backend_for_store_and_lookup(
    request: ResponseCacheRequest,
    #[values(false, true)] asynchronous: bool,
) {
    let backend = Arc::new(SemanticBackend::default());
    let cache = ResponseCache::new(backend.clone());
    let context = SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": "hello"}])),
        ..Default::default()
    };
    let request = request.with_context(context.clone());
    let response = json!({"answer": 42});
    let now = Duration::from_secs(100);

    let hit = if asynchronous {
        cache
            .async_store(&request, response.clone(), now)
            .await
            .unwrap();
        cache.async_lookup(&request, now).await.unwrap()
    } else {
        cache.store(&request, response.clone(), now).unwrap();
        cache.lookup(&request, now).unwrap()
    };

    assert_eq!(hit, Some(response));
    assert_eq!(
        backend.contexts.lock().unwrap().as_slice(),
        &[context.clone(), context]
    );
}

/// A semantic backend that answers every read with one fixed lookup.
struct ScoredBackend(Result<SemanticLookup<CacheEntry>, Error>);

impl BaseCache for ScoredBackend {
    type Value = CacheEntry;
    type Context = SemanticCacheContext;

    fn get_ttl(&self, _: &Self::Context) -> Option<Duration> {
        None
    }

    fn set_cache(&self, _: &str, _: Self::Value, _: &Self::Context) -> Result<(), Error> {
        Ok(())
    }

    fn get_cache(&self, key: &str, context: &Self::Context) -> Result<Option<Self::Value>, Error> {
        self.get_cache_with_similarity(key, context)
            .map(|lookup| lookup.value)
    }
}

impl SemanticCache for ScoredBackend {
    fn get_cache_with_similarity(
        &self,
        _: &str,
        _: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        self.0.clone()
    }

    async fn async_get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error> {
        self.get_cache_with_similarity(key, context)
    }
}

fn scored(timestamp: f64, similarity: f64) -> Result<SemanticLookup<CacheEntry>, Error> {
    Ok(SemanticLookup {
        value: Some(CacheEntry {
            timestamp: Some(timestamp),
            response: json!({"answer": 42}),
        }),
        similarity: Some(similarity),
    })
}

#[rstest]
#[case::fresh_hit(scored(95.0, 0.95), true, Ok(SemanticLookup { value: Some(json!({"answer": 42})), similarity: Some(0.95) }))]
#[case::stale_hit_keeps_the_similarity(
    scored(50.0, 0.95),
    true,
    Ok(SemanticLookup::miss(Some(0.95)))
)]
#[case::miss_keeps_the_similarity(
    Ok(SemanticLookup::miss(Some(0.4))),
    true,
    Ok(SemanticLookup::miss(Some(0.4)))
)]
#[case::no_search(Ok(SemanticLookup::miss(None)), true, Ok(SemanticLookup::miss(None)))]
#[case::disabled_reads_skip_the_backend(scored(95.0, 0.95), false, Ok(SemanticLookup::miss(None)))]
#[case::invalid_entry_is_a_miss(Err(Error::InvalidEntry), true, Ok(SemanticLookup::miss(None)))]
#[case::backend_errors_propagate(Err(Error::Unavailable), true, Err(Error::Unavailable))]
#[tokio::test]
async fn semantic_lookup_applies_freshness_to_the_value_only(
    #[case] backend: Result<SemanticLookup<CacheEntry>, Error>,
    #[case] reads: bool,
    #[case] expected: Result<SemanticLookup<Value>, Error>,
    #[values(false, true)] asynchronous: bool,
    request: ResponseCacheRequest,
) {
    let cache = ResponseCache::new(Arc::new(ScoredBackend(backend)));
    let mut request = request.with_context(SemanticCacheContext::default());
    request.max_age = Some(Duration::from_secs(10));
    request.controls.no_cache = !reads;
    let now = Duration::from_secs(100);

    let lookup = if asynchronous {
        cache.async_lookup_semantic(&request, now).await
    } else {
        cache.lookup_semantic(&request, now)
    };

    assert_eq!(lookup, expected);
}

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

#[derive(Clone, Copy, Debug)]
enum Directive {
    Plain,
    NoCache,
    NoStore,
    DefaultOff,
    UseCache,
    CachingOff,
    Unsupported,
}

impl Directive {
    fn apply(self, controls: &mut CacheControls) {
        match self {
            Self::Plain => {}
            Self::NoCache => controls.no_cache = true,
            Self::NoStore => controls.no_store = true,
            Self::DefaultOff => controls.default_on = false,
            Self::UseCache => {
                controls.default_on = false;
                controls.use_cache = true;
            }
            Self::CachingOff => controls.caching = Some(false),
            Self::Unsupported => controls.supported_call_type = false,
        }
    }
}

#[rstest]
#[case::plain(Directive::Plain, Directive::Plain, true)]
#[case::no_store_skips_the_write(Directive::NoStore, Directive::Plain, false)]
#[case::no_store_keeps_reads(Directive::Plain, Directive::NoStore, true)]
#[case::no_cache_keeps_writes(Directive::NoCache, Directive::Plain, true)]
#[case::no_cache_skips_the_read(Directive::Plain, Directive::NoCache, false)]
#[case::default_off_skips_the_write(Directive::DefaultOff, Directive::Plain, false)]
#[case::default_off_skips_the_read(Directive::Plain, Directive::DefaultOff, false)]
#[case::use_cache_opts_in_under_default_off(Directive::UseCache, Directive::UseCache, true)]
#[case::caching_off_skips_the_write(Directive::CachingOff, Directive::Plain, false)]
#[case::caching_off_skips_the_read(Directive::Plain, Directive::CachingOff, false)]
#[case::unsupported_call_type_skips_the_write(Directive::Unsupported, Directive::Plain, false)]
#[case::unsupported_call_type_skips_the_read(Directive::Plain, Directive::Unsupported, false)]
#[tokio::test]
async fn directives_skip_io_and_keep_reads_and_writes_independent(
    memory: Memory,
    request: ResponseCacheRequest,
    #[case] write: Directive,
    #[case] read: Directive,
    #[case] hit: bool,
    #[values(false, true)] asynchronous: bool,
) {
    let now = Duration::from_secs(100);
    let mut writer = request.clone();
    write.apply(&mut writer.controls);
    let mut reader = request;
    read.apply(&mut reader.controls);

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
    request.controls.no_cache = true;
    assert_eq!(cache.lookup(&request, Duration::ZERO).unwrap(), None);
    request.controls.no_cache = false;
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
#[case::with_namespace(Some("tenant"))]
#[case::without_namespace(None)]
fn generated_keys_preserve_namespace_and_explicit_keys(
    memory: Memory,
    #[case] namespace: Option<&str>,
) {
    let key = CacheKeyInput {
        fields: vec![CacheKeyField {
            name: "model".into(),
            value: Some("a".into()),
            api_parameter: true,
            internal_parameter: false,
        }],
        namespace: namespace.map(str::to_owned),
        ..Default::default()
    };
    let generated = ResponseCacheRequest::new(key.clone());
    let explicit = keyed(&cache_key(&key));
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
#[tokio::test]
async fn batch_lookup_reports_partial_hits_and_batch_store_populates_misses(
    memory: Memory,
    #[values(false, true)] asynchronous: bool,
) {
    let now = Duration::from_secs(100);
    let mut requests = ["hit", "miss", "disabled"].map(keyed).to_vec();
    memory
        .store(&requests[0], json!({"value": 1}), now)
        .unwrap();
    requests[2].controls.caching = Some(false);

    let partial = if asynchronous {
        memory.async_lookup_batch(&requests, now).await.unwrap()
    } else {
        memory.lookup_batch(&requests, now).unwrap()
    };
    assert_eq!(partial.values, vec![Some(json!({"value": 1})), None, None]);
    assert_eq!(partial.missing_indices, vec![1, 2]);

    memory
        .async_store_batch(
            vec![
                (requests[1].clone(), json!({"value": 2})),
                (requests[2].clone(), json!({"value": 3})),
            ],
            now,
        )
        .await
        .unwrap();
    assert_eq!(
        memory.lookup(&requests[1], now).unwrap(),
        Some(json!({"value": 2}))
    );
    requests[2].controls.caching = None;
    assert_eq!(memory.lookup(&requests[2], now).unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn batch_lookup_with_no_readable_request_skips_the_backend(
    #[values(false, true)] asynchronous: bool,
) {
    let cache = redis(Vec::new(), None);
    let mut request = keyed("key");
    request.controls.no_cache = true;
    let requests = [request.clone(), request];
    let partial = if asynchronous {
        cache
            .async_lookup_batch(&requests, Duration::ZERO)
            .await
            .unwrap()
    } else {
        cache.lookup_batch(&requests, Duration::ZERO).unwrap()
    };
    assert_eq!(partial.values, vec![None, None]);
    assert_eq!(partial.missing_indices, vec![0, 1]);
}

#[rstest]
#[tokio::test]
async fn deferred_entries_keep_the_time_they_were_produced(
    memory: Memory,
    mut request: ResponseCacheRequest,
) {
    request.max_age = Some(Duration::from_secs(10));
    memory
        .async_store_entries(vec![(
            request.clone(),
            json!({"answer": 7}),
            Duration::from_secs(100),
        )])
        .await
        .unwrap();

    assert_eq!(
        memory.lookup(&request, Duration::from_secs(110)).unwrap(),
        Some(json!({"answer": 7}))
    );
    assert_eq!(
        memory.lookup(&request, Duration::from_secs(111)).unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn write_buffer_flushes_at_its_size_and_keeps_each_produced_time(
    memory: Memory,
    request: ResponseCacheRequest,
) {
    let buffer = WriteBuffer::new(2);
    let mut first = request;
    first.max_age = Some(Duration::from_secs(10));
    let second = keyed("tenant:other");

    buffer
        .async_store(
            memory.as_ref(),
            &first,
            json!({"answer": 7}),
            Duration::from_secs(100),
        )
        .await
        .unwrap();
    assert_eq!(
        memory.lookup(&first, Duration::from_secs(100)).unwrap(),
        None
    );

    buffer
        .async_store(
            memory.as_ref(),
            &second,
            json!({"answer": 8}),
            Duration::from_secs(200),
        )
        .await
        .unwrap();
    assert_eq!(
        memory.lookup(&first, Duration::from_secs(110)).unwrap(),
        Some(json!({"answer": 7}))
    );
    assert_eq!(
        memory.lookup(&first, Duration::from_secs(111)).unwrap(),
        None
    );
    assert_eq!(
        memory.lookup(&second, Duration::from_secs(200)).unwrap(),
        Some(json!({"answer": 8}))
    );
}

#[rstest]
#[tokio::test]
async fn write_buffer_clear_drops_pending_entries(memory: Memory, request: ResponseCacheRequest) {
    let buffer = WriteBuffer::new(2);
    let other = keyed("tenant:other");
    let now = Duration::from_secs(100);

    buffer
        .async_store(memory.as_ref(), &request, json!({"answer": 7}), now)
        .await
        .unwrap();
    buffer.clear().unwrap();
    buffer
        .async_store(memory.as_ref(), &other, json!({"answer": 8}), now)
        .await
        .unwrap();

    assert_eq!(memory.lookup(&request, now).unwrap(), None);
    assert_eq!(memory.lookup(&other, now).unwrap(), None);
}
