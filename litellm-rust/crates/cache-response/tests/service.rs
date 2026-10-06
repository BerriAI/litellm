use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use futures_util::future::BoxFuture;
use litellm_cache::{CacheCodec, Error};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::RedisCache;
use litellm_cache_response::{
    CacheEntry, CacheKey, CacheKeyInput, CacheScope, CacheTarget, ResponseCache,
    ResponseCacheCodec, ResponseCacheConfig, ResponseCacheService,
};
use redis_test::{MockCmd, MockRedisConnection};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

fn input(prompt: &str) -> CacheKeyInput {
    CacheKeyInput::new(
        "responses",
        CacheTarget::ModelGroup("group".into()),
        json!({"input": prompt}),
    )
}

fn keyed(key: &str) -> CacheKey {
    CacheKey::External(key.into())
}

#[rstest]
#[tokio::test]
async fn service_honors_per_call_expiry_and_freshness() {
    let clock = Arc::new(AtomicU64::new(0));
    let cache_clock = clock.clone();
    let cache: Arc<dyn ResponseCacheService> = Arc::new(ResponseCache::new(Arc::new(
        InMemoryCache::with_clock(Some(100), Some(Duration::from_secs(60)), move || {
            Duration::from_secs(cache_clock.load(Ordering::SeqCst))
        }),
    )));
    let key = cache
        .key(&input("entry"), &CacheScope::Shared)
        .await
        .unwrap();
    cache
        .store(
            &key,
            Some(Duration::from_secs(5)),
            json!({"answer":7}),
            Duration::ZERO,
        )
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&key, None, Duration::ZERO).await.unwrap(),
        Some(json!({"answer":7}))
    );
    clock.store(2, Ordering::SeqCst);
    assert_eq!(
        cache
            .lookup(&key, Some(Duration::from_secs(1)), Duration::from_secs(2))
            .await
            .unwrap(),
        None
    );
    assert!(
        cache
            .lookup(&key, None, Duration::from_secs(2))
            .await
            .unwrap()
            .is_some()
    );
    clock.store(6, Ordering::SeqCst);
    assert_eq!(
        cache
            .lookup(&key, None, Duration::from_secs(6))
            .await
            .unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn native_service_reads_and_writes_under_the_key_it_is_given() {
    let cache: Arc<dyn ResponseCacheService> = Arc::new(ResponseCache::new(Arc::new(
        InMemoryCache::<CacheEntry>::default(),
    )));
    let key = cache
        .key(&input("stored"), &CacheScope::Shared)
        .await
        .unwrap();
    cache
        .store(&key, None, json!({"answer":7}), Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&key, None, Duration::ZERO).await.unwrap(),
        Some(json!({"answer":7}))
    );
    let other_key = cache
        .key(&input("other"), &CacheScope::Shared)
        .await
        .unwrap();
    assert_ne!(other_key, key);
    assert_eq!(
        cache
            .lookup(&other_key, None, Duration::ZERO)
            .await
            .unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn entry_limit_applies_to_sync_async_and_batch_writes() {
    let storage = Arc::new(InMemoryCache::<CacheEntry>::default());
    let cache = ResponseCache::new(storage.clone()).with_config(ResponseCacheConfig {
        namespace: "service-test".into(),
        max_entry_bytes: json!({"answer":7}).to_string().len(),
    });
    let small = json!({"answer":7});
    let large = json!({"answer":"too large"});
    let context = litellm_cache::ExactCacheContext::default();
    cache
        .store(&keyed("sync"), &context, large.clone(), Duration::ZERO)
        .unwrap();
    cache
        .async_store(
            &keyed("async"),
            context.clone(),
            large.clone(),
            Duration::ZERO,
        )
        .await
        .unwrap();
    cache
        .async_store_batch(
            vec![
                (keyed("batch-large"), large),
                (keyed("batch-small"), small.clone()),
            ],
            context,
            Duration::ZERO,
        )
        .await
        .unwrap();
    let service: Arc<dyn ResponseCacheService> = Arc::new(cache);
    service
        .store(&keyed("service"), None, small.clone(), Duration::ZERO)
        .await
        .unwrap();
    for key in ["sync", "async", "batch-large"] {
        assert!(storage.get_cache(key).unwrap().is_none());
    }
    for key in ["batch-small", "service"] {
        assert_eq!(
            service
                .lookup(&keyed(key), None, Duration::ZERO)
                .await
                .unwrap(),
            Some(small.clone())
        );
    }
}

struct SingleLookupService {
    config: ResponseCacheConfig,
}

impl ResponseCacheService for SingleLookupService {
    fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    fn key<'a>(
        &'a self,
        _: &'a CacheKeyInput,
        _: &'a CacheScope,
    ) -> BoxFuture<'a, Result<CacheKey, Error>> {
        Box::pin(async { Err(Error::Unavailable) })
    }

    fn lookup<'a>(
        &'a self,
        key: &'a CacheKey,
        _: Option<Duration>,
        now: Duration,
    ) -> BoxFuture<'a, Result<Option<Value>, Error>> {
        Box::pin(async move {
            if now != Duration::from_secs(100) {
                return Err(Error::Unavailable);
            }
            match key.as_str() {
                "hit" => Ok(Some(json!({"answer": 7}))),
                "failure" => Err(Error::Unavailable),
                _ => Ok(None),
            }
        })
    }

    fn store<'a>(
        &'a self,
        _: &'a CacheKey,
        _: Option<Duration>,
        _: Value,
        _: Duration,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }
}

#[fixture]
fn service() -> Arc<dyn ResponseCacheService> {
    Arc::new(SingleLookupService {
        config: ResponseCacheConfig::default(),
    })
}

#[rstest]
#[case::empty(&[], &[])]
#[case::all_hits(&["hit", "hit"], &[])]
#[case::all_misses(&["miss", "miss"], &[0, 1])]
#[case::mixed_duplicates(&["hit", "miss", "hit"], &[1])]
#[tokio::test]
async fn batch_fallback_preserves_single_lookup_results(
    service: Arc<dyn ResponseCacheService>,
    #[case] names: &[&str],
    #[case] missing: &[usize],
) {
    let keys: Vec<_> = names.iter().map(|name| keyed(name)).collect();
    let now = Duration::from_secs(100);
    let batch = service.lookup_batch(&keys, None, now).await.unwrap();
    let expected: Vec<_> =
        futures_util::future::try_join_all(keys.iter().map(|key| service.lookup(key, None, now)))
            .await
            .unwrap();
    assert_eq!(batch.values, expected);
    assert_eq!(batch.missing_indices(), missing);
    assert_eq!(
        serde_json::to_value(&batch).unwrap(),
        json!({"values": expected, "missing_indices": missing}),
    );
}

#[rstest]
#[tokio::test]
async fn batch_fallback_preserves_storage_failures(service: Arc<dyn ResponseCacheService>) {
    let result = service
        .lookup_batch(
            &[keyed("hit"), keyed("failure")],
            None,
            Duration::from_secs(100),
        )
        .await;
    assert_eq!(result, Err(Error::Unavailable));
}

#[rstest]
#[case::fresh(None, true)]
#[case::stale(Some(Duration::from_secs(1)), false)]
#[tokio::test]
async fn response_service_uses_bulk_reads_and_applies_freshness(
    #[case] max_age: Option<Duration>,
    #[case] fresh: bool,
) {
    let response = json!({"answer": 7});
    let encoded = ResponseCacheCodec
        .encode(&CacheEntry::produced_at(
            response.clone(),
            Duration::from_secs(100),
        ))
        .unwrap();
    let connection = MockRedisConnection::new(vec![MockCmd::new(
        redis::cmd("MGET").arg(["hit", "miss", "hit"].as_slice()),
        Ok(vec![
            redis::Value::BulkString(encoded.clone()),
            redis::Value::Nil,
            redis::Value::BulkString(encoded),
        ]),
    )])
    .assert_all_commands_consumed();
    let cache: Arc<dyn ResponseCacheService> = Arc::new(ResponseCache::new(Arc::new(
        RedisCache::with_connection(connection, None, ResponseCacheCodec),
    )));
    let batch = cache
        .lookup_batch(
            &[keyed("hit"), keyed("miss"), keyed("hit")],
            max_age,
            Duration::from_secs(102),
        )
        .await
        .unwrap();
    let hit = fresh.then(|| response.clone());
    assert_eq!(batch.values, vec![hit.clone(), None, hit]);
}
