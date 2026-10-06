use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use futures_util::future::BoxFuture;
use litellm_cache::{CacheCodec, Error, ExactCacheContext};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::RedisCache;
use litellm_cache_response::{
    CacheAccess, CacheEntry, CacheKey, CacheKeyInput, ResponseCache, ResponseCacheCodec,
    ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService,
};
use redis_test::{MockCmd, MockRedisConnection};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

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
    let request = ResponseCacheRequest {
        context: ExactCacheContext {
            ttl: Some(Duration::from_secs(5)),
        },
        ..ResponseCacheRequest::new(CacheKeyInput {
            preset: Some("entry".into()),
            ..Default::default()
        })
    };
    let key = cache.key(&request).await.unwrap();
    cache
        .store(&key, &request, json!({"answer":7}), Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&key, &request, Duration::ZERO).await.unwrap(),
        Some(json!({"answer":7}))
    );
    let stale_request = ResponseCacheRequest {
        max_age: Some(Duration::from_secs(1)),
        ..request.clone()
    };
    clock.store(2, Ordering::SeqCst);
    assert_eq!(
        cache
            .lookup(&key, &stale_request, Duration::from_secs(2))
            .await
            .unwrap(),
        None
    );
    assert!(
        cache
            .lookup(&key, &request, Duration::from_secs(2))
            .await
            .unwrap()
            .is_some()
    );
    clock.store(6, Ordering::SeqCst);
    assert_eq!(
        cache
            .lookup(&key, &request, Duration::from_secs(6))
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
    let stored = request("stored");
    let other = request("other");
    let key = cache.key(&stored).await.unwrap();
    cache
        .store(&key, &stored, json!({"answer":7}), Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&key, &other, Duration::ZERO).await.unwrap(),
        Some(json!({"answer":7}))
    );
    let other_key = cache.key(&other).await.unwrap();
    assert_ne!(other_key, key);
    assert_eq!(
        cache
            .lookup(&other_key, &stored, Duration::ZERO)
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
    let request = |key: &str| {
        ResponseCacheRequest::new(CacheKeyInput {
            preset: Some(key.into()),
            ..Default::default()
        })
    };
    cache
        .store(&request("sync"), large.clone(), Duration::ZERO)
        .unwrap();
    cache
        .async_store(&request("async"), large.clone(), Duration::ZERO)
        .await
        .unwrap();
    cache
        .async_store_batch(
            vec![
                (request("batch-large"), large),
                (request("batch-small"), small.clone()),
            ],
            Duration::ZERO,
        )
        .await
        .unwrap();
    let service: Arc<dyn ResponseCacheService> = Arc::new(cache);
    let stored = request("service");
    service
        .store(
            &service.key(&stored).await.unwrap(),
            &stored,
            small.clone(),
            Duration::ZERO,
        )
        .await
        .unwrap();
    for key in ["sync", "async", "batch-large"] {
        assert!(storage.get_cache(key).unwrap().is_none());
    }
    for key in ["batch-small", "service"] {
        assert_eq!(
            service
                .lookup(
                    &CacheKey::delegated(key.into()),
                    &request(key),
                    Duration::ZERO
                )
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
        request: &'a ResponseCacheRequest,
    ) -> BoxFuture<'a, Result<CacheKey, Error>> {
        Box::pin(async move {
            request
                .key
                .preset
                .clone()
                .map(CacheKey::delegated)
                .ok_or(Error::Unavailable)
        })
    }

    fn lookup<'a>(
        &'a self,
        key: &'a CacheKey,
        _: &'a ResponseCacheRequest,
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
        _: &'a ResponseCacheRequest,
        _: Value,
        _: Duration,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }
}

fn request(key: &str) -> ResponseCacheRequest {
    ResponseCacheRequest::new(CacheKeyInput {
        preset: Some(key.into()),
        ..Default::default()
    })
}

async fn keyed(
    service: &dyn ResponseCacheService,
    requests: impl IntoIterator<Item = ResponseCacheRequest>,
) -> Vec<(CacheKey, ResponseCacheRequest)> {
    futures_util::future::try_join_all(
        requests
            .into_iter()
            .map(|request| async move { Ok::<_, Error>((service.key(&request).await?, request)) }),
    )
    .await
    .unwrap()
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
    #[case] keys: &[&str],
    #[case] missing: &[usize],
) {
    let requests = keyed(service.as_ref(), keys.iter().map(|key| request(key))).await;
    let now = Duration::from_secs(100);
    let batch = service.lookup_batch(&requests, now).await.unwrap();
    let expected: Vec<_> = futures_util::future::try_join_all(
        requests
            .iter()
            .map(|(key, request)| service.lookup(key, request, now)),
    )
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
    let requests = keyed(service.as_ref(), [request("hit"), request("failure")]).await;
    let result = service
        .lookup_batch(&requests, Duration::from_secs(100))
        .await;
    assert_eq!(result, Err(Error::Unavailable));
}

#[rstest]
#[tokio::test]
async fn response_service_uses_bulk_reads_and_applies_per_request_policy() {
    let response = json!({"answer": 7});
    let encoded = ResponseCacheCodec
        .encode(&CacheEntry::produced_at(
            response.clone(),
            Duration::from_secs(100),
        ))
        .unwrap();
    let connection = MockRedisConnection::new(vec![MockCmd::new(
        redis::cmd("MGET").arg(["hit", "miss", "hit", "stale"].as_slice()),
        Ok(vec![
            redis::Value::BulkString(encoded.clone()),
            redis::Value::Nil,
            redis::Value::BulkString(encoded.clone()),
            redis::Value::BulkString(encoded),
        ]),
    )])
    .assert_all_commands_consumed();
    let cache: Arc<dyn ResponseCacheService> = Arc::new(ResponseCache::new(Arc::new(
        RedisCache::with_connection(connection, None, ResponseCacheCodec),
    )));
    let disabled = ResponseCacheRequest {
        access: CacheAccess {
            reads: false,
            writes: true,
        },
        ..request("disabled")
    };
    let stale = ResponseCacheRequest {
        max_age: Some(Duration::from_secs(1)),
        ..request("stale")
    };
    let requests = keyed(
        cache.as_ref(),
        [
            request("hit"),
            disabled,
            request("miss"),
            request("hit"),
            stale,
        ],
    )
    .await;
    let batch = cache
        .lookup_batch(&requests, Duration::from_secs(102))
        .await
        .unwrap();
    assert_eq!(
        batch.values,
        vec![Some(response.clone()), None, None, Some(response), None]
    );
    assert_eq!(batch.missing_indices(), vec![1, 2, 4]);
}
