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
    CacheAccess, CacheEntry, CacheKeyInput, ResponseCache, ResponseCacheCodec, ResponseCacheConfig,
    ResponseCacheRequest, ResponseCacheService, get_cache_key,
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
    cache
        .store(&request, json!({"answer":7}), Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(
        cache.lookup(&request, Duration::ZERO).await.unwrap(),
        Some(json!({"answer":7}))
    );
    let stale_request = ResponseCacheRequest {
        max_age: Some(Duration::from_secs(1)),
        ..request.clone()
    };
    clock.store(2, Ordering::SeqCst);
    assert_eq!(
        cache
            .lookup(&stale_request, Duration::from_secs(2))
            .await
            .unwrap(),
        None
    );
    assert!(
        cache
            .lookup(&request, Duration::from_secs(2))
            .await
            .unwrap()
            .is_some()
    );
    clock.store(6, Ordering::SeqCst);
    assert_eq!(
        cache
            .lookup(&request, Duration::from_secs(6))
            .await
            .unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn default_key_resolution_preserves_the_native_cache_key() {
    let cache: Arc<dyn ResponseCacheService> = Arc::new(ResponseCache::new(Arc::new(
        InMemoryCache::<CacheEntry>::default(),
    )));
    let request = ResponseCacheRequest::new(CacheKeyInput {
        namespace: Some("namespace".into()),
        ..Default::default()
    });
    assert_eq!(
        cache.get_cache_key(&request).await.unwrap(),
        get_cache_key(&request.key)
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
    service
        .store(&request("service"), small.clone(), Duration::ZERO)
        .await
        .unwrap();
    for key in ["sync", "async", "batch-large"] {
        assert!(storage.get_cache(key).unwrap().is_none());
    }
    for key in ["batch-small", "service"] {
        assert_eq!(
            service.lookup(&request(key), Duration::ZERO).await.unwrap(),
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

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        now: Duration,
    ) -> BoxFuture<'a, Result<Option<Value>, Error>> {
        Box::pin(async move {
            if now != Duration::from_secs(100) {
                return Err(Error::Unavailable);
            }
            match request.key.preset.as_deref() {
                Some("hit") => Ok(Some(json!({"answer": 7}))),
                Some("failure") => Err(Error::Unavailable),
                _ => Ok(None),
            }
        })
    }

    fn store<'a>(
        &'a self,
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
    let requests: Vec<_> = keys.iter().map(|key| request(key)).collect();
    let now = Duration::from_secs(100);
    let batch = service.lookup_batch(&requests, now).await.unwrap();
    let expected: Vec<_> = futures_util::future::try_join_all(
        requests.iter().map(|request| service.lookup(request, now)),
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
    let result = service
        .lookup_batch(
            &[request("hit"), request("failure")],
            Duration::from_secs(100),
        )
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
    let requests = [
        request("hit"),
        disabled,
        request("miss"),
        request("hit"),
        stale,
    ];
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
