use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::ExactCacheContext;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheEntry, CacheKeyInput, CacheKeyRequest, CacheOptions, CacheScope, RequestRewrite,
    ResponseCache, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, get_cache_key,
};
use rstest::rstest;
use serde_json::json;

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
