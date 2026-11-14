mod support;

use std::{sync::Arc, time::Duration};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheEntry, ExactResponseCache, ResponseCache, ResponseCacheRequest};
use rstest::rstest;
use serde_json::json;
use support::{keyed, memory, request};

/// The in-memory backend has no `test_connection`, as in Python, and still serves every response
/// operation.
#[rstest]
#[tokio::test]
async fn backends_without_a_connection_test_serve_every_response_operation(
    #[from(memory)] service: Arc<ResponseCache<InMemoryCache<CacheEntry>>>,
    request: ResponseCacheRequest,
) {
    let cache: Arc<dyn ExactResponseCache> = service;
    let now = Duration::from_secs(100);
    let other = keyed("tenant:other");
    let missing = keyed("tenant:missing");

    assert_eq!(cache.default_ttl(), Some(Duration::from_secs(600)));
    cache.store(&request, json!({"v": 1}), now).unwrap();
    assert_eq!(cache.lookup(&request, now).unwrap(), Some(json!({"v": 1})));
    cache
        .async_store(&other, json!({"v": 2}), now)
        .await
        .unwrap();
    assert_eq!(
        cache.async_lookup(&other, now).await.unwrap(),
        Some(json!({"v": 2}))
    );

    let requests = [request.clone(), missing.clone(), other.clone()];
    let partial = cache.lookup_batch(&requests, now).unwrap();
    assert_eq!(
        partial.values,
        vec![Some(json!({"v": 1})), None, Some(json!({"v": 2}))]
    );
    assert_eq!(partial.missing_indices, vec![1]);

    cache
        .async_store_batch(vec![(missing.clone(), json!({"v": 3}))], now)
        .await
        .unwrap();
    let partial = cache.async_lookup_batch(&requests, now).await.unwrap();
    assert!(partial.missing_indices.is_empty());
    assert_eq!(partial.values[1], Some(json!({"v": 3})));

    cache.async_flush().await.unwrap();
    let partial = cache.async_lookup_batch(&requests, now).await.unwrap();
    assert_eq!(partial.missing_indices, vec![0, 1, 2]);
}
