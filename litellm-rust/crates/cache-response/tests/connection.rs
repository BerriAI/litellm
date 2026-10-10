mod support;

use std::{sync::Arc, time::Duration};

use litellm_cache::CacheConnectionStatus;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheEntry, ConnectionProbe, ExactResponseCache, ResponseCache, ResponseCacheRequest,
};
use redis_test::MockCmd;
use rstest::rstest;
use serde_json::json;
use support::{keyed, memory, redis, request};

#[rstest]
#[case::reachable(
    Ok("PONG"),
    CacheConnectionStatus::Success,
    "Redis connection test successful",
    false
)]
#[case::unexpected_reply(
    Ok("NOPE"),
    CacheConnectionStatus::Failed,
    "Redis ping returned False",
    false
)]
#[case::connection_refused(
    Err(redis::RedisError::from((redis::ErrorKind::Io, "connection refused"))),
    CacheConnectionStatus::Failed,
    "Redis connection failed:",
    true
)]
#[tokio::test]
async fn connection_backends_are_reachable_as_a_probe(
    #[case] reply: redis::RedisResult<&'static str>,
    #[case] status: CacheConnectionStatus,
    #[case] message: &str,
    #[case] has_error: bool,
) {
    let probe: Arc<dyn ConnectionProbe> =
        Arc::new(redis(vec![MockCmd::new(redis::cmd("PING"), reply)], None));

    let result = probe.test_connection().await.unwrap();
    assert_eq!(result.status, status);
    assert!(result.message.starts_with(message), "{}", result.message);
    assert_eq!(result.error.is_some(), has_error);
}

#[rstest]
#[tokio::test]
async fn one_service_serves_both_the_exact_cache_and_its_probe(request: ResponseCacheRequest) {
    let service = Arc::new(redis(
        vec![
            MockCmd::new(redis::cmd("PING"), Ok("PONG")),
            MockCmd::new(
                redis::cmd("GET").arg("tenant:key"),
                Ok(br#"{"timestamp":100.0,"response":{"ok":true}}"#.to_vec()),
            ),
        ],
        Some("tenant"),
    ));
    let probe: Arc<dyn ConnectionProbe> = service.clone();
    let exact: Arc<dyn ExactResponseCache> = service;

    assert_eq!(
        probe.test_connection().await.unwrap().status,
        CacheConnectionStatus::Success
    );
    assert_eq!(
        exact
            .async_lookup(&request, Duration::from_secs(100))
            .await
            .unwrap(),
        Some(json!({"ok": true}))
    );
}

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
