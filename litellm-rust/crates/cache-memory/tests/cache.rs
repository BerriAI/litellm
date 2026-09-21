use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use litellm_cache::{
    BaseCache, CacheBackend, CacheConnectionStatus, CacheKwargs, ClaimCache, CounterCache, Error,
    get_cache, set_cache,
};
use litellm_cache_memory::{CacheWrite, InMemoryCache};
use rstest::{fixture, rstest};

#[fixture]
fn clock() -> Arc<AtomicU64> {
    Arc::new(AtomicU64::new(100))
}

fn cache(clock: Arc<AtomicU64>, capacity: usize) -> InMemoryCache<String> {
    InMemoryCache::with_clock(Some(capacity), Some(Duration::from_secs(60)), move || {
        Duration::from_secs(clock.load(Ordering::SeqCst))
    })
}

#[rstest]
fn default_explicit_and_override_ttls_follow_python_rules(clock: Arc<AtomicU64>) {
    let cache = cache(clock.clone(), 4);
    cache.set_cache("key", "first".into(), None).unwrap();
    assert_eq!(
        cache.expires_at("key").unwrap(),
        Some(Duration::from_secs(160))
    );
    cache
        .set_cache("key", "second".into(), Some(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(
        cache.expires_at("key").unwrap(),
        Some(Duration::from_secs(160))
    );
    clock.store(160, Ordering::SeqCst);
    assert_eq!(cache.get_cache("key").unwrap(), Some("second".into()));
    clock.store(161, Ordering::SeqCst);
    assert_eq!(cache.get_cache("key").unwrap(), None);
    cache
        .set_cache("key", "third".into(), Some(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(
        cache.expires_at("key").unwrap(),
        Some(Duration::from_secs(171))
    );
}

#[rstest]
fn write_at_expiry_boundary_refreshes_ttl(clock: Arc<AtomicU64>) {
    let cache = cache(clock.clone(), 4);
    cache
        .set_cache("key", "first".into(), Some(Duration::from_secs(10)))
        .unwrap();
    clock.store(110, Ordering::SeqCst);
    cache
        .set_cache("key", "second".into(), Some(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(
        cache.expires_at("key").unwrap(),
        Some(Duration::from_secs(120))
    );
    clock.store(115, Ordering::SeqCst);
    assert_eq!(cache.get_cache("key").unwrap(), Some("second".into()));
}

#[rstest]
fn capacity_evicts_earliest_and_ignores_stale_heap_entries(clock: Arc<AtomicU64>) {
    let cache = cache(clock, 2);
    cache
        .set_cache("early", "a".into(), Some(Duration::from_secs(10)))
        .unwrap();
    cache
        .set_cache("late", "b".into(), Some(Duration::from_secs(20)))
        .unwrap();
    cache.delete_cache("early").unwrap();
    cache
        .set_cache("new", "c".into(), Some(Duration::from_secs(30)))
        .unwrap();
    assert_eq!(cache.get_cache("late").unwrap(), Some("b".into()));
    cache
        .set_cache("last", "d".into(), Some(Duration::from_secs(40)))
        .unwrap();
    assert_eq!(cache.get_cache("late").unwrap(), None);
}

#[test]
fn disabled_size_limited_and_validated_writes_are_observable() {
    let cache = |capacity| {
        InMemoryCache::with_clock_and_size_measurement(
            Some(capacity),
            Some(Duration::from_secs(60)),
            Some(4),
            Some(Arc::new(|value: &String| {
                if value.is_empty() {
                    return Err(Error::InvalidEntry);
                }
                Ok(value.len())
            })),
            || Duration::from_secs(100),
        )
    };
    let disabled = cache(0);
    assert_eq!(
        disabled.set_cache("a", "x".into(), None).unwrap(),
        CacheWrite::Disabled
    );
    let cache = cache(2);
    assert_eq!(
        cache.set_cache("large", "oversized".into(), None).unwrap(),
        CacheWrite::TooLarge
    );
    assert_eq!(cache.get_cache("large").unwrap(), None);
    assert_eq!(
        cache.set_cache("small", "ok".into(), None).unwrap(),
        CacheWrite::Stored
    );
    assert_eq!(cache.get_cache("small").unwrap(), Some("ok".into()));
    assert_eq!(
        cache.set_cache("invalid", String::new(), None),
        Err(Error::InvalidEntry)
    );
    assert_eq!(cache.get_cache("invalid").unwrap(), None);
    cache.delete_cache("small").unwrap();
    assert_eq!(cache.get_cache("small").unwrap(), None);
}

#[tokio::test]
async fn connection_test_matches_python_result_contract() {
    let cache = InMemoryCache::<String>::default();
    let result = BaseCache::test_connection(&cache).await.unwrap();
    assert_eq!(result.status, CacheConnectionStatus::Success);
    assert_eq!(result.message, "In-memory cache connection test successful");
    assert_eq!(result.error, None);
    assert_eq!(
        serde_json::to_value(result).unwrap(),
        serde_json::json!({
            "status": "success",
            "message": "In-memory cache connection test successful"
        })
    );
}

#[tokio::test]
async fn generic_consumers_share_typed_values_and_honor_expiration() {
    let clock = clock();
    let cache: CacheBackend<InMemoryCache<String>> = Arc::new(cache(clock.clone(), 4));
    let reader = Arc::clone(&cache);
    let kwargs = CacheKwargs {
        ttl: Some(Duration::from_secs(5)),
        ..Default::default()
    };
    set_cache(cache.as_ref(), "sync", "first".into(), kwargs.clone()).unwrap();
    assert_eq!(
        get_cache(reader.as_ref(), "sync", &kwargs).unwrap(),
        Some("first".into())
    );
    cache
        .batch_cache_write("async", "second".into(), kwargs.clone())
        .await
        .unwrap();
    cache
        .async_set_cache_pipeline(vec![("batch".into(), "third".into())], kwargs.clone())
        .await
        .unwrap();
    drop(cache);
    for (key, value) in [("sync", "first"), ("async", "second"), ("batch", "third")] {
        assert_eq!(
            reader.async_get_cache(key, &kwargs).await.unwrap(),
            Some(value.into())
        );
    }
    reader.async_delete_cache("async").await.unwrap();
    assert_eq!(
        reader.async_get_cache("async", &kwargs).await.unwrap(),
        None
    );
    clock.store(106, Ordering::SeqCst);
    assert_eq!(get_cache(reader.as_ref(), "sync", &kwargs).unwrap(), None);
    assert_eq!(
        reader.async_get_cache("batch", &kwargs).await.unwrap(),
        None
    );
}

#[test]
fn claims_are_atomic_and_refresh_eligible_winners() {
    let clock = clock();
    let cache = InMemoryCache::with_clock(Some(4), Some(Duration::from_secs(60)), {
        let clock = clock.clone();
        move || Duration::from_secs(clock.load(Ordering::SeqCst))
    });
    let kwargs = CacheKwargs {
        ttl: Some(Duration::from_secs(10)),
        ..Default::default()
    };
    assert_eq!(
        cache
            .claim_cache("affinity", "first".to_string(), &[], kwargs.clone())
            .unwrap(),
        "first"
    );
    clock.store(105, Ordering::SeqCst);
    assert_eq!(
        cache
            .claim_cache(
                "affinity",
                "second".to_string(),
                &["first".to_string(), "second".to_string()],
                kwargs,
            )
            .unwrap(),
        "first"
    );
    assert_eq!(
        cache.expires_at("affinity").unwrap(),
        Some(Duration::from_secs(115))
    );
}

#[test]
fn counters_increment_under_one_lock() {
    let cache = InMemoryCache::<f64>::default();
    assert_eq!(
        CounterCache::increment_cache(&cache, "counter", 1.5, CacheKwargs::default()).unwrap(),
        1.5
    );
    assert_eq!(
        CounterCache::increment_cache(&cache, "counter", 2.0, CacheKwargs::default()).unwrap(),
        3.5
    );
}
