use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use litellm_cache::{CacheEntry, Error};
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
        cache.get_ttl("key").unwrap(),
        Some(Duration::from_secs(160))
    );
    cache
        .set_cache("key", "second".into(), Some(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(
        cache.get_ttl("key").unwrap(),
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
        cache.get_ttl("key").unwrap(),
        Some(Duration::from_secs(171))
    );
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
fn disabled_size_limited_and_synchronized_response_writes_are_observable() {
    let disabled = InMemoryCache::<CacheEntry>::response_cache(0, Duration::from_secs(60), 80);
    assert_eq!(
        disabled
            .set_cache(
                "a",
                CacheEntry {
                    timestamp: 1.0,
                    response: serde_json::json!("x")
                },
                None
            )
            .unwrap(),
        CacheWrite::Disabled
    );
    let cache = InMemoryCache::<CacheEntry>::response_cache(2, Duration::from_secs(60), 80);
    assert_eq!(
        cache
            .set_cache(
                "large",
                CacheEntry {
                    timestamp: 1.0,
                    response: serde_json::json!("x".repeat(100))
                },
                None
            )
            .unwrap(),
        CacheWrite::TooLarge
    );
    cache
        .set_cache(
            "small",
            CacheEntry {
                timestamp: 1.0,
                response: serde_json::json!("ok"),
            },
            None,
        )
        .unwrap();
    assert!(cache.get_cache("small").unwrap().is_some());
    assert_eq!(
        cache
            .set_cache(
                "invalid",
                CacheEntry {
                    timestamp: f64::NAN,
                    response: serde_json::json!("bad"),
                },
                None,
            )
            .unwrap_err(),
        Error::InvalidEntry
    );
    cache.delete_cache("small").unwrap();
    cache.flush_cache().unwrap();
}
