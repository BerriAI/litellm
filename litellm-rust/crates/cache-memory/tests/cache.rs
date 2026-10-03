use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheBackend, ClaimCache, CounterCache, DeleteCache,
    DisconnectCache, Error, ExactCacheContext, FlushCache, IncrementOperation, SetCache, TtlCache,
    get_cache, set_cache,
};
use litellm_cache_memory::{CacheWrite, InMemoryCache};
use rstest::{fixture, rstest};

type Clock = Arc<AtomicU64>;

#[fixture]
fn clock() -> Clock {
    Arc::new(AtomicU64::new(100))
}

fn cache_with<V: Clone>(clock: &Clock, capacity: usize) -> InMemoryCache<V> {
    let clock = clock.clone();
    InMemoryCache::with_clock(Some(capacity), Some(Duration::from_secs(60)), move || {
        Duration::from_millis(clock.load(Ordering::SeqCst) * 1000)
    })
}

fn cache(clock: &Clock, capacity: usize) -> InMemoryCache<String> {
    cache_with(clock, capacity)
}

fn at(clock: &Clock, seconds: u64) {
    clock.store(seconds, Ordering::SeqCst);
}

fn secs(seconds: u64) -> Option<Duration> {
    Some(Duration::from_secs(seconds))
}

fn ttl(seconds: u64) -> ExactCacheContext {
    ExactCacheContext { ttl: secs(seconds) }
}

fn measured(capacity: usize) -> InMemoryCache<String> {
    InMemoryCache::with_clock_and_size_measurement(
        Some(capacity),
        secs(60),
        Some(4),
        Some(Arc::new(|value: &String| {
            if value.is_empty() {
                return Err(Error::InvalidEntry);
            }
            Ok(value.len())
        })),
        || Duration::from_secs(100),
    )
}

#[rstest]
fn default_explicit_and_override_ttls_follow_python_rules(clock: Clock) {
    let cache = cache(&clock, 4);
    cache.set_cache("key", "first".into(), None).unwrap();
    assert_eq!(cache.expires_at("key").unwrap(), secs(160));
    cache.set_cache("key", "second".into(), secs(10)).unwrap();
    assert_eq!(cache.expires_at("key").unwrap(), secs(160));
    at(&clock, 160);
    assert_eq!(cache.get_cache("key").unwrap(), Some("second".into()));
    at(&clock, 161);
    assert_eq!(cache.get_cache("key").unwrap(), None);
    assert_eq!(cache.expires_at("key").unwrap(), None);
    cache.set_cache("key", "third".into(), secs(10)).unwrap();
    assert_eq!(cache.expires_at("key").unwrap(), secs(171));
}

#[rstest]
#[case::unset(None, secs(600))]
#[case::zero_falls_back_like_python_or(Some(Duration::ZERO), secs(600))]
#[case::explicit(secs(5), secs(5))]
fn default_ttl_falls_back_to_ten_minutes(
    #[case] default_ttl: Option<Duration>,
    #[case] expected: Option<Duration>,
) {
    let cache = InMemoryCache::<String>::with_clock(None, default_ttl, || Duration::ZERO);
    assert_eq!(cache.get_ttl(&ExactCacheContext::default()), expected);
    cache.set_cache("key", "value".into(), None).unwrap();
    assert_eq!(cache.expires_at("key").unwrap(), expected);
    assert_eq!(cache.max_size_in_memory(), 200);
}

#[rstest]
fn write_at_expiry_boundary_refreshes_ttl(clock: Clock) {
    let cache = cache(&clock, 4);
    cache.set_cache("key", "first".into(), secs(10)).unwrap();
    at(&clock, 110);
    cache.set_cache("key", "second".into(), secs(10)).unwrap();
    assert_eq!(cache.expires_at("key").unwrap(), secs(120));
    at(&clock, 115);
    assert_eq!(cache.get_cache("key").unwrap(), Some("second".into()));
}

#[rstest]
fn expired_key_without_a_read_allows_a_ttl_override(clock: Clock) {
    let cache = cache(&clock, 4);
    cache.set_cache("key", "first".into(), secs(1)).unwrap();
    assert_eq!(cache.allow_ttl_override("key"), Ok(false));
    at(&clock, 102);
    assert_eq!(cache.allow_ttl_override("key"), Ok(true));
    cache.set_cache("key", "second".into(), secs(1)).unwrap();
    assert_eq!(cache.expires_at("key").unwrap(), secs(103));
    assert_eq!(cache.allow_ttl_override("missing"), Ok(true));
}

#[rstest]
fn capacity_evicts_earliest_and_ignores_stale_heap_entries(clock: Clock) {
    let cache = cache(&clock, 2);
    cache.set_cache("early", "a".into(), secs(10)).unwrap();
    cache.set_cache("late", "b".into(), secs(20)).unwrap();
    cache.delete_cache("early").unwrap();
    cache.set_cache("new", "c".into(), secs(30)).unwrap();
    assert_eq!(cache.get_cache("late").unwrap(), Some("b".into()));
    cache.set_cache("last", "d".into(), secs(40)).unwrap();
    assert_eq!(cache.get_cache("late").unwrap(), None);
}

#[rstest]
fn max_size_is_respected_when_every_item_has_a_long_ttl(clock: Clock) {
    let cache = cache(&clock, 3);
    for index in 0..3 {
        at(&clock, 100 + index);
        cache
            .set_cache(
                format!("key_{index}"),
                format!("value_{index}"),
                secs(86_400),
            )
            .unwrap();
    }
    assert_eq!(cache.len(), Ok(3));
    cache
        .set_cache("key_3", "value_3".into(), secs(86_400))
        .unwrap();
    assert_eq!(cache.len(), Ok(3));
    assert_eq!(cache.get_cache("key_0").unwrap(), None);
    assert_eq!(cache.expires_at("key_0").unwrap(), None);
    for key in ["key_1", "key_2", "key_3"] {
        assert!(cache.get_cache(key).unwrap().is_some(), "{key}");
    }
}

#[rstest]
fn expired_items_are_evicted_before_live_ones(clock: Clock) {
    let cache = cache(&clock, 3);
    cache.set_cache("expired_1", "1".into(), secs(1)).unwrap();
    cache.set_cache("expired_2", "2".into(), secs(1)).unwrap();
    cache
        .set_cache("long_lived", "3".into(), secs(86_400))
        .unwrap();
    assert_eq!(cache.len(), Ok(3));
    at(&clock, 102);
    cache
        .set_cache("new_item", "4".into(), secs(86_400))
        .unwrap();
    assert_eq!(cache.len(), Ok(2));
    assert_eq!(cache.get_cache("long_lived").unwrap(), Some("3".into()));
    assert_eq!(cache.get_cache("new_item").unwrap(), Some("4".into()));
    for key in ["expired_1", "expired_2"] {
        assert_eq!(cache.expires_at(key).unwrap(), None, "{key}");
    }
}

#[rstest]
fn injected_clock_controls_expiry_and_eviction(clock: Clock) {
    let cache = cache(&clock, 2);
    at(&clock, 0);
    cache
        .set_cache("first", "original".into(), secs(10))
        .unwrap();
    at(&clock, 9);
    cache.set_cache("second", "survivor".into(), None).unwrap();
    assert_eq!(cache.get_cache("first").unwrap(), Some("original".into()));
    at(&clock, 11);
    assert_eq!(cache.get_cache("first").unwrap(), None);
    cache
        .set_cache("third", "replacement".into(), None)
        .unwrap();
    assert_eq!(cache.get_cache("second").unwrap(), Some("survivor".into()));
    at(&clock, 70);
    cache.set_cache("fourth", "new".into(), None).unwrap();
    assert_eq!(cache.get_cache("second").unwrap(), None);
    assert_eq!(
        cache.get_cache("third").unwrap(),
        Some("replacement".into())
    );
    assert_eq!(cache.get_cache("fourth").unwrap(), Some("new".into()));
}

#[rstest]
fn rewriting_one_key_keeps_one_heap_entry(clock: Clock) {
    let cache = cache(&clock, 10);
    for index in 0..1_000 {
        cache
            .set_cache("hot_key", format!("value_{index}"), secs(60))
            .unwrap();
    }
    assert_eq!(cache.expiration_heap_len(), Ok(1));
}

#[rstest]
fn repeated_increments_keep_one_heap_entry_per_expiration() {
    let cache = InMemoryCache::<f64>::new(Some(4), None);
    for _ in 0..100 {
        cache
            .increment_cache("counter", 1.0, ExactCacheContext::default())
            .unwrap();
    }
    assert_eq!(cache.expiration_heap_len(), Ok(1));
}

#[rstest]
fn reinserting_expired_keys_below_capacity_prunes_the_heap(clock: Clock) {
    let cache = cache(&clock, 200);
    for cycle in 0..3 {
        for index in 0..5 {
            cache
                .set_cache(format!("key_{index}"), format!("value_{cycle}"), secs(1))
                .unwrap();
        }
        at(&clock, 100 + 2 * (cycle + 1));
    }
    for index in 0..5 {
        cache
            .set_cache(format!("key_{index}"), "final".into(), secs(1))
            .unwrap();
    }
    assert_eq!(cache.len(), Ok(5));
    assert_eq!(cache.expiration_heap_len(), Ok(5));
}

#[rstest]
fn evict_cache_drops_expired_entries_then_makes_room(clock: Clock) {
    let cache = cache(&clock, 2);
    assert_eq!(cache.is_empty(), Ok(true));
    cache.set_cache("short", "a".into(), secs(1)).unwrap();
    cache.set_cache("long", "b".into(), secs(50)).unwrap();
    at(&clock, 102);
    cache.evict_cache().unwrap();
    assert_eq!(cache.len(), Ok(1));
    assert_eq!(cache.expires_at("short").unwrap(), None);
    cache.set_cache("longer", "c".into(), secs(90)).unwrap();
    cache.evict_cache().unwrap();
    assert_eq!(cache.len(), Ok(1));
    assert_eq!(cache.get_cache("long").unwrap(), None);
    assert_eq!(cache.get_cache("longer").unwrap(), Some("c".into()));
}

#[rstest]
fn evict_element_if_expired_reports_removal(clock: Clock) {
    let cache = cache(&clock, 4);
    cache.set_cache("key", "value".into(), secs(10)).unwrap();
    assert_eq!(cache.evict_element_if_expired("key"), Ok(false));
    assert_eq!(cache.evict_element_if_expired("missing"), Ok(false));
    at(&clock, 110);
    assert_eq!(cache.evict_element_if_expired("key"), Ok(false));
    at(&clock, 111);
    assert_eq!(cache.evict_element_if_expired("key"), Ok(true));
    assert_eq!(cache.len(), Ok(0));
    assert_eq!(cache.expires_at("key").unwrap(), None);
}

#[rstest]
#[case::fits("ok", Ok(true))]
#[case::at_limit("four", Ok(true))]
#[case::too_large("oversized", Ok(false))]
#[case::measure_error("", Err(Error::InvalidEntry))]
fn check_value_size_applies_the_entry_limit(
    #[case] value: &str,
    #[case] expected: Result<bool, Error>,
) {
    assert_eq!(measured(2).check_value_size(&value.to_string()), expected);
}

#[rstest]
fn values_are_unbounded_without_a_measure() {
    let cache = InMemoryCache::<String>::default();
    assert_eq!(cache.max_entry_bytes(), None);
    assert_eq!(cache.check_value_size(&"x".repeat(1 << 20)), Ok(true));
}

#[rstest]
fn disabled_size_limited_and_validated_writes_are_observable() {
    assert_eq!(
        measured(0).set_cache("a", "x".into(), None).unwrap(),
        CacheWrite::Disabled
    );
    let cache = measured(2);
    assert_eq!(cache.max_entry_bytes(), Some(4));
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

#[rstest]
#[tokio::test]
async fn disconnect_is_a_no_op_that_keeps_entries() {
    let cache = InMemoryCache::<String>::default();
    cache.set_cache("key", "value".into(), None).unwrap();
    cache.disconnect().await.unwrap();
    assert_eq!(cache.get_cache("key").unwrap(), Some("value".into()));
}

#[rstest]
#[tokio::test]
async fn generic_consumers_share_typed_values_and_honor_expiration(clock: Clock) {
    let cache: CacheBackend<InMemoryCache<String>> = Arc::new(self::cache(&clock, 4));
    let reader = Arc::clone(&cache);
    let context = ttl(5);
    set_cache(cache.as_ref(), "sync", "first".into(), &context).unwrap();
    assert_eq!(
        get_cache(reader.as_ref(), "sync", &context).unwrap(),
        Some("first".into())
    );
    cache
        .batch_cache_write("async", "second".into(), context.clone())
        .await
        .unwrap();
    cache
        .async_set_cache_pipeline(vec![("batch".into(), "third".into())], context.clone())
        .await
        .unwrap();
    drop(cache);
    for (key, value) in [("sync", "first"), ("async", "second"), ("batch", "third")] {
        assert_eq!(
            reader.async_get_cache(key, &context).await.unwrap(),
            Some(value.into())
        );
    }
    reader.async_delete_cache("async").await.unwrap();
    assert_eq!(
        reader.async_get_cache("async", &context).await.unwrap(),
        None
    );
    at(&clock, 106);
    assert_eq!(get_cache(reader.as_ref(), "sync", &context).unwrap(), None);
    assert_eq!(
        reader.async_get_cache("batch", &context).await.unwrap(),
        None
    );
}

#[rstest]
#[case::context_ttl(ttl(5), secs(105))]
#[case::default_ttl(ExactCacheContext::default(), secs(160))]
#[tokio::test]
async fn pipeline_writes_use_the_context_ttl_or_the_default(
    clock: Clock,
    #[case] context: ExactCacheContext,
    #[case] expected: Option<Duration>,
) {
    let cache = cache(&clock, 4);
    cache
        .async_set_cache_pipeline(
            vec![("a".into(), "1".into()), ("b".into(), "2".into())],
            context,
        )
        .await
        .unwrap();
    assert_eq!(cache.expires_at("a").unwrap(), expected);
    assert_eq!(cache.expires_at("b").unwrap(), expected);
}

#[rstest]
#[tokio::test]
async fn batch_reads_return_one_entry_per_key_and_drop_expired_ones(clock: Clock) {
    let cache = cache(&clock, 4);
    cache.set_cache("short", "a".into(), secs(1)).unwrap();
    cache.set_cache("long", "b".into(), secs(50)).unwrap();
    let keys = vec!["short".to_string(), "missing".into(), "long".into()];
    assert_eq!(
        cache
            .batch_get_cache(&keys, &ExactCacheContext::default())
            .unwrap(),
        [
            BatchEntry::Hit("a".to_string()),
            BatchEntry::Miss,
            BatchEntry::Hit("b".into()),
        ]
    );
    at(&clock, 102);
    assert_eq!(
        cache
            .async_batch_get_cache(keys, ExactCacheContext::default())
            .await
            .unwrap(),
        [
            BatchEntry::Miss,
            BatchEntry::Miss,
            BatchEntry::Hit("b".into())
        ]
    );
}

#[rstest]
#[tokio::test]
async fn flush_clears_values_and_expirations(clock: Clock) {
    let cache = cache(&clock, 4);
    cache.set_cache("a", "1".into(), None).unwrap();
    cache.set_cache("b", "2".into(), None).unwrap();
    cache.flush_cache().unwrap();
    assert_eq!(cache.len(), Ok(0));
    assert_eq!(cache.expiration_heap_len(), Ok(0));
    cache.set_cache("c", "3".into(), None).unwrap();
    FlushCache::async_flush_cache(&cache).await.unwrap();
    assert_eq!(cache.is_empty(), Ok(true));
    assert_eq!(
        cache.async_get_oldest_n_keys(5).await.unwrap(),
        Vec::<String>::new()
    );
}

#[rstest]
fn claims_are_atomic_and_refresh_eligible_winners(clock: Clock) {
    let cache = cache(&clock, 4);
    let context = ttl(10);
    assert_eq!(
        cache
            .claim_cache("affinity", "first".to_string(), &[], context.clone())
            .unwrap(),
        "first"
    );
    at(&clock, 103);
    assert_eq!(
        cache
            .claim_cache("affinity", "second".to_string(), &[], context.clone())
            .unwrap(),
        "first"
    );
    assert_eq!(cache.expires_at("affinity").unwrap(), secs(110));
    at(&clock, 105);
    assert_eq!(
        cache
            .claim_cache(
                "affinity",
                "second".to_string(),
                &["first".to_string(), "second".to_string()],
                context,
            )
            .unwrap(),
        "first"
    );
    assert_eq!(cache.expires_at("affinity").unwrap(), secs(115));
}

#[rstest]
fn counters_increment_under_one_lock() {
    let cache = InMemoryCache::<f64>::default();
    assert_eq!(
        CounterCache::increment_cache(&cache, "counter", 1.5, ExactCacheContext::default())
            .unwrap(),
        1.5
    );
    assert_eq!(
        CounterCache::increment_cache(&cache, "counter", 2.0, ExactCacheContext::default())
            .unwrap(),
        3.5
    );
}

#[rstest]
fn concurrent_increments_are_atomic() {
    let cache = Arc::new(InMemoryCache::<f64>::default());
    cache.set_cache("counter", 1000.0, None).unwrap();
    let threads = (0..8)
        .map(|_| {
            let cache = cache.clone();
            std::thread::spawn(move || {
                cache
                    .increment_cache("counter", 1.0, ExactCacheContext::default())
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(cache.get_cache("counter").unwrap(), Some(1008.0));
}

#[rstest]
#[case::window_semantics(false)]
#[case::refresh_ttl_is_ignored(true)]
#[tokio::test]
async fn async_increment_delegates_to_the_locked_sync_path(
    clock: Clock,
    #[case] refresh_ttl: bool,
) {
    let cache = cache_with::<f64>(&clock, 4);
    assert_eq!(
        cache
            .async_increment("counter", 2.0, ttl(10), refresh_ttl)
            .await,
        Ok(2.0)
    );
    at(&clock, 105);
    assert_eq!(
        cache
            .async_increment("counter", 3.0, ttl(10), refresh_ttl)
            .await,
        Ok(5.0)
    );
    assert_eq!(cache.get_cache("counter").unwrap(), Some(5.0));
    assert_eq!(cache.expires_at("counter").unwrap(), secs(110));
}

#[rstest]
fn expired_counters_restart_from_zero_with_a_new_ttl(clock: Clock) {
    let cache = cache_with::<f64>(&clock, 4);
    cache.increment_cache("counter", 2.0, ttl(10)).unwrap();
    at(&clock, 111);
    assert_eq!(cache.increment_cache("counter", 1.0, ttl(10)), Ok(1.0));
    assert_eq!(cache.expires_at("counter").unwrap(), secs(121));
}

/// Python `InMemoryCache.set_cache` runs `evict_cache()` before every insert, and step 2 evicts
/// the earliest expiry while `len(cache_dict) >= max_size_in_memory`, even when the key being
/// written already exists.
#[rstest]
fn overwriting_an_existing_key_at_capacity_evicts_the_earliest_expiry_like_python(clock: Clock) {
    let cache = cache(&clock, 2);
    cache.set_cache("hot", "1".into(), secs(10)).unwrap();
    cache.set_cache("cold", "2".into(), secs(20)).unwrap();

    cache.set_cache("cold", "3".into(), None).unwrap();

    assert_eq!(cache.get_cache("hot").unwrap(), None);
    assert_eq!(cache.get_cache("cold").unwrap(), Some("3".into()));
}

/// `claim_cache` has no Python counterpart; it never evicts another entry for a key it holds.
#[rstest]
fn claiming_an_existing_key_at_capacity_keeps_other_entries(clock: Clock) {
    let cache = cache(&clock, 2);
    cache.set_cache("hot", "1".into(), secs(10)).unwrap();
    cache.set_cache("cold", "2".into(), secs(20)).unwrap();

    cache
        .claim_cache("cold", "4".into(), &[], ExactCacheContext::default())
        .unwrap();
    assert_eq!(cache.get_cache("hot").unwrap(), Some("1".into()));
    assert_eq!(cache.get_cache("cold").unwrap(), Some("2".into()));
}

/// Python `increment_cache` is `get_cache` then `set_cache`, so at capacity the write evicts
/// the earliest expiry first: equal expiries tie-break on the key, and the value read before
/// eviction is the one written back.
#[rstest]
fn incrementing_at_capacity_evicts_the_earliest_expiry_like_python(clock: Clock) {
    let cache = cache_with::<f64>(&clock, 2);
    for key in ["a", "b", "a", "b"] {
        cache
            .increment_cache(key, 1.0, ExactCacheContext::default())
            .unwrap();
    }
    assert_eq!(cache.get_cache("a").unwrap(), None);
    assert_eq!(cache.get_cache("b").unwrap(), Some(2.0));
}

#[rstest]
#[tokio::test]
async fn disabled_cache_does_not_retain_claims_counters_or_sets() {
    let claims = InMemoryCache::<String>::new(Some(0), None);
    assert_eq!(
        claims
            .claim_cache("key", "first".into(), &[], ExactCacheContext::default())
            .unwrap(),
        "first"
    );
    assert_eq!(claims.get_cache("key").unwrap(), None);

    let counters = InMemoryCache::<f64>::new(Some(0), None);
    assert_eq!(
        counters
            .increment_cache("key", 2.0, ExactCacheContext::default())
            .unwrap(),
        2.0
    );
    assert_eq!(counters.get_cache("key").unwrap(), None);

    let sets = InMemoryCache::<HashSet<String>>::new(Some(0), None);
    assert_eq!(
        sets.async_set_cache_sadd("key", vec!["a".into()], None)
            .await
            .unwrap(),
        ["a"]
    );
    assert_eq!(sets.get_cache("key").unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn ttl_and_oldest_key_operations_use_the_stored_expirations(clock: Clock) {
    let cache = cache(&clock, 3);
    cache.set_cache("later", "2".into(), secs(20)).unwrap();
    cache.set_cache("first", "1".into(), secs(10)).unwrap();
    cache.set_cache("latest", "3".into(), secs(30)).unwrap();

    assert_eq!(cache.async_get_ttl("first").await.unwrap(), secs(110));
    assert_eq!(
        TtlCache::async_get_ttl(&cache, "later").await.unwrap(),
        secs(120)
    );
    assert_eq!(cache.async_get_oldest_n_keys(1).await.unwrap(), ["first"]);
    assert_eq!(
        cache.async_get_oldest_n_keys(10).await.unwrap(),
        ["first", "later", "latest"]
    );
    assert_eq!(
        cache.async_get_oldest_n_keys(0).await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(cache.async_get_ttl("missing").await.unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn increment_pipeline_preserves_operation_order(clock: Clock) {
    let cache = cache_with::<f64>(&clock, 3);
    let operation = |key: &str, amount, ttl| IncrementOperation {
        key: key.into(),
        amount,
        ttl: secs(ttl),
    };
    assert_eq!(
        cache
            .async_increment_pipeline(vec![
                operation("a", 1.0, 10),
                operation("b", 5.0, 30),
                operation("a", 2.0, 20),
            ])
            .await
            .unwrap(),
        [1.0, 5.0, 3.0]
    );
    assert_eq!(cache.get_cache("a").unwrap(), Some(3.0));
    assert_eq!(cache.expires_at("a").unwrap(), secs(110));
    assert_eq!(cache.expires_at("b").unwrap(), secs(130));
    assert_eq!(
        cache.async_increment_pipeline(Vec::new()).await.unwrap(),
        Vec::<f64>::new()
    );
}

#[rstest]
#[tokio::test]
async fn set_capability_preserves_python_result_and_deduplicates_storage(clock: Clock) {
    let cache = cache_with::<HashSet<String>>(&clock, 4);
    let inserted = vec!["a".to_string(), "a".into(), "b".into()];
    assert_eq!(
        cache
            .async_set_cache_sadd("members", inserted.clone(), secs(10))
            .await
            .unwrap(),
        inserted
    );
    assert_eq!(
        cache
            .async_set_cache_sadd("members", vec!["c".into()], secs(99))
            .await
            .unwrap(),
        ["c"]
    );
    assert_eq!(
        cache.get_cache("members").unwrap(),
        Some(HashSet::from(["a".into(), "b".into(), "c".into()]))
    );
    assert_eq!(cache.expires_at("members").unwrap(), secs(110));
    at(&clock, 111);
    cache
        .async_set_cache_sadd("members", vec!["d".into()], None)
        .await
        .unwrap();
    assert_eq!(
        cache.get_cache("members").unwrap(),
        Some(HashSet::from(["d".into()]))
    );
    assert_eq!(cache.expires_at("members").unwrap(), secs(171));
}

#[rstest]
#[tokio::test]
async fn oversized_set_additions_are_not_stored() {
    let cache = InMemoryCache::<HashSet<String>>::with_clock_and_size_measurement(
        Some(4),
        None,
        Some(2),
        Some(Arc::new(|value: &HashSet<String>| Ok(value.len()))),
        || Duration::ZERO,
    );
    cache
        .async_set_cache_sadd("members", vec!["a".into(), "b".into()], None)
        .await
        .unwrap();
    assert_eq!(
        cache
            .async_set_cache_sadd("members", vec!["c".into()], None)
            .await
            .unwrap(),
        ["c"]
    );
    assert_eq!(
        cache.get_cache("members").unwrap(),
        Some(HashSet::from(["a".into(), "b".into()]))
    );
}
