mod support;

use std::{sync::Arc, time::Duration};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{BatchLookup, CacheEntry, CacheKey, PendingWrite, ResponseCache};
use rstest::rstest;
use serde_json::json;
use support::{DEFAULT, key, keyed, memory, redis};

type Memory = Arc<ResponseCache<InMemoryCache<CacheEntry>>>;

#[rstest]
#[tokio::test]
async fn batch_lookup_reports_partial_hits_and_batch_store_populates_misses(
    memory: Memory,
    #[values(false, true)] asynchronous: bool,
) {
    let now = Duration::from_secs(100);
    let keys = ["hit", "miss"].map(keyed);
    memory
        .store(&keys[0], &DEFAULT, json!({"value": 1}), now)
        .unwrap();

    let partial = if asynchronous {
        memory
            .async_lookup_batch(&keys, &DEFAULT, None, now)
            .await
            .unwrap()
    } else {
        memory.lookup_batch(&keys, &DEFAULT, None, now).unwrap()
    };
    assert_eq!(partial.values, vec![Some(json!({"value": 1})), None]);
    assert_eq!(partial.missing_indices(), vec![1]);

    memory
        .async_store_batch(vec![(keys[1].clone(), json!({"value": 2}))], DEFAULT, now)
        .await
        .unwrap();
    assert_eq!(
        memory.lookup(&keys[1], &DEFAULT, None, now).unwrap(),
        Some(json!({"value": 2}))
    );
}

#[rstest]
#[tokio::test]
async fn an_empty_batch_lookup_skips_the_backend(#[values(false, true)] asynchronous: bool) {
    let cache = redis(Vec::new(), None);
    let partial = if asynchronous {
        cache
            .async_lookup_batch(&[], &DEFAULT, None, Duration::ZERO)
            .await
            .unwrap()
    } else {
        cache
            .lookup_batch(&[], &DEFAULT, None, Duration::ZERO)
            .unwrap()
    };
    assert!(partial.values.is_empty());
}

#[rstest]
#[tokio::test]
async fn deferred_entries_keep_the_time_they_were_produced(memory: Memory, key: CacheKey) {
    let max_age = Some(Duration::from_secs(10));
    memory
        .async_store_entries(vec![PendingWrite {
            key: key.clone(),
            context: DEFAULT,
            response: json!({"answer": 7}),
            produced_at: Duration::from_secs(100),
        }])
        .await
        .unwrap();

    assert_eq!(
        memory
            .lookup(&key, &DEFAULT, max_age, Duration::from_secs(110))
            .unwrap(),
        Some(json!({"answer": 7}))
    );
    assert_eq!(
        memory
            .lookup(&key, &DEFAULT, max_age, Duration::from_secs(111))
            .unwrap(),
        None
    );
}

#[rstest]
fn generic_batch_results_do_not_require_clone() {
    struct NonClone;
    let misses = BatchLookup::<NonClone>::misses(3);
    assert_eq!(misses.missing_indices(), vec![0, 1, 2]);
    let mixed = BatchLookup {
        values: vec![Some(7_u32), None, Some(9)],
    };
    assert_eq!(
        serde_json::to_value(mixed).unwrap(),
        json!({"values": [7, null, 9], "missing_indices": [1]}),
    );
}
