mod support;

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::{
    BaseCache, Error, SemanticCacheContext,
    semantic::{SemanticCache, SemanticLookup},
};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheControls, CacheEntry, CacheKeyField, CacheKeyInput, CacheKeyParticipation, PendingWrite,
    ResponseCache, ResponseCacheRequest, WriteBuffer, get_cache_key,
};
use redis_test::MockCmd;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::{keyed, memory, redis, request};

type Memory = Arc<ResponseCache<InMemoryCache<CacheEntry>>>;

#[rstest]
#[tokio::test]
async fn batch_lookup_reports_partial_hits_and_batch_store_populates_misses(
    memory: Memory,
    #[values(false, true)] asynchronous: bool,
) {
    let now = Duration::from_secs(100);
    let mut requests = ["hit", "miss", "disabled"].map(keyed).to_vec();
    memory
        .store(&requests[0], json!({"value": 1}), now)
        .unwrap();
    requests[2].controls.caching = Some(false);

    let partial = if asynchronous {
        memory.async_lookup_batch(&requests, now).await.unwrap()
    } else {
        memory.lookup_batch(&requests, now).unwrap()
    };
    assert_eq!(partial.values, vec![Some(json!({"value": 1})), None, None]);
    assert_eq!(partial.missing_indices(), vec![1, 2]);

    memory
        .async_store_batch(
            vec![
                (requests[1].clone(), json!({"value": 2})),
                (requests[2].clone(), json!({"value": 3})),
            ],
            now,
        )
        .await
        .unwrap();
    assert_eq!(
        memory.lookup(&requests[1], now).unwrap(),
        Some(json!({"value": 2}))
    );
    requests[2].controls.caching = None;
    assert_eq!(memory.lookup(&requests[2], now).unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn batch_lookup_with_no_readable_request_skips_the_backend(
    #[values(false, true)] asynchronous: bool,
) {
    let cache = redis(Vec::new(), None);
    let mut request = keyed("key");
    request.controls.no_cache = true;
    let requests = [request.clone(), request];
    let partial = if asynchronous {
        cache
            .async_lookup_batch(&requests, Duration::ZERO)
            .await
            .unwrap()
    } else {
        cache.lookup_batch(&requests, Duration::ZERO).unwrap()
    };
    assert_eq!(partial.values, vec![None, None]);
    assert_eq!(partial.missing_indices(), vec![0, 1]);
}

#[rstest]
#[tokio::test]
async fn deferred_entries_keep_the_time_they_were_produced(
    memory: Memory,
    mut request: ResponseCacheRequest,
) {
    request.max_age = Some(Duration::from_secs(10));
    memory
        .async_store_entries(vec![PendingWrite {
            request: request.clone(),
            response: json!({"answer": 7}),
            produced_at: Duration::from_secs(100),
        }])
        .await
        .unwrap();

    assert_eq!(
        memory.lookup(&request, Duration::from_secs(110)).unwrap(),
        Some(json!({"answer": 7}))
    );
    assert_eq!(
        memory.lookup(&request, Duration::from_secs(111)).unwrap(),
        None
    );
}
