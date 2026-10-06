mod support;

use std::{sync::Arc, time::Duration};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheEntry, ResponseCache, ResponseCacheRequest, WriteBuffer};
use rstest::rstest;
use serde_json::json;
use support::{keyed, memory, request};

type Memory = Arc<ResponseCache<InMemoryCache<CacheEntry>>>;

#[rstest]
#[tokio::test]
async fn write_buffer_flushes_at_its_size_and_keeps_each_produced_time(
    memory: Memory,
    request: ResponseCacheRequest,
) {
    let buffer = WriteBuffer::new(2);
    let mut first = request;
    first.max_age = Some(Duration::from_secs(10));
    let second = keyed("tenant:other");

    buffer
        .async_store(
            memory.as_ref(),
            &first,
            json!({"answer": 7}),
            Duration::from_secs(100),
        )
        .await
        .unwrap();
    assert_eq!(
        memory.lookup(&first, Duration::from_secs(100)).unwrap(),
        None
    );

    buffer
        .async_store(
            memory.as_ref(),
            &second,
            json!({"answer": 8}),
            Duration::from_secs(200),
        )
        .await
        .unwrap();
    assert_eq!(
        memory.lookup(&first, Duration::from_secs(110)).unwrap(),
        Some(json!({"answer": 7}))
    );
    assert_eq!(
        memory.lookup(&first, Duration::from_secs(111)).unwrap(),
        None
    );
    assert_eq!(
        memory.lookup(&second, Duration::from_secs(200)).unwrap(),
        Some(json!({"answer": 8}))
    );
}

#[rstest]
#[tokio::test]
async fn write_buffer_clear_drops_pending_entries(memory: Memory, request: ResponseCacheRequest) {
    let buffer = WriteBuffer::new(2);
    let other = keyed("tenant:other");
    let now = Duration::from_secs(100);

    buffer
        .async_store(memory.as_ref(), &request, json!({"answer": 7}), now)
        .await
        .unwrap();
    buffer.clear().unwrap();
    buffer
        .async_store(memory.as_ref(), &other, json!({"answer": 8}), now)
        .await
        .unwrap();

    assert_eq!(memory.lookup(&request, now).unwrap(), None);
    assert_eq!(memory.lookup(&other, now).unwrap(), None);
}
