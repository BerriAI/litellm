use std::time::Duration;

use litellm_cache::ExactCacheContext;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_testing as contract;
use rstest::{fixture, rstest};

#[fixture]
fn strings() -> InMemoryCache<String> {
    InMemoryCache::new(Some(16), None)
}

#[fixture]
fn counters() -> InMemoryCache<f64> {
    InMemoryCache::new(Some(16), None)
}

#[fixture]
fn context() -> ExactCacheContext {
    ExactCacheContext {
        ttl: Some(Duration::from_secs(60)),
    }
}

#[rstest]
#[tokio::test]
async fn hit_and_miss(strings: InMemoryCache<String>, context: ExactCacheContext) {
    contract::hit_and_miss(&strings, context, "memory:", "value".into()).await;
}

#[rstest]
#[tokio::test]
async fn sync_async_equivalence(strings: InMemoryCache<String>, context: ExactCacheContext) {
    contract::sync_async_equivalence(
        &strings,
        context,
        "memory:",
        "first".into(),
        "second".into(),
    )
    .await;
}

#[rstest]
#[tokio::test]
async fn overwrite_replaces(strings: InMemoryCache<String>, context: ExactCacheContext) {
    contract::overwrite_replaces(
        &strings,
        context,
        "memory:",
        "first".into(),
        "second".into(),
    )
    .await;
}

#[rstest]
#[tokio::test]
async fn pipeline_writes_every_entry(strings: InMemoryCache<String>, context: ExactCacheContext) {
    contract::pipeline_writes_every_entry(
        &strings,
        context,
        "memory:",
        vec!["a".into(), "b".into(), "c".into()],
    )
    .await;
}

#[rstest]
#[tokio::test]
async fn batch_preserves_order(strings: InMemoryCache<String>, context: ExactCacheContext) {
    contract::batch_preserves_order(
        &strings,
        context,
        "memory:",
        "first".into(),
        "second".into(),
    )
    .await;
}

#[rstest]
#[tokio::test]
async fn delete_removes_key(strings: InMemoryCache<String>, context: ExactCacheContext) {
    contract::delete_removes_key(&strings, context, "memory:", "value".into()).await;
}

#[rstest]
#[tokio::test]
async fn flush_clears(strings: InMemoryCache<String>, context: ExactCacheContext) {
    contract::flush_clears(&strings, context, "memory:", "value".into()).await;
}

#[rstest]
#[tokio::test]
async fn counter_accumulates(counters: InMemoryCache<f64>, context: ExactCacheContext) {
    contract::counter_accumulates(&counters, context, "memory:").await;
}
