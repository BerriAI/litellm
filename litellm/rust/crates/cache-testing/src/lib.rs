//! Backend-neutral contract checks every cache backend runs from its own `rstest` suite.
//!
//! Each check takes the cache under test, the context to call it with, a key `prefix` that
//! keeps runs apart on shared servers, and distinct sample values. A check panics with the
//! violated invariant, so a backend test is one `#[rstest]` case per contract.

use std::fmt::Debug;

use litellm_cache::{BaseCache, BatchCache, BatchEntry, CounterCache, DeleteCache, FlushCache};

fn key(prefix: &str, name: &str) -> String {
    format!("{prefix}{name}")
}

/// A missing key reads as `None`, and a written key reads back through sync and async gets.
pub async fn hit_and_miss<B>(cache: &B, context: B::Context, prefix: &str, value: B::Value)
where
    B: BaseCache,
    B::Value: Debug + PartialEq,
{
    let key = key(prefix, "hit-and-miss");
    assert_eq!(
        cache.get_cache(&key, &context).unwrap(),
        None,
        "unwritten key must miss"
    );
    assert_eq!(
        cache.async_get_cache(&key, &context).await.unwrap(),
        None,
        "unwritten key must miss asynchronously"
    );
    cache.set_cache(&key, value.clone(), &context).unwrap();
    assert_eq!(
        cache.get_cache(&key, &context).unwrap(),
        Some(value.clone())
    );
    assert_eq!(
        cache.async_get_cache(&key, &context).await.unwrap(),
        Some(value)
    );
}

/// Sync and async writes land in the same store: each is visible to the other read path.
pub async fn sync_async_equivalence<B>(
    cache: &B,
    context: B::Context,
    prefix: &str,
    first: B::Value,
    second: B::Value,
) where
    B: BaseCache,
    B::Value: Debug + PartialEq,
{
    let async_written = key(prefix, "async-written");
    let sync_written = key(prefix, "sync-written");
    cache
        .async_set_cache(&async_written, first.clone(), context.clone())
        .await
        .unwrap();
    assert_eq!(
        cache.get_cache(&async_written, &context).unwrap(),
        Some(first)
    );
    cache
        .set_cache(&sync_written, second.clone(), &context)
        .unwrap();
    assert_eq!(
        cache
            .async_get_cache(&sync_written, &context)
            .await
            .unwrap(),
        Some(second)
    );
}

/// A second write to a key replaces the first.
pub async fn overwrite_replaces<B>(
    cache: &B,
    context: B::Context,
    prefix: &str,
    first: B::Value,
    second: B::Value,
) where
    B: BaseCache,
    B::Value: Debug + PartialEq,
{
    let key = key(prefix, "overwrite");
    cache.set_cache(&key, first, &context).unwrap();
    cache.set_cache(&key, second.clone(), &context).unwrap();
    assert_eq!(cache.get_cache(&key, &context).unwrap(), Some(second));
}

/// `async_set_cache_pipeline` writes every entry, and an empty pipeline succeeds.
pub async fn pipeline_writes_every_entry<B>(
    cache: &B,
    context: B::Context,
    prefix: &str,
    values: Vec<B::Value>,
) where
    B: BaseCache,
    B::Value: Debug + PartialEq,
{
    cache
        .async_set_cache_pipeline(Vec::new(), context.clone())
        .await
        .unwrap();
    let entries = values
        .iter()
        .enumerate()
        .map(|(index, value)| (key(prefix, &format!("pipeline-{index}")), value.clone()))
        .collect::<Vec<_>>();
    cache
        .async_set_cache_pipeline(entries.clone(), context.clone())
        .await
        .unwrap();
    for (key, value) in entries {
        assert_eq!(
            cache.get_cache(&key, &context).unwrap(),
            Some(value),
            "{key}"
        );
    }
}

/// Batch reads answer in request order, with a `Miss` in place of each absent key.
pub async fn batch_preserves_order<B>(
    cache: &B,
    context: B::Context,
    prefix: &str,
    first: B::Value,
    second: B::Value,
) where
    B: BatchCache,
    B::Value: Debug + PartialEq,
{
    let keys = vec![
        key(prefix, "batch-first"),
        key(prefix, "batch-missing"),
        key(prefix, "batch-second"),
    ];
    cache.set_cache(&keys[0], first.clone(), &context).unwrap();
    cache.set_cache(&keys[2], second.clone(), &context).unwrap();
    let expected = vec![
        BatchEntry::Hit(first),
        BatchEntry::Miss,
        BatchEntry::Hit(second),
    ];
    assert_eq!(cache.batch_get_cache(&keys, &context).unwrap(), expected);
    assert_eq!(
        cache.async_batch_get_cache(keys, context).await.unwrap(),
        expected
    );
}

/// Sync and async deletes remove only the named key, and deleting a missing key succeeds.
pub async fn delete_removes_key<B>(cache: &B, context: B::Context, prefix: &str, value: B::Value)
where
    B: DeleteCache,
    B::Value: Debug + PartialEq,
{
    let sync_deleted = key(prefix, "delete-sync");
    let async_deleted = key(prefix, "delete-async");
    let kept = key(prefix, "delete-kept");
    for key in [&sync_deleted, &async_deleted, &kept] {
        cache.set_cache(key, value.clone(), &context).unwrap();
    }
    cache.delete_cache(&sync_deleted).unwrap();
    cache.async_delete_cache(&async_deleted).await.unwrap();
    cache
        .delete_cache(&key(prefix, "delete-never-written"))
        .unwrap();
    assert_eq!(cache.get_cache(&sync_deleted, &context).unwrap(), None);
    assert_eq!(cache.get_cache(&async_deleted, &context).unwrap(), None);
    assert_eq!(cache.get_cache(&kept, &context).unwrap(), Some(value));
}

/// `flush_cache` and `async_flush_cache` each leave the cache empty.
pub async fn flush_clears<B>(cache: &B, context: B::Context, prefix: &str, value: B::Value)
where
    B: FlushCache,
    B::Value: Debug + PartialEq,
{
    let key = key(prefix, "flush");
    cache.set_cache(&key, value.clone(), &context).unwrap();
    cache.flush_cache().unwrap();
    assert_eq!(cache.get_cache(&key, &context).unwrap(), None);
    cache.set_cache(&key, value, &context).unwrap();
    cache.async_flush_cache().await.unwrap();
    assert_eq!(cache.get_cache(&key, &context).unwrap(), None);
}

/// Sync and async increments accumulate on one counter, starting from zero. Whole-number
/// steps, since Python's disk cache restarts any counter whose stored value is not an `int`.
pub async fn counter_accumulates<B>(cache: &B, context: B::Context, prefix: &str)
where
    B: CounterCache,
{
    let key = key(prefix, "counter");
    assert_eq!(
        cache.increment_cache(&key, 1.0, context.clone()).unwrap(),
        1.0
    );
    assert_eq!(
        cache
            .async_increment(&key, 2.0, context.clone(), false)
            .await
            .unwrap(),
        3.0
    );
    assert_eq!(cache.increment_cache(&key, -1.0, context).unwrap(), 2.0);
}
