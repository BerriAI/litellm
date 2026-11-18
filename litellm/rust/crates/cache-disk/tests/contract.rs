use litellm_cache::{ExactCacheContext, JsonCodec};
use litellm_cache_disk::DiskCache;
use litellm_cache_testing as contract;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tempfile::TempDir;

struct Disk {
    cache: DiskCache<JsonCodec<Value>>,
    _directory: TempDir,
}

#[fixture]
fn disk() -> Disk {
    let directory = tempfile::tempdir().unwrap();
    Disk {
        cache: DiskCache::open(directory.path(), JsonCodec::new()).unwrap(),
        _directory: directory,
    }
}

#[fixture]
fn context() -> ExactCacheContext {
    ExactCacheContext::default()
}

const PREFIX: &str = "contract:";

#[rstest]
#[tokio::test]
async fn hit_and_miss(disk: Disk, context: ExactCacheContext) {
    contract::hit_and_miss(&disk.cache, context, PREFIX, json!({"answer": 42})).await;
}

#[rstest]
#[tokio::test]
async fn sync_async_equivalence(disk: Disk, context: ExactCacheContext) {
    contract::sync_async_equivalence(&disk.cache, context, PREFIX, json!("first"), json!([2]))
        .await;
}

#[rstest]
#[tokio::test]
async fn overwrite_replaces(disk: Disk, context: ExactCacheContext) {
    contract::overwrite_replaces(&disk.cache, context, PREFIX, json!(1), json!({"b": 2})).await;
}

#[rstest]
#[tokio::test]
async fn pipeline_writes_every_entry(disk: Disk, context: ExactCacheContext) {
    contract::pipeline_writes_every_entry(
        &disk.cache,
        context,
        PREFIX,
        vec![json!("a"), json!(2), json!({"c": true})],
    )
    .await;
}

#[rstest]
#[tokio::test]
async fn batch_preserves_order(disk: Disk, context: ExactCacheContext) {
    contract::batch_preserves_order(&disk.cache, context, PREFIX, json!("first"), json!(2)).await;
}

#[rstest]
#[tokio::test]
async fn delete_removes_key(disk: Disk, context: ExactCacheContext) {
    contract::delete_removes_key(&disk.cache, context, PREFIX, json!("value")).await;
}

#[rstest]
#[tokio::test]
async fn flush_clears(disk: Disk, context: ExactCacheContext) {
    contract::flush_clears(&disk.cache, context, PREFIX, json!("value")).await;
}

#[rstest]
#[tokio::test]
async fn counter_accumulates(disk: Disk, context: ExactCacheContext) {
    contract::counter_accumulates(&disk.cache, context, PREFIX).await;
}
