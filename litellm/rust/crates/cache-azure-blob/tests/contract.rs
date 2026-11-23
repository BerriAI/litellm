mod support;

use litellm_cache::{ExactCacheContext, JsonCodec};
use litellm_cache_azure_blob::AzureBlobCache;
use litellm_cache_testing as contract;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::{ACCOUNT_URL, FakeBlobService};
use tokio::runtime::Handle;

#[fixture]
async fn azure() -> AzureBlobCache<JsonCodec<Value>> {
    support::connect(
        &FakeBlobService::default(),
        ACCOUNT_URL,
        JsonCodec::new(),
        Handle::current(),
    )
    .await
    .unwrap()
}

#[fixture]
fn context() -> ExactCacheContext {
    ExactCacheContext::default()
}

const PREFIX: &str = "contract:";

// `overwrite_replaces` does not apply: sync `set_cache` never overwrites a blob, as in Python.

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn hit_and_miss(
    #[future(awt)] azure: AzureBlobCache<JsonCodec<Value>>,
    context: ExactCacheContext,
) {
    contract::hit_and_miss(&azure, context, PREFIX, json!({"answer": 42})).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn sync_async_equivalence(
    #[future(awt)] azure: AzureBlobCache<JsonCodec<Value>>,
    context: ExactCacheContext,
) {
    contract::sync_async_equivalence(&azure, context, PREFIX, json!("first"), json!([2])).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn pipeline_writes_every_entry(
    #[future(awt)] azure: AzureBlobCache<JsonCodec<Value>>,
    context: ExactCacheContext,
) {
    contract::pipeline_writes_every_entry(
        &azure,
        context,
        PREFIX,
        vec![json!("a"), json!(2), json!({"c": true})],
    )
    .await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn batch_preserves_order(
    #[future(awt)] azure: AzureBlobCache<JsonCodec<Value>>,
    context: ExactCacheContext,
) {
    contract::batch_preserves_order(&azure, context, PREFIX, json!("first"), json!(2)).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn flush_clears(
    #[future(awt)] azure: AzureBlobCache<JsonCodec<Value>>,
    context: ExactCacheContext,
) {
    contract::flush_clears(&azure, context, PREFIX, json!("value")).await;
}
