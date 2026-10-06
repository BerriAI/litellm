mod support;

use litellm_cache::ExactCacheContext;
use litellm_cache_testing as contract;
use rstest::{fixture, rstest};
use serde_json::json;
use support::{FakeBucket, JsonGcsCache};
use wiremock::MockServer;

struct Gcs {
    cache: JsonGcsCache,
    _server: MockServer,
}

#[fixture]
async fn gcs() -> Gcs {
    let server = FakeBucket::serve().await;
    Gcs {
        cache: support::cache(&server, Some("contract")),
        _server: server,
    }
}

#[fixture]
fn context() -> ExactCacheContext {
    ExactCacheContext::default()
}

const PREFIX: &str = "contract:";

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn hit_and_miss(#[future(awt)] gcs: Gcs, context: ExactCacheContext) {
    contract::hit_and_miss(&gcs.cache, context, PREFIX, json!({"answer": 42})).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn sync_async_equivalence(#[future(awt)] gcs: Gcs, context: ExactCacheContext) {
    contract::sync_async_equivalence(&gcs.cache, context, PREFIX, json!("first"), json!([2])).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn overwrite_replaces(#[future(awt)] gcs: Gcs, context: ExactCacheContext) {
    contract::overwrite_replaces(&gcs.cache, context, PREFIX, json!(1), json!({"b": 2})).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn pipeline_writes_every_entry(#[future(awt)] gcs: Gcs, context: ExactCacheContext) {
    contract::pipeline_writes_every_entry(
        &gcs.cache,
        context,
        PREFIX,
        vec![json!("a"), json!(2), json!({"c": true})],
    )
    .await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn batch_preserves_order(#[future(awt)] gcs: Gcs, context: ExactCacheContext) {
    contract::batch_preserves_order(&gcs.cache, context, PREFIX, json!("first"), json!(2)).await;
}
