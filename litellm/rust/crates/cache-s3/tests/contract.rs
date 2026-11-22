mod support;

use litellm_cache::ExactCacheContext;
use litellm_cache_testing as contract;
use rstest::{fixture, rstest};
use serde_json::json;
use support::{FakeBucket, JsonS3Cache};
use wiremock::MockServer;

struct S3 {
    cache: JsonS3Cache,
    _server: MockServer,
}

#[fixture]
async fn s3() -> S3 {
    let server = FakeBucket::serve().await;
    S3 {
        cache: support::cache(&server.uri()),
        _server: server,
    }
}

#[fixture]
fn context() -> ExactCacheContext {
    ExactCacheContext::default()
}

const PREFIX: &str = "contract:";

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hit_and_miss(#[future(awt)] s3: S3, context: ExactCacheContext) {
    contract::hit_and_miss(&s3.cache, context, PREFIX, json!({"answer": 42})).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_async_equivalence(#[future(awt)] s3: S3, context: ExactCacheContext) {
    contract::sync_async_equivalence(&s3.cache, context, PREFIX, json!("first"), json!([2])).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overwrite_replaces(#[future(awt)] s3: S3, context: ExactCacheContext) {
    contract::overwrite_replaces(&s3.cache, context, PREFIX, json!(1), json!({"b": 2})).await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_writes_every_entry(#[future(awt)] s3: S3, context: ExactCacheContext) {
    contract::pipeline_writes_every_entry(
        &s3.cache,
        context,
        PREFIX,
        vec![json!("a"), json!(2), json!({"c": true})],
    )
    .await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_preserves_order(#[future(awt)] s3: S3, context: ExactCacheContext) {
    contract::batch_preserves_order(&s3.cache, context, PREFIX, json!("first"), json!(2)).await;
}
