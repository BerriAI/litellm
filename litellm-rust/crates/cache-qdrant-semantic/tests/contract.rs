//! `overwrite_replaces` does not apply: like Python, every write upserts a new `uuid4` point,
//! so a second write with the same prompt adds a tie instead of replacing the first.

mod support;

use std::future::Future;

use litellm_cache::{JsonCodec, SemanticCacheContext, semantic::PreparedEmbedding};
use litellm_cache_qdrant_semantic::{QdrantSemanticCache, QdrantSemanticConfig, Quantization};
use litellm_cache_testing as contract;
use qdrant_client::Qdrant;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::{FakeQdrant, FakeState};

type Cache = QdrantSemanticCache<PreparedEmbedding, JsonCodec<Value>>;

const PREFIX: &str = "contract:";

#[fixture]
fn context() -> SemanticCacheContext {
    SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": "contract prompt"}])),
        ..Default::default()
    }
}

/// Runs a contract against a fresh fake Qdrant. The sync cache methods block on the runtime, so
/// the contract is polled on a blocking thread outside the runtime's own executor.
async fn run<F, Fut>(check: F)
where
    F: FnOnce(Cache) -> Fut + Send + 'static,
    Fut: Future<Output = ()>,
{
    let server = FakeQdrant::start(FakeState::default()).await;
    let runtime = tokio::runtime::Handle::current();
    let cache = QdrantSemanticCache::connect(
        Qdrant::from_url(&server.url()).build().unwrap(),
        PreparedEmbedding(vec![0.6, 0.8]),
        JsonCodec::new(),
        QdrantSemanticConfig {
            collection_name: "contract".to_owned(),
            similarity_threshold: 0.9,
            vector_size: 2,
            quantization: Quantization::Binary,
        },
        runtime.clone(),
    )
    .await
    .unwrap();
    tokio::task::spawn_blocking(move || {
        let _guard = runtime.enter();
        futures_executor::block_on(check(cache));
    })
    .await
    .unwrap();
    server.stop();
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn hit_and_miss(context: SemanticCacheContext) {
    run(|cache| async move {
        contract::hit_and_miss(&cache, context, PREFIX, json!({"answer": 42})).await;
    })
    .await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn sync_async_equivalence(context: SemanticCacheContext) {
    run(|cache| async move {
        contract::sync_async_equivalence(&cache, context, PREFIX, json!("first"), json!([2])).await;
    })
    .await;
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn pipeline_writes_every_entry(context: SemanticCacheContext) {
    run(|cache| async move {
        contract::pipeline_writes_every_entry(
            &cache,
            context,
            PREFIX,
            vec![json!("a"), json!(2), json!({"c": true})],
        )
        .await;
    })
    .await;
}
