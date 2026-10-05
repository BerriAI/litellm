mod support;

use litellm_cache::{JsonCodec, SemanticCacheContext, semantic::PreparedEmbedding};
use litellm_cache_redis_semantic::{DEFAULT_INDEX_NAME, RedisSemanticCache, RedisSemanticConfig};
use litellm_cache_testing as contract;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::FakeSearch;

type Cache = RedisSemanticCache<PreparedEmbedding, JsonCodec<Value>, FakeSearch>;

const PREFIX: &str = "contract:";

#[fixture]
fn cache() -> Cache {
    RedisSemanticCache::with_connection(
        FakeSearch::default(),
        PreparedEmbedding(vec![0.6, 0.8]),
        JsonCodec::new(),
        RedisSemanticConfig {
            index_name: DEFAULT_INDEX_NAME.into(),
            similarity_threshold: 0.9,
        },
    )
}

#[fixture]
fn context() -> SemanticCacheContext {
    SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": "contract prompt"}])),
        ..Default::default()
    }
}

#[rstest]
#[tokio::test]
async fn hit_and_miss(cache: Cache, context: SemanticCacheContext) {
    contract::hit_and_miss(&cache, context, PREFIX, json!({"answer": 42})).await;
}

#[rstest]
#[tokio::test]
async fn sync_async_equivalence(cache: Cache, context: SemanticCacheContext) {
    contract::sync_async_equivalence(&cache, context, PREFIX, json!("first"), json!([2])).await;
}

#[rstest]
#[tokio::test]
async fn overwrite_replaces(cache: Cache, context: SemanticCacheContext) {
    contract::overwrite_replaces(&cache, context, PREFIX, json!(1), json!({"b": 2})).await;
}

#[rstest]
#[tokio::test]
async fn pipeline_writes_every_entry(cache: Cache, context: SemanticCacheContext) {
    contract::pipeline_writes_every_entry(
        &cache,
        context,
        PREFIX,
        vec![json!("a"), json!(2), json!({"c": true})],
    )
    .await;
}
