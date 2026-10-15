//! The shared cache contracts, run against the in-process fake connection and, when
//! `LITELLM_TEST_REDIS_CLUSTER_NODES` is set, against a live Redis Cluster.

mod support;

use std::time::Duration;

use litellm_cache::{ExactCacheContext, JsonCodec};
use litellm_cache_testing as contract;
use rstest::rstest;
use serde_json::json;
use support::{JsonCache, cluster_cache, fake_cache};

const PREFIX: &str = "contract:";

#[derive(Clone, Copy, Debug)]
enum Contract {
    HitAndMiss,
    SyncAsyncEquivalence,
    OverwriteReplaces,
    PipelineWritesEveryEntry,
    BatchPreservesOrder,
    DeleteRemovesKey,
    FlushClears,
    CounterAccumulates,
}

#[derive(Clone, Copy, Debug)]
enum Server {
    Fake,
    Cluster,
}

async fn check<C>(contract: Contract, cache: &JsonCache<C>)
where
    C: redis::ConnectionLike + Send + 'static,
{
    let context = ExactCacheContext::default();
    match contract {
        Contract::HitAndMiss => {
            contract::hit_and_miss(cache, context, PREFIX, json!({"answer": 42})).await
        }
        Contract::SyncAsyncEquivalence => {
            contract::sync_async_equivalence(cache, context, PREFIX, json!("first"), json!([2]))
                .await
        }
        Contract::OverwriteReplaces => {
            contract::overwrite_replaces(cache, context, PREFIX, json!(1), json!({"b": 2})).await
        }
        Contract::PipelineWritesEveryEntry => {
            contract::pipeline_writes_every_entry(
                cache,
                context,
                PREFIX,
                vec![json!("a"), json!(2), json!({"c": true})],
            )
            .await
        }
        Contract::BatchPreservesOrder => {
            contract::batch_preserves_order(cache, context, PREFIX, json!("first"), json!(2)).await
        }
        Contract::DeleteRemovesKey => {
            contract::delete_removes_key(cache, context, PREFIX, json!("value")).await
        }
        Contract::FlushClears => {
            contract::flush_clears(cache, context, PREFIX, json!("value")).await
        }
        Contract::CounterAccumulates => contract::counter_accumulates(cache, context, PREFIX).await,
    }
}

#[rstest]
#[case::hit_and_miss(Contract::HitAndMiss)]
#[case::sync_async_equivalence(Contract::SyncAsyncEquivalence)]
#[case::overwrite_replaces(Contract::OverwriteReplaces)]
#[case::pipeline_writes_every_entry(Contract::PipelineWritesEveryEntry)]
#[case::batch_preserves_order(Contract::BatchPreservesOrder)]
#[case::delete_removes_key(Contract::DeleteRemovesKey)]
#[case::flush_clears(Contract::FlushClears)]
#[case::counter_accumulates(Contract::CounterAccumulates)]
#[tokio::test]
async fn redis_satisfies_the_cache_contract(
    #[case] contract: Contract,
    #[values(Server::Fake, Server::Cluster)] server: Server,
) {
    match server {
        Server::Fake => check(contract, &fake_cache("contract")).await,
        Server::Cluster => {
            let label = format!("{contract:?}");
            if let Some(cache) = cluster_cache(&label, Duration::from_secs(120), JsonCodec::new()) {
                check(contract, &cache).await;
            }
        }
    }
}
