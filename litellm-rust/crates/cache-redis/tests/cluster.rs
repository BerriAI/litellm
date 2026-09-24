//! Tests against a real Redis Cluster. Set `LITELLM_TEST_REDIS_CLUSTER_NODES` to a comma
//! separated `host:port` list (for example `127.0.0.1:7000,127.0.0.1:7001`) to run them.

mod support;

use std::{collections::HashSet, time::Duration};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, BoundedCounterCache, BulkDeleteCache, CacheConnectionStatus,
    CacheScript, ClaimCache, ClientInfoCache, ConnectionCache, CounterCache, DeleteCache,
    DisconnectCache, Error, ExactCacheContext, FlushCache, IncrementOperation, JsonCodec,
    PingCache, QueueCache, RefreshTtlCache, ScanCache, ScriptCache, SetCache, TtlCache,
    TtlPipelineCache,
};
use litellm_cache_redis::{
    RedisArg, RedisCache, RedisLpopOperation, RedisLpopResult, RedisNode, RedisRpushOperation,
    RedisTopology,
};
use redis::cluster_routing::Slot;
use rstest::{fixture, rstest};
use serde_json::json;
use support::{JsonCache, cluster_cache, cluster_url};

type Counter = RedisCache<JsonCodec<f64>>;

#[fixture]
fn cache(#[default("cache")] label: &str) -> Option<JsonCache> {
    cluster_cache(label, Duration::from_secs(120), JsonCodec::new())
}

#[fixture]
fn counter(#[default("counter")] label: &str) -> Option<Counter> {
    cluster_cache(label, Duration::from_secs(60), JsonCodec::new())
}

#[fixture]
fn context() -> ExactCacheContext {
    ExactCacheContext::default()
}

fn multi_slot_keys(count: usize) -> Vec<String> {
    let keys: Vec<String> = (0..count).map(|index| format!("key-{index}")).collect();
    let slots: HashSet<Slot> = keys.iter().map(Slot::for_key).collect();
    assert!(slots.len() > 1, "keys must span multiple slots");
    keys
}

fn seconds(seconds: u64) -> Option<Duration> {
    Some(Duration::from_secs(seconds))
}

#[rstest]
#[case::no_startup_nodes("redis://127.0.0.1:7000", Vec::new())]
#[case::unix_socket_url(
    "redis+unix:///tmp/redis.sock",
    vec![RedisNode { host: "127.0.0.1".into(), port: 7000 }]
)]
fn constructor_rejects_unusable_cluster_configs(
    #[case] url: &str,
    #[case] startup_nodes: Vec<RedisNode>,
) {
    let error = JsonCache::connect(
        url,
        &RedisTopology::Cluster { startup_nodes },
        None,
        JsonCodec::new(),
    )
    .err();
    assert!(matches!(error, Some(Error::Unavailable)));
}

#[rstest]
#[tokio::test]
async fn single_key_operations_round_trip_with_ttl_rounding(
    #[with("single")] cache: Option<JsonCache>,
) {
    let Some(cache) = cache else { return };
    let context = ExactCacheContext {
        ttl: Some(Duration::from_millis(1500)),
    };
    let keys = multi_slot_keys(12);
    for (index, key) in keys.iter().enumerate() {
        cache
            .set_cache(key, json!({ "index": index }), &context)
            .unwrap();
    }
    for (index, key) in keys.iter().enumerate() {
        assert_eq!(
            cache.get_cache(key, &context).unwrap(),
            Some(json!({ "index": index }))
        );
    }
    assert_eq!(cache.async_get_ttl(&keys[0]).await.unwrap(), seconds(2));
    assert!(
        cache
            .async_refresh_ttl(&keys[0], seconds(40))
            .await
            .unwrap()
    );
    assert_eq!(cache.async_get_ttl(&keys[0]).await.unwrap(), seconds(40));
    cache.delete_cache(&keys[0]).unwrap();
    assert_eq!(cache.get_cache(&keys[0], &context).unwrap(), None);
    assert!(!cache.async_refresh_ttl(&keys[0], None).await.unwrap());
    assert!(cache.sync_ping().unwrap());
    cache.async_flush_cache().await.unwrap();
}

#[rstest]
#[tokio::test]
async fn batch_reads_span_slots_and_preserve_order_with_malformed_entries(
    #[with("batch")] cache: Option<JsonCache>,
    context: ExactCacheContext,
) {
    let Some(cache) = cache else { return };
    let keys = multi_slot_keys(40);
    for (index, key) in keys.iter().enumerate() {
        if index % 5 == 0 {
            continue;
        }
        cache
            .async_set_cache(key, json!(index), context.clone())
            .await
            .unwrap();
    }
    let mut raw = redis::cluster::ClusterClient::new(vec![cluster_url()])
        .unwrap()
        .get_connection()
        .unwrap();
    let malformed = format!("{}:{}", cache.namespace().unwrap(), keys[1]);
    redis::cmd("SET")
        .arg(&malformed)
        .arg("not json")
        .exec(&mut raw)
        .unwrap();

    let entries = cache
        .async_batch_get_cache(keys.clone(), context.clone())
        .await
        .unwrap();
    assert_eq!(entries.len(), keys.len());
    for (index, entry) in entries.iter().enumerate() {
        let expected = if index == 1 {
            BatchEntry::Invalid
        } else if index % 5 == 0 {
            BatchEntry::Miss
        } else {
            BatchEntry::Hit(json!(index))
        };
        assert_eq!(*entry, expected, "entry {index}");
    }
    assert_eq!(cache.batch_get_cache(&keys, &context).unwrap(), entries);

    cache.delete_cache_keys(keys.clone()).await.unwrap();
    let entries = cache.async_batch_get_cache(keys, context).await.unwrap();
    assert!(entries.iter().all(|entry| *entry == BatchEntry::Miss));
}

#[rstest]
#[tokio::test]
async fn pipelines_group_by_slot_and_return_results_in_submission_order(
    #[with("pipeline")] cache: Option<JsonCache>,
    counter: Option<Counter>,
    context: ExactCacheContext,
) {
    let (Some(cache), Some(counter)) = (cache, counter) else {
        return;
    };
    let keys = multi_slot_keys(30);
    cache
        .async_set_cache_pipeline(
            keys.iter()
                .enumerate()
                .map(|(index, key)| (key.clone(), json!(index)))
                .collect(),
            context.clone(),
        )
        .await
        .unwrap();
    let hits = cache
        .async_batch_get_cache(keys.clone(), context.clone())
        .await
        .unwrap();
    assert!(
        hits.iter()
            .enumerate()
            .all(|(index, entry)| *entry == BatchEntry::Hit(json!(index)))
    );

    cache
        .async_set_cache_pipeline_with_ttls(
            keys.iter()
                .enumerate()
                .map(|(index, key)| (key.clone(), json!(index), seconds(index as u64 + 10)))
                .collect(),
        )
        .await
        .unwrap();
    for (index, key) in keys.iter().enumerate() {
        assert_eq!(
            cache.async_get_ttl(key).await.unwrap(),
            seconds(index as u64 + 10),
            "{key}"
        );
    }

    let queues: Vec<String> = keys.iter().map(|key| format!("queue:{key}")).collect();
    let pushed = cache
        .async_rpush_pipeline(
            queues
                .iter()
                .enumerate()
                .map(|(index, key)| RedisRpushOperation {
                    key: key.clone(),
                    values: (0..=index)
                        .map(|value| RedisArg::Integer(value as i64))
                        .collect(),
                })
                .collect(),
        )
        .await
        .unwrap();
    assert_eq!(pushed, (1..=keys.len()).collect::<Vec<_>>());
    let popped = cache
        .async_lpop_pipeline(
            queues
                .iter()
                .enumerate()
                .map(|(index, key)| RedisLpopOperation {
                    key: key.clone(),
                    count: (index % 2 == 0).then_some(2),
                })
                .collect(),
        )
        .await
        .unwrap();
    for (index, result) in popped.into_iter().enumerate() {
        match result {
            RedisLpopResult::Value(value) => {
                assert_eq!(index % 2, 1, "queue {index}");
                assert_eq!(value, b"0");
            }
            RedisLpopResult::Values(values) => {
                assert_eq!(index % 2, 0, "queue {index}");
                let expected: Vec<Vec<u8>> = (0..=index)
                    .take(2)
                    .map(|value| value.to_string().into_bytes())
                    .collect();
                assert_eq!(values, expected);
            }
            other => panic!("queue {index}: {other:?}"),
        }
    }

    let counters: Vec<String> = keys.iter().map(|key| format!("counter:{key}")).collect();
    let totals = counter
        .async_increment_pipeline(
            counters
                .iter()
                .enumerate()
                .map(|(index, key)| IncrementOperation {
                    key: key.clone(),
                    amount: index as f64 + 0.5,
                    ttl: (index % 3 == 0).then_some(Duration::from_secs(30)),
                })
                .collect(),
        )
        .await
        .unwrap();
    let expected: Vec<f64> = (0..keys.len()).map(|index| index as f64 + 0.5).collect();
    assert_eq!(totals, expected);
    assert_eq!(
        counter.async_get_ttl(&counters[0]).await.unwrap(),
        seconds(30)
    );
    assert_eq!(counter.async_get_ttl(&counters[1]).await.unwrap(), None);
    counter.async_flush_cache().await.unwrap();
    cache.async_flush_cache().await.unwrap();
}

#[rstest]
#[tokio::test]
async fn rpush_and_trim_is_one_transaction_on_the_key_slot(
    #[with("trim")] cache: Option<JsonCache>,
) {
    let Some(cache) = cache else { return };
    let values = |values: &[&str]| values.iter().map(|value| RedisArg::from(*value)).collect();
    assert_eq!(
        cache
            .async_rpush_and_trim("buf", values(&["a", "b"]), 3)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        cache
            .async_rpush_and_trim("buf", values(&["c", "d"]), 3)
            .await
            .unwrap(),
        4
    );
    assert_eq!(
        cache.async_lpop("buf", Some(10)).await.unwrap(),
        RedisLpopResult::Values(vec![b"b".to_vec(), b"c".to_vec(), b"d".to_vec()])
    );
    cache.async_flush_cache().await.unwrap();
}

#[rstest]
#[tokio::test]
async fn scan_and_scoped_flush_cover_every_primary(
    #[with("flush")] cache: Option<JsonCache>,
    #[from(cache)]
    #[with("other")]
    other: Option<JsonCache>,
    context: ExactCacheContext,
) {
    let (Some(cache), Some(other)) = (cache, other) else {
        return;
    };
    let keys = multi_slot_keys(60);
    for key in &keys {
        cache
            .async_set_cache(key, json!(true), context.clone())
            .await
            .unwrap();
        other
            .async_set_cache(key, json!(true), context.clone())
            .await
            .unwrap();
    }
    let mut scanned = cache.async_scan_iter("key-", 1000).await.unwrap();
    scanned.sort();
    let mut expected: Vec<String> = keys
        .iter()
        .map(|key| format!("{}:{key}", cache.namespace().unwrap()))
        .collect();
    expected.sort();
    assert_eq!(scanned, expected);
    assert_eq!(cache.async_scan_iter("key-", 7).await.unwrap().len(), 7);

    cache.flush_cache().unwrap();
    let flushed = cache
        .async_batch_get_cache(keys.clone(), context.clone())
        .await
        .unwrap();
    assert!(flushed.iter().all(|entry| *entry == BatchEntry::Miss));
    let kept = other.async_batch_get_cache(keys, context).await.unwrap();
    assert!(
        kept.iter()
            .all(|entry| *entry == BatchEntry::Hit(json!(true)))
    );
    other.async_flush_cache().await.unwrap();
}

fn ping_calls_per_node(startup: &redis::Client) -> Vec<(String, u64)> {
    let mut connection = startup.get_connection().unwrap();
    let nodes: String = redis::cmd("CLUSTER")
        .arg("NODES")
        .query(&mut connection)
        .unwrap();
    let mut counts: Vec<(String, u64)> = nodes
        .lines()
        .map(|line| {
            let address = line.split_whitespace().nth(1).unwrap();
            let address = address.split('@').next().unwrap();
            let mut node = redis::Client::open(format!("redis://{address}"))
                .unwrap()
                .get_connection()
                .unwrap();
            let stats: String = redis::cmd("INFO")
                .arg("commandstats")
                .query(&mut node)
                .unwrap();
            let calls = stats
                .lines()
                .find_map(|stat| stat.strip_prefix("cmdstat_ping:calls="))
                .and_then(|rest| rest.split(',').next())
                .map_or(0, |calls| calls.parse().unwrap());
            (address.to_string(), calls)
        })
        .collect();
    counts.sort();
    counts
}

#[rstest]
#[tokio::test]
async fn ping_reaches_every_node(#[with("ping")] cache: Option<JsonCache>) {
    let Some(cache) = cache else { return };
    let startup = redis::Client::open(cluster_url()).unwrap();
    let before = ping_calls_per_node(&startup);
    assert!(before.len() >= 2, "{before:?}");
    assert!(cache.ping().await.unwrap());
    let after = ping_calls_per_node(&startup);
    for ((node, calls_before), (_, calls_after)) in before.iter().zip(&after) {
        assert!(calls_after > calls_before, "{node} was not pinged");
    }
    assert!(cache.sync_ping().unwrap());
    let result = cache.test_connection().await.unwrap();
    assert_eq!(result.status, CacheConnectionStatus::Success);
    assert_eq!(result.message, "Redis Cluster connection test successful");
}

#[rstest]
#[tokio::test]
async fn disconnect_closes_idle_connections_and_reconnects_on_demand(
    #[with("disconnect")] cache: Option<JsonCache>,
) {
    let Some(cache) = cache else { return };
    assert!(cache.ping().await.unwrap());
    cache.disconnect().await.unwrap();
    assert!(cache.ping().await.unwrap());
}

#[rstest]
#[case::keep_existing_ttl(false)]
#[case::refresh_ttl(true)]
#[tokio::test]
async fn increments_refresh_the_ttl_only_when_asked(
    counter: Option<Counter>,
    #[case] refresh_ttl: bool,
) {
    let Some(counter) = counter else { return };
    let context = ExactCacheContext { ttl: seconds(60) };
    counter
        .async_set_cache("spend", 0.0, ExactCacheContext { ttl: seconds(600) })
        .await
        .unwrap();
    assert_eq!(
        counter
            .async_increment("spend", 1.5, context.clone(), refresh_ttl)
            .await
            .unwrap(),
        1.5
    );
    assert_eq!(
        counter
            .async_increment("spend", 2.0, context, refresh_ttl)
            .await
            .unwrap(),
        3.5
    );
    let ttl = counter.async_get_ttl("spend").await.unwrap().unwrap();
    assert_eq!(ttl <= Duration::from_secs(60), refresh_ttl, "{ttl:?}");
    counter.async_flush_cache().await.unwrap();
}

#[rstest]
#[tokio::test]
async fn counters_claims_scripts_and_sets_work_on_the_cluster(
    counter: Option<Counter>,
    #[with("claim")] cache: Option<JsonCache>,
    context: ExactCacheContext,
) {
    let (Some(counter), Some(cache)) = (counter, cache) else {
        return;
    };
    assert_eq!(
        counter
            .increment_cache("spend", 1.5, context.clone())
            .unwrap(),
        1.5
    );
    assert_eq!(
        counter
            .async_increment("spend", 2.0, context.clone(), false)
            .await
            .unwrap(),
        3.5
    );
    assert_eq!(
        counter
            .increment_with_floor("budget", -3, Duration::from_secs(30))
            .unwrap(),
        0
    );
    assert_eq!(
        counter
            .async_increment_with_floor("budget", 7, Duration::from_secs(30))
            .await
            .unwrap(),
        7
    );
    assert_eq!(counter.async_set_max("peak", 4.0, None).await.unwrap(), 4.0);
    assert_eq!(counter.async_set_max("peak", 2.0, None).await.unwrap(), 4.0);
    counter.flush_cache().unwrap();

    let owner = json!("owner-a");
    let rival = json!("owner-b");
    assert_eq!(
        cache
            .claim_cache("lock", owner.clone(), &[], context.clone())
            .unwrap(),
        owner
    );
    assert_eq!(
        cache
            .async_claim_cache("lock", rival.clone(), vec![owner.clone()], context.clone())
            .await
            .unwrap(),
        owner
    );
    assert_eq!(
        cache
            .claim_cache("lock", rival.clone(), &[], context.clone())
            .unwrap(),
        owner
    );
    assert_eq!(
        cache
            .async_claim_cache("lock", rival.clone(), vec![rival.clone()], context.clone())
            .await
            .unwrap(),
        rival
    );

    let script = cache
        .async_register_script("return redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])".into());
    let reply = script
        .invoke(
            vec!["scripted".into()],
            vec![RedisArg::Bytes(b"payload".to_vec()), RedisArg::Integer(5)],
        )
        .await
        .unwrap();
    assert_eq!(reply, redis::Value::Okay);
    assert_eq!(cache.async_get_ttl("scripted").await.unwrap(), seconds(5));
    let evaluated = cache
        .async_eval(
            "return redis.call('GET', KEYS[1])".into(),
            vec!["scripted".into()],
            Vec::new(),
        )
        .await
        .unwrap();
    assert_eq!(evaluated, redis::Value::BulkString(b"payload".to_vec()));

    assert_eq!(
        cache
            .async_set_cache_sadd(
                "members",
                vec![
                    RedisArg::Bytes(b"a".to_vec()),
                    RedisArg::Bytes(b"b".to_vec())
                ],
                seconds(9),
            )
            .await
            .unwrap(),
        2
    );
    assert_eq!(cache.async_get_ttl("members").await.unwrap(), seconds(9));

    let result = cache.test_connection().await.unwrap();
    assert_eq!(result.status, CacheConnectionStatus::Success);
    assert!(cache.ping().await.unwrap());
    let info = cache.info().unwrap();
    assert!(info.matches("redis_version").count() > 1, "{info}");
    assert!(cache.client_list().unwrap().contains("id="));
    cache.async_flush_cache().await.unwrap();
    assert_eq!(cache.async_get_ttl("members").await.unwrap(), None);
    assert_eq!(cache.get_cache("lock", &context).unwrap(), None);
}
