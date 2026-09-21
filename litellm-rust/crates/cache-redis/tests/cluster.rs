//! Contract tests against a real Redis Cluster. Set `LITELLM_TEST_REDIS_CLUSTER_NODES` to a
//! comma separated `host:port` list (for example `127.0.0.1:7000,127.0.0.1:7001`) to run them.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheConnectionStatus, CacheScript, ClaimCache,
    CounterCache, DeleteCache, Error, ExactCacheContext, FlushCache, IncrementOperation, JsonCodec,
    ScriptCache,
};
use litellm_cache_redis::{
    RedisArg, RedisCache, RedisLpopOperation, RedisLpopResult, RedisNode, RedisRpushOperation,
    RedisTopology,
};
use redis::cluster_routing::Slot;

type Cache = RedisCache<JsonCodec<serde_json::Value>>;

fn topology() -> Option<RedisTopology> {
    let nodes = std::env::var("LITELLM_TEST_REDIS_CLUSTER_NODES").ok()?;
    let startup_nodes = nodes
        .split(',')
        .map(|node| {
            let (host, port) = node.trim().rsplit_once(':').expect("host:port");
            RedisNode {
                host: host.to_string(),
                port: port.parse().expect("port"),
            }
        })
        .collect();
    Some(RedisTopology::Cluster { startup_nodes })
}

fn namespace(label: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("cluster-test:{label}:{nanos}")
}

fn cluster_url() -> String {
    std::env::var("LITELLM_TEST_REDIS_CLUSTER_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:7000".into())
}

fn cluster_cache(label: &str) -> Option<Cache> {
    let topology = topology()?;
    Some(
        Cache::connect(
            &cluster_url(),
            &topology,
            Some(Duration::from_secs(120)),
            JsonCodec::new(),
        )
        .expect("cluster connection")
        .with_namespace(Some(namespace(label))),
    )
}

fn counter_cache(label: &str) -> Option<RedisCache<JsonCodec<f64>>> {
    let topology = topology()?;
    Some(
        RedisCache::connect(
            &cluster_url(),
            &topology,
            Some(Duration::from_secs(60)),
            JsonCodec::new(),
        )
        .expect("cluster connection")
        .with_namespace(Some(namespace(label))),
    )
}

fn multi_slot_keys(count: usize) -> Vec<String> {
    let keys: Vec<String> = (0..count).map(|index| format!("key-{index}")).collect();
    let slots: std::collections::HashSet<Slot> = keys.iter().map(Slot::for_key).collect();
    assert!(slots.len() > 1, "keys must span multiple slots");
    keys
}

macro_rules! cluster_or_skip {
    ($label:expr) => {
        match cluster_cache($label) {
            Some(cache) => cache,
            None => return,
        }
    };
}

#[test]
fn constructor_rejects_clusters_without_startup_nodes() {
    let error = Cache::connect(
        "redis://127.0.0.1:7000",
        &RedisTopology::Cluster {
            startup_nodes: Vec::new(),
        },
        None,
        JsonCodec::new(),
    )
    .err();
    assert!(matches!(error, Some(Error::Unavailable)));
}

#[test]
fn constructor_rejects_unix_socket_urls_for_clusters() {
    let error = Cache::connect(
        "redis+unix:///tmp/redis.sock",
        &RedisTopology::Cluster {
            startup_nodes: vec![RedisNode {
                host: "127.0.0.1".into(),
                port: 7000,
            }],
        },
        None,
        JsonCodec::new(),
    )
    .err();
    assert!(matches!(error, Some(Error::Unavailable)));
}

#[test]
fn single_key_operations_round_trip_with_ttl_rounding() {
    let cache = cluster_or_skip!("single");
    let context = ExactCacheContext {
        ttl: Some(Duration::from_millis(1500)),
    };
    let keys = multi_slot_keys(12);
    for (index, key) in keys.iter().enumerate() {
        cache
            .set_cache(key, serde_json::json!({ "index": index }), &context)
            .unwrap();
    }
    for (index, key) in keys.iter().enumerate() {
        assert_eq!(
            cache.get_cache(key, &context).unwrap(),
            Some(serde_json::json!({ "index": index }))
        );
    }
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let ttl = runtime.block_on(cache.async_get_ttl(&keys[0])).unwrap();
    assert_eq!(ttl, Some(2));
    cache.delete_cache(&keys[0]).unwrap();
    assert_eq!(cache.get_cache(&keys[0], &context).unwrap(), None);
    assert!(cache.sync_ping().unwrap());
}

#[tokio::test]
async fn batch_reads_span_slots_and_preserve_order_with_malformed_entries() {
    let cache = cluster_or_skip!("batch");
    let context = ExactCacheContext::default();
    let keys = multi_slot_keys(40);
    for (index, key) in keys.iter().enumerate() {
        if index % 5 == 0 {
            continue;
        }
        cache
            .async_set_cache(key, serde_json::json!(index), context.clone())
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
            BatchEntry::Hit(serde_json::json!(index))
        };
        assert_eq!(*entry, expected, "entry {index}");
    }
    let sync_entries = cache.batch_get_cache(&keys, &context).unwrap();
    assert_eq!(sync_entries, entries);

    cache.delete_cache_keys(keys.clone()).await.unwrap();
    let entries = cache.async_batch_get_cache(keys, context).await.unwrap();
    assert!(entries.iter().all(|entry| *entry == BatchEntry::Miss));
}

#[tokio::test]
async fn pipelines_group_by_slot_and_return_results_in_submission_order() {
    let cache = cluster_or_skip!("pipeline");
    let keys = multi_slot_keys(30);
    let entries = keys
        .iter()
        .enumerate()
        .map(|(index, key)| (key.clone(), serde_json::json!(index)))
        .collect();
    cache
        .async_set_cache_pipeline(entries, ExactCacheContext::default())
        .await
        .unwrap();
    let hits = cache
        .async_batch_get_cache(keys.clone(), ExactCacheContext::default())
        .await
        .unwrap();
    assert!(
        hits.iter()
            .enumerate()
            .all(|(index, entry)| *entry == BatchEntry::Hit(serde_json::json!(index)))
    );

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
    let Some(counter) = counter_cache("counter") else {
        return;
    };
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
    assert_eq!(counter.async_get_ttl(&counters[0]).await.unwrap(), Some(30));
    assert_eq!(counter.async_get_ttl(&counters[1]).await.unwrap(), None);
    counter.async_flush_cache().await.unwrap();
    cache.async_flush_cache().await.unwrap();
}

#[tokio::test]
async fn scan_and_scoped_flush_cover_every_primary() {
    let cache = cluster_or_skip!("flush");
    let other = cluster_or_skip!("other");
    let context = ExactCacheContext::default();
    let keys = multi_slot_keys(60);
    for key in &keys {
        cache
            .async_set_cache(key, serde_json::json!(true), context.clone())
            .await
            .unwrap();
        other
            .async_set_cache(key, serde_json::json!(true), context.clone())
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
            .all(|entry| *entry == BatchEntry::Hit(serde_json::json!(true)))
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

#[tokio::test]
async fn ping_reaches_every_node() {
    let cache = cluster_or_skip!("ping");
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
}

#[tokio::test]
async fn counters_claims_scripts_and_sets_work_on_the_cluster() {
    let Some(counter) = counter_cache("counter") else {
        return;
    };
    let context = ExactCacheContext::default();
    assert_eq!(
        counter
            .increment_cache("spend", 1.5, context.clone())
            .unwrap(),
        1.5
    );
    assert_eq!(
        counter
            .async_increment("spend", 2.0, context.clone())
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

    let cache = cluster_or_skip!("claim");
    let owner = serde_json::json!("owner-a");
    let rival = serde_json::json!("owner-b");
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
    assert_eq!(cache.async_get_ttl("scripted").await.unwrap(), Some(5));
    let evaluated: redis::Value = cache
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
                Some(Duration::from_secs(9)),
            )
            .await
            .unwrap(),
        2
    );
    assert_eq!(cache.async_get_ttl("members").await.unwrap(), Some(9));

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
