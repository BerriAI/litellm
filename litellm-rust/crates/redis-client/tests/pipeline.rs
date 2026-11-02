mod support;

use std::time::{Duration, Instant};

use litellm_redis_client::Value;
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;

use litellm_redis_client::{DEFAULT_TIMEOUT, Error, Script};
use support::{KEEP, connect};

async fn begin_migration(cluster: &litellm_redis_testing::RedisCluster, key: &str) {
    let (_, first) = cluster.nodes().next().unwrap();
    let slot = cluster
        .cli_on_node(first, &["CLUSTER", "KEYSLOT", key])
        .await
        .unwrap();
    let slot = slot.trim();
    let number: u16 = slot.parse().unwrap();
    let nodes = cluster
        .cli_on_node(first, &["CLUSTER", "NODES"])
        .await
        .unwrap();
    let owner = nodes.lines().find(|line| line.split_whitespace().skip(8).any(|range| {
        let (start, end) = range.split_once('-').unwrap_or((range, range));
        matches!((start.parse::<u16>(), end.parse::<u16>()), (Ok(start), Ok(end)) if (start..=end).contains(&number))
    })).unwrap();
    let target = nodes.lines().find(|line| *line != owner).unwrap();
    let address = |line: &str| {
        let value = line.split_whitespace().nth(1).unwrap();
        value
            .split_once('@')
            .unwrap()
            .0
            .rsplit_once(':')
            .unwrap()
            .1
            .parse::<u16>()
            .unwrap()
    };
    let source_port = address(owner);
    let target_port = address(target);
    let source_id = owner.split_whitespace().next().unwrap();
    let target_id = target.split_whitespace().next().unwrap();
    assert_eq!(
        cluster
            .cli_on_node(source_port, &["GET", key])
            .await
            .unwrap()
            .trim(),
        "0"
    );
    assert_eq!(
        cluster
            .cli_on_node(
                target_port,
                &["CLUSTER", "SETSLOT", slot, "IMPORTING", source_id]
            )
            .await
            .unwrap()
            .trim(),
        "OK"
    );
    assert_eq!(
        cluster
            .cli_on_node(
                source_port,
                &["CLUSTER", "SETSLOT", slot, "MIGRATING", target_id]
            )
            .await
            .unwrap()
            .trim(),
        "OK"
    );
    let state = cluster
        .cli_on_node(source_port, &["CLUSTER", "NODES"])
        .await
        .unwrap();
    assert!(
        state.contains(&format!("[{slot}->-{target_id}]")),
        "{state}"
    );
}

#[rstest]
#[tokio::test]
async fn a_retrying_neighbor_cannot_reapply_a_cluster_counter() {
    let cluster = litellm_redis_testing::RedisCluster::start().await.unwrap();
    let server = RedisDeployment::Cluster(cluster);
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    let counter = "{migration}:counter";
    client
        .set(counter, 0, Duration::from_secs(60))
        .await
        .unwrap();
    let RedisDeployment::Cluster(cluster) = &server else {
        unreachable!()
    };
    begin_migration(cluster, counter).await;
    let both_keys = Script::new("return redis.call('MGET', KEYS[1], KEYS[2])");

    let results = client
        .run(&client.pipeline().incr_by(counter, 1, KEEP).eval(
            &both_keys,
            &[counter, "{migration}:missing"],
            Vec::<String>::new(),
        ))
        .await;

    assert_eq!(client.get::<i64>(counter).await.unwrap(), Some(1));
    assert_eq!(results[0].as_ref().unwrap(), &Value::Int(1));
    assert!(
        matches!(&results[1], Err(Error::Server(error)) if error.code() == Some("TRYAGAIN")),
        "{:?}",
        results[1]
    );
}

#[rstest]
#[tokio::test]
async fn each_op_settles_on_its_own_in_input_order(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .set("text", "not a number", Duration::from_secs(60))
        .await
        .unwrap();
    let pipeline = client
        .pipeline()
        .incr_by("a", 1, KEEP)
        .incr_by("text", 1, KEEP)
        .incr_by("c", 5, KEEP)
        .set("b", "2", Duration::from_secs(60))
        .get("b")
        .incr_by("a", 1, KEEP)
        .get("missing");

    let results = client.run(&pipeline).await;

    assert_eq!(results.len(), 7);
    assert_eq!(results[0].as_ref().unwrap(), &Value::Int(1));
    assert!(
        matches!(results[1], Err(Error::Server(_))),
        "{:?}",
        results[1]
    );
    assert_eq!(results[2].as_ref().unwrap(), &Value::Int(5));
    assert_eq!(results[3].as_ref().unwrap(), &Value::Okay);
    assert_eq!(
        results[4].as_ref().unwrap(),
        &Value::BulkString(b"2".to_vec())
    );
    assert_eq!(results[5].as_ref().unwrap(), &Value::Int(2));
    assert_eq!(results[6].as_ref().unwrap(), &Value::Nil);
}

#[rstest]
#[tokio::test]
async fn a_script_runs_in_pipeline_order_on_first_use_and_after_a_flush(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(false, true)] flushed: bool,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "ns", DEFAULT_TIMEOUT);
    let incr = Script::new("return redis.call('INCRBY', KEYS[1], ARGV[1])");
    if flushed {
        client.eval::<i64>(&incr, &["warmup"], 1).await.unwrap();
        server
            .cli_on_every_node(&["SCRIPT", "FLUSH"])
            .await
            .unwrap();
    }
    let pipeline = client
        .pipeline()
        .eval(&incr, &["counter"], 2)
        .get("counter")
        .incr_by("counter", 5, KEEP)
        .get("counter");

    let results = client.run(&pipeline).await;

    assert_eq!(results[0].as_ref().unwrap(), &Value::Int(2));
    assert_eq!(
        results[1].as_ref().unwrap(),
        &Value::BulkString(b"2".to_vec())
    );
    assert_eq!(results[2].as_ref().unwrap(), &Value::Int(7));
    assert_eq!(
        results[3].as_ref().unwrap(),
        &Value::BulkString(b"7".to_vec())
    );
    assert_eq!(client.get::<i64>("counter").await.unwrap(), Some(7));
}

#[rstest]
#[tokio::test]
async fn an_unreachable_server_fails_every_op_as_transport(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", Duration::from_millis(500));
    let pipeline = client
        .pipeline()
        .incr_by("a", 1, KEEP)
        .incr_by("c", 1, KEEP);

    server.freeze().await.unwrap();
    let results = client.run(&pipeline).await;
    server.thaw().await.unwrap();

    assert_eq!(results.len(), 2);
    assert!(
        matches!(results[0], Err(Error::Transport(_))),
        "{:?}",
        results[0]
    );
    assert!(
        matches!(results[1], Err(Error::Transport(_))),
        "{:?}",
        results[1]
    );
}

#[rstest]
#[tokio::test]
async fn an_empty_pipeline_returns_without_asking_redis(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", Duration::from_secs(2));

    server.freeze().await.unwrap();
    let started = Instant::now();
    let results = client.run(&client.pipeline()).await;
    let elapsed = started.elapsed();
    server.thaw().await.unwrap();

    assert!(results.is_empty());
    assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
}
