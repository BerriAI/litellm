mod support;

use std::time::{Duration, Instant};

use litellm_redis_client::Value;
use litellm_redis_client::commands;
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;

use litellm_redis_client::{Amount, DEFAULT_TIMEOUT, Error, Expiry, Script};
use support::connect;

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
        .send(commands::set(counter, 0, Duration::from_secs(60)))
        .await
        .unwrap();
    let RedisDeployment::Cluster(cluster) = &server else {
        unreachable!()
    };
    begin_migration(cluster, counter).await;
    let both_keys = Script::new("return redis.call('MGET', KEYS[1], KEYS[2])");

    let results = client
        .run(
            &client
                .pipeline()
                .with(commands::incr_by_float(
                    counter,
                    Amount::Int(1),
                    Some(Expiry::Refresh(Duration::from_secs(60))),
                ))
                .with(commands::eval::<Value>(
                    &both_keys,
                    &[counter, "{migration}:missing"],
                    Vec::<String>::new(),
                )),
        )
        .await;

    assert_eq!(
        client.send(commands::get::<i64>(counter)).await.unwrap(),
        Some(1)
    );
    assert_eq!(
        results[0].as_ref().unwrap(),
        &Value::BulkString(b"1".to_vec())
    );
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
        .send(commands::set(
            "text",
            "not a number",
            Duration::from_secs(60),
        ))
        .await
        .unwrap();
    let pipeline = client
        .pipeline()
        .with(commands::incr_by_float(
            "a",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ))
        .with(commands::incr_by_float(
            "text",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ))
        .with(commands::incr_by_float(
            "c",
            Amount::Int(5),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ))
        .with(commands::set("b", "2", Duration::from_secs(60)))
        .with(commands::get::<Value>("b"))
        .with(commands::incr_by_float(
            "a",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ))
        .with(commands::get::<Value>("missing"));

    let results = client.run(&pipeline).await;

    assert_eq!(results.len(), 7);
    assert_eq!(
        results[0].as_ref().unwrap(),
        &Value::BulkString(b"1".to_vec())
    );
    assert!(
        matches!(results[1], Err(Error::Server(_))),
        "{:?}",
        results[1]
    );
    assert_eq!(
        results[2].as_ref().unwrap(),
        &Value::BulkString(b"5".to_vec())
    );
    assert_eq!(results[3].as_ref().unwrap(), &Value::Okay);
    assert_eq!(
        results[4].as_ref().unwrap(),
        &Value::BulkString(b"2".to_vec())
    );
    assert_eq!(
        results[5].as_ref().unwrap(),
        &Value::BulkString(b"2".to_vec())
    );
    assert_eq!(results[6].as_ref().unwrap(), &Value::Nil);
}

#[rstest]
#[case::cached(true, Value::BulkString(b"2".to_vec()))]
#[case::lost_by_the_server(false, Value::Nil)]
#[tokio::test]
async fn a_lost_script_reruns_after_the_rest_of_its_pipeline_like_redis_py(
    #[case] cached: bool,
    #[case] read_in_the_same_pipeline: Value,
) {
    let server = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let client = connect(&server, "ns", DEFAULT_TIMEOUT);
    let incr = Script::new("return redis.call('INCRBY', KEYS[1], ARGV[1])");
    client
        .send(commands::eval::<i64>(&incr, &["warmup"], 1))
        .await
        .unwrap();
    if !cached {
        server
            .cli_on_every_node(&["SCRIPT", "FLUSH"])
            .await
            .unwrap();
    }
    let pipeline = client
        .pipeline()
        .with(commands::eval::<Value>(&incr, &["counter"], 2))
        .with(commands::get::<Value>("counter"));

    let results = client.run(&pipeline).await;

    assert_eq!(results[0].as_ref().unwrap(), &Value::Int(2));
    assert_eq!(results[1].as_ref().unwrap(), &read_in_the_same_pipeline);
    assert_eq!(
        client.send(commands::get::<i64>("counter")).await.unwrap(),
        Some(2)
    );
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
        .with(commands::incr_by_float(
            "a",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ))
        .with(commands::incr_by_float(
            "c",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ));

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

#[rstest]
#[tokio::test]
async fn a_pipeline_sends_scripts_by_sha_and_reruns_only_the_ones_the_server_lost(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&deployment, "scripts", DEFAULT_TIMEOUT);
    let count = Script::new("return redis.call('INCR', KEYS[1])");
    deployment
        .cli_on_every_node(&["SCRIPT", "FLUSH"])
        .await
        .unwrap();
    deployment
        .cli_on_every_node(&["CONFIG", "RESETSTAT"])
        .await
        .unwrap();

    let first = client
        .run(
            &client
                .pipeline()
                .with(commands::incr_by_float(
                    "{a}spend",
                    Amount::Float(0.5),
                    Some(Expiry::Refresh(Duration::from_secs(60))),
                ))
                .with(commands::eval::<Value>(
                    &count,
                    &["{a}"],
                    Vec::<String>::new(),
                ))
                .with(commands::eval::<Value>(
                    &count,
                    &["b"],
                    Vec::<String>::new(),
                )),
        )
        .await;
    let second = client
        .run(&client.pipeline().with(commands::eval::<Value>(
            &count,
            &["{a}"],
            Vec::<String>::new(),
        )))
        .await;

    let replies: Vec<Value> = first
        .into_iter()
        .chain(second)
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        replies,
        [
            Value::BulkString(b"0.5".to_vec()),
            Value::Int(1),
            Value::Int(1),
            Value::Int(2)
        ]
    );
    assert_eq!(
        client.send(commands::get::<f64>("{a}spend")).await.unwrap(),
        Some(0.5)
    );
    let stats = support::command_stats(&deployment).await;
    assert!(support::calls(&stats, "evalsha") > 0, "{stats}");
    assert_eq!(support::calls(&stats, "eval"), 0, "{stats}");
}
