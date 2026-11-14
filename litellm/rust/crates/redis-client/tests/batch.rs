mod support;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_circuit_breaker::{State, Transition};
use litellm_redis_client::commands;
use litellm_redis_client::{
    Amount, CallEvent, Credentials, DEFAULT_TIMEOUT, Declared, Error, Expiry, Namespace, Operation,
    RedisBatch, RedisClient, RedisConfig, RedisEvents, Script, Timeouts, Ttl,
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, endpoint};

const MINUTE: Option<Expiry> = Some(Expiry::KeepExisting(Duration::from_secs(60)));

// RedisBatch pipelines a standalone flush and runs each op on its own on a cluster
fn expected_trips(topology: Topology, alone: &[Operation]) -> Vec<Operation> {
    match topology {
        Topology::Standalone => vec![Operation::Pipeline { ops: alone.len() }],
        Topology::Cluster => alone.to_vec(),
    }
}

#[derive(Default)]
struct RoundTrips(Mutex<Vec<Operation>>);

impl RoundTrips {
    fn seen(&self) -> Vec<Operation> {
        self.0.lock().unwrap().clone()
    }
}

impl RedisEvents for RoundTrips {
    fn call(&self, event: &CallEvent) {
        self.0.lock().unwrap().push(event.operation);
    }

    fn breaker(&self, _transition: &Transition<State>) {}
}

fn observed(server: &RedisDeployment) -> (RedisClient, Arc<RoundTrips>) {
    let trips = Arc::new(RoundTrips::default());
    let client = RedisClient::new(RedisConfig {
        target: endpoint(server),
        credentials: Credentials::NONE,
        namespace: Namespace::new("ns"),
        timeouts: Timeouts::default(),
        breaker: None,
        events: Some(trips.clone()),
        health_check: None,
        keepalive: true,
    })
    .unwrap();
    (client, trips)
}

#[rstest]
#[tokio::test]
async fn declared_ops_wait_for_the_flush_and_share_one_round_trip(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = observed(&server);
    let batch = client.batch();
    let counter = batch.add(commands::incr_by_float("counter", Amount::Int(3), MINUTE));
    let total = batch.add(commands::incr_by_float("total", Amount::Float(1.5), MINUTE));
    let written = batch.add(commands::set("other", "v", Duration::from_secs(60)));
    let read = batch.add(commands::get::<String>("value"));

    let outside = connect(&server, "ns", DEFAULT_TIMEOUT);
    assert_eq!(
        outside.send(commands::get::<i64>("counter")).await.unwrap(),
        None
    );
    assert!(trips.seen().is_empty());

    batch.flush().await;

    let expected = expected_trips(
        topology,
        &[
            Operation::Command("INCRBYFLOAT"),
            Operation::Command("INCRBYFLOAT"),
            Operation::Command("SET"),
            Operation::Command("GET"),
        ],
    );
    let mut seen = trips.seen();
    seen.sort_by_key(|operation| format!("{operation:?}"));
    let mut sorted = expected.clone();
    sorted.sort_by_key(|operation| format!("{operation:?}"));
    assert_eq!(seen, sorted);
    assert_eq!(counter.resolve().await.unwrap(), 3.0);
    assert_eq!(total.resolve().await.unwrap(), 1.5);
    written.resolve().await.unwrap();
    assert_eq!(read.resolve().await.unwrap(), None);
    assert_eq!(trips.seen().len(), expected.len());
}

#[rstest]
#[tokio::test]
async fn resolving_before_a_flush_sends_everything_declared_so_far(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = observed(&server);
    let batch = client.batch();
    let first = batch.add(commands::incr_by_float("a", Amount::Int(1), MINUTE));
    let second = batch.add(commands::incr_by_float("b", Amount::Int(2), MINUTE));
    let increment = Operation::Command("INCRBYFLOAT");

    assert_eq!(second.resolve().await.unwrap(), 2.0);
    let late = batch.add(commands::incr_by_float("a", Amount::Int(10), MINUTE));
    assert_eq!(first.resolve().await.unwrap(), 1.0);
    assert_eq!(
        trips.seen(),
        expected_trips(topology, &[increment, increment])
    );

    assert_eq!(late.resolve().await.unwrap(), 11.0);
    let both: Vec<Operation> = expected_trips(topology, &[increment, increment])
        .into_iter()
        .chain(expected_trips(topology, &[increment]))
        .collect();
    assert_eq!(trips.seen(), both);
    batch.flush().await;
    assert_eq!(trips.seen().len(), both.len());
}

#[rstest]
#[tokio::test]
async fn a_pipelined_batch_is_one_pipeline_on_every_topology(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = observed(&server);
    let batch = RedisBatch::pipelined(client);
    let declared =
        ["a", "b", "c"].map(|key| batch.add(commands::incr_by_float(key, Amount::Int(2), MINUTE)));

    let totals = Declared::all(declared).await;

    assert_eq!(
        totals.into_iter().map(Result::unwrap).collect::<Vec<_>>(),
        [2.0; 3]
    );
    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 3 }]);
}

#[rstest]
#[tokio::test]
async fn each_declared_op_gets_its_own_typed_result(
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
    let batch = client.batch();
    let rejected = batch.add(commands::incr_by_float("text", Amount::Int(1), MINUTE));
    let undecodable = batch.add(commands::get::<i64>("text"));
    let missing = batch.add(commands::get::<String>("missing"));
    let script = batch.add(commands::eval::<Vec<String>>(
        &Script::new("return {KEYS[1], ARGV[1]}"),
        &["k"],
        "arg",
    ));
    let removed = batch.add(commands::del(&["text"]));

    batch.flush().await;

    assert!(matches!(rejected.resolve().await, Err(Error::Server(_))));
    assert!(matches!(undecodable.resolve().await, Err(Error::Decode(_))));
    assert_eq!(missing.resolve().await.unwrap(), None);
    assert_eq!(script.resolve().await.unwrap(), ["k", "arg"]);
    assert_eq!(removed.resolve().await.unwrap(), 1);
}

#[rstest]
#[tokio::test]
async fn an_op_whose_flush_was_cancelled_resolves_as_transport(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", Duration::from_secs(5));
    let batch = client.batch();
    let counter = batch.add(commands::incr_by_float("counter", Amount::Int(1), MINUTE));

    server.freeze().await.unwrap();
    let cancelled = tokio::time::timeout(Duration::from_millis(200), batch.flush()).await;
    server.thaw().await.unwrap();

    assert!(cancelled.is_err());
    assert!(matches!(counter.resolve().await, Err(Error::Transport(_))));
}

#[rstest]
#[tokio::test]
async fn a_settled_op_waits_for_someone_else_to_flush_and_sends_nothing_itself(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = observed(&server);
    let batch = client.batch();
    let counter = tokio::spawn(
        batch
            .add(commands::incr_by_float("counter", Amount::Int(4), MINUTE))
            .settled(),
    );

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!counter.is_finished());
    assert!(trips.seen().is_empty());

    batch.flush().await;

    assert_eq!(counter.await.unwrap().unwrap(), 4.0);
    assert_eq!(
        trips.seen(),
        expected_trips(topology, &[Operation::Command("INCRBYFLOAT")])
    );
}

#[rstest]
#[tokio::test]
async fn a_settled_op_whose_flush_was_cancelled_resolves_as_transport() {
    let server = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let client = connect(&server, "", Duration::from_secs(5));
    let batch = client.batch();
    let counter = batch.add(commands::incr_by_float("counter", Amount::Int(1), MINUTE));

    server.freeze().await.unwrap();
    let cancelled = tokio::time::timeout(Duration::from_millis(200), batch.flush()).await;
    server.thaw().await.unwrap();

    assert!(cancelled.is_err());
    assert!(matches!(counter.settled().await, Err(Error::Transport(_))));
}

#[rstest]
#[tokio::test]
async fn clones_are_the_same_client_and_a_second_connection_is_not() {
    let server = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    let twin = connect(&server, "", DEFAULT_TIMEOUT);

    assert!(client.same_client(&client.clone()));
    assert!(!client.same_client(&twin));
}

#[rstest]
#[tokio::test]
async fn mget_reads_each_key_once_and_answers_every_key_in_order(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .send(commands::set("a", "1", Duration::from_secs(60)))
        .await
        .unwrap();
    client
        .send(commands::set("c", "3", Duration::from_secs(60)))
        .await
        .unwrap();
    server
        .cli_on_every_node(&["CONFIG", "RESETSTAT"])
        .await
        .unwrap();
    let batch = client.batch();

    let values = batch.add(commands::mget::<String>(&["c", "missing", "a", "c"]));
    batch.flush().await;

    assert_eq!(
        values.resolve().await.unwrap(),
        [
            Some("3".to_owned()),
            None,
            Some("1".to_owned()),
            Some("3".to_owned())
        ]
    );
    let stats = support::command_stats(&server).await;
    assert!(support::calls(&stats, "mget") > 0, "{stats}");
    assert_eq!(support::calls(&stats, "get"), 0, "{stats}");
    assert_eq!(support::keys_read(&server).await, 3);
}

// RedisBatch's _Increment refreshes the TTL in a pipeline, while its cluster path calls
// async_increment, which keeps a TTL the key already has
#[rstest]
#[case::pipelined(Topology::Standalone, false)]
#[case::alone_on_a_cluster(Topology::Cluster, true)]
#[tokio::test]
async fn a_batched_increment_treats_the_ttl_like_redis_batch(
    #[case] topology: Topology,
    #[case] keeps_the_ttl: bool,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .send(commands::set("counter", "1", Duration::from_secs(30)))
        .await
        .unwrap();
    let batch = client.batch();

    let counter = batch.add(commands::incr_by_float(
        "counter",
        Amount::Int(1),
        Some(Expiry::KeepExisting(Duration::from_secs(600))),
    ));
    batch.flush().await;

    assert_eq!(counter.resolve().await.unwrap(), 2.0);
    let Ttl::Expires(left) = client.send(commands::ttl("counter")).await.unwrap() else {
        panic!("expected an expiring counter")
    };
    assert_eq!(left <= Duration::from_secs(30), keeps_the_ttl, "{left:?}");
}
