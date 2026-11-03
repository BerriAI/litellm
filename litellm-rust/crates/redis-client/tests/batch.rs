mod support;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_circuit_breaker::{State, Transition};
use litellm_redis_client::{
    CallEvent, Credentials, DEFAULT_TIMEOUT, Error, Namespace, Operation, RedisClient, RedisConfig,
    RedisEvents, Script, Timeouts,
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{KEEP, connect, endpoint};

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
        endpoint: endpoint(server),
        credentials: Credentials::NONE,
        namespace: Namespace::new("ns"),
        timeouts: Timeouts::default(),
        breaker: None,
        events: Some(trips.clone()),
        health_check: None,
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
    let counter = batch.incr_by("counter", 3, KEEP);
    let total = batch.incr_by_float("total", 1.5, KEEP);
    let written = batch.set("value", "v", Duration::from_secs(60));
    let read = batch.get::<String>("value");

    let outside = connect(&server, "ns", DEFAULT_TIMEOUT);
    assert_eq!(outside.get::<i64>("counter").await.unwrap(), None);
    assert!(trips.seen().is_empty());

    batch.flush().await;

    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 4 }]);
    assert_eq!(counter.resolve().await.unwrap(), 3);
    assert_eq!(total.resolve().await.unwrap(), 1.5);
    written.resolve().await.unwrap();
    assert_eq!(read.resolve().await.unwrap().as_deref(), Some("v"));
    assert_eq!(trips.seen().len(), 1);
}

#[rstest]
#[tokio::test]
async fn resolving_before_a_flush_sends_everything_declared_so_far(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = observed(&server);
    let batch = client.batch();
    let first = batch.incr_by("a", 1, KEEP);
    let second = batch.incr_by("b", 2, KEEP);

    assert_eq!(second.resolve().await.unwrap(), 2);
    let late = batch.incr_by("a", 10, KEEP);
    assert_eq!(first.resolve().await.unwrap(), 1);
    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 2 }]);

    assert_eq!(late.resolve().await.unwrap(), 11);
    assert_eq!(
        trips.seen(),
        [
            Operation::Pipeline { ops: 2 },
            Operation::Pipeline { ops: 1 }
        ]
    );
    batch.flush().await;
    assert_eq!(trips.seen().len(), 2);
}

#[rstest]
#[tokio::test]
async fn each_declared_op_gets_its_own_typed_result(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .set("text", "not a number", Duration::from_secs(60))
        .await
        .unwrap();
    let batch = client.batch();
    let rejected = batch.incr_by("text", 1, KEEP);
    let undecodable = batch.get::<i64>("text");
    let missing = batch.get::<String>("missing");
    let script =
        batch.eval::<Vec<String>>(&Script::new("return {KEYS[1], ARGV[1]}"), &["k"], "arg");
    let removed = batch.del("text");

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
    let counter = batch.incr_by("counter", 1, KEEP);

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
    let counter = tokio::spawn(batch.incr_by("counter", 4, KEEP).settled());

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!counter.is_finished());
    assert!(trips.seen().is_empty());

    batch.flush().await;

    assert_eq!(counter.await.unwrap().unwrap(), 4);
    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 1 }]);
}

#[rstest]
#[tokio::test]
async fn a_settled_op_whose_flush_was_cancelled_resolves_as_transport() {
    let server = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let client = connect(&server, "", Duration::from_secs(5));
    let batch = client.batch();
    let counter = batch.incr_by("counter", 1, KEEP);

    server.freeze().await.unwrap();
    let cancelled = tokio::time::timeout(Duration::from_millis(200), batch.flush()).await;
    server.thaw().await.unwrap();

    assert!(cancelled.is_err());
    assert!(matches!(counter.settled().await, Err(Error::Transport(_))));
}
