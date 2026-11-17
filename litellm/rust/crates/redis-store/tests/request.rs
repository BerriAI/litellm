mod support;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_redis_client::commands;
use litellm_redis_client::{Amount, CallTag, Expiry, Operation, tagged};
use litellm_redis_store::RequestRedis;
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, connect_observed, connect_tagged};
use tokio::{runtime::Handle, sync::oneshot};

const TTL: Option<Expiry> = Some(Expiry::KeepExisting(Duration::from_secs(60)));

// A batch is one pipeline on a standalone Redis and one call per increment on a cluster, like RedisBatch
fn increments(topology: Topology, count: usize) -> Vec<Operation> {
    match topology {
        Topology::Standalone => vec![Operation::Pipeline { ops: count }],
        Topology::Cluster => vec![Operation::Command("INCRBYFLOAT"); count],
    }
}
const LONG_DEADLINE: Duration = Duration::from_secs(60);

#[rstest]
#[tokio::test]
async fn pre_declarations_on_one_client_share_one_pipeline_and_another_client_gets_its_own(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = connect_observed(&deployment);
    let (other, other_trips) = connect_observed(&deployment);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);

    let first = request
        .pre(&client)
        .add(commands::incr_by_float("a", Amount::Int(1), TTL));
    let second =
        request
            .pre(&client.clone())
            .add(commands::incr_by_float("b", Amount::Int(2), TTL));
    let third = request
        .pre(&other)
        .add(commands::incr_by_float("c", Amount::Int(3), TTL));
    request.flush_pre().await;

    assert_eq!(trips.seen(), increments(topology, 2));
    assert_eq!(other_trips.seen(), increments(topology, 1));
    assert_eq!(
        [
            first.resolve().await.unwrap(),
            second.resolve().await.unwrap(),
            third.resolve().await.unwrap(),
        ],
        [1.0, 2.0, 3.0]
    );
    assert_eq!(trips.seen(), increments(topology, 2));
}

#[rstest]
#[tokio::test]
async fn a_post_declaration_waits_for_flush_post_and_flush_pre_leaves_it_alone(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = connect_observed(&deployment);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);

    let total = tokio::spawn(
        request
            .post(&client)
            .add(commands::incr_by_float("spend", Amount::Float(2.5), TTL))
            .settled(),
    );
    request.flush_pre().await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(!total.is_finished());
    assert!(trips.seen().is_empty());
    assert_eq!(
        raw.send(commands::get::<String>("spend")).await.unwrap(),
        None
    );

    request.flush_post().await;

    assert_eq!(total.await.unwrap().unwrap(), 2.5);
    assert_eq!(trips.seen(), increments(topology, 1));
}

#[rstest]
#[tokio::test]
async fn flush_post_returns_after_the_follow_ups_of_the_ops_it_sent(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&deployment);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let total =
        request
            .post(&client)
            .add(commands::incr_by_float("spend", Amount::Float(2.5), TTL));
    let (mirror, mut mirrored) = oneshot::channel();
    request.spawn(async move {
        mirror.send(total.settled().await.unwrap()).unwrap();
    });

    request.flush_post().await;

    assert_eq!(mirrored.try_recv(), Ok(2.5));
}

#[rstest]
#[tokio::test]
async fn a_post_batch_nobody_flushes_goes_out_at_the_deadline(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = connect_observed(&deployment);
    let request = RequestRedis::new(Handle::current(), Duration::from_millis(100));

    let first = request
        .post(&client)
        .add(commands::incr_by_float("a", Amount::Int(1), TTL));
    let second = request
        .post(&client)
        .add(commands::incr_by_float("b", Amount::Int(2), TTL));
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        (first.settled().await, second.settled().await)
    })
    .await
    .unwrap();

    assert_eq!((settled.0.unwrap(), settled.1.unwrap()), (1.0, 2.0));
    assert_eq!(trips.seen(), increments(topology, 2));
}

#[rstest]
#[tokio::test]
async fn a_post_declaration_from_a_thread_without_a_runtime_goes_out_at_the_deadline(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = connect_observed(&deployment);
    let request = RequestRedis::new(Handle::current(), Duration::from_millis(100));

    let declared = std::thread::spawn({
        let request = request.clone();
        move || {
            request
                .post(&client)
                .add(commands::incr_by_float("spend", Amount::Float(2.5), TTL))
        }
    })
    .join()
    .expect("a Python thread has no Tokio context, and declaring there must not panic");
    let total = tokio::time::timeout(Duration::from_secs(5), declared.settled())
        .await
        .unwrap();

    assert_eq!(total.unwrap(), 2.5);
    assert_eq!(trips.seen(), increments(topology, 1));
}

#[rstest]
#[tokio::test]
async fn a_post_declaration_after_a_flush_arms_a_new_deadline(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = connect_observed(&deployment);
    let request = RequestRedis::new(Handle::current(), Duration::from_millis(300));

    let early = request
        .post(&client)
        .add(commands::incr_by_float("a", Amount::Int(1), TTL));
    request.flush_post().await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let late = request
        .post(&client)
        .add(commands::incr_by_float("a", Amount::Int(1), TTL));
    tokio::time::sleep(Duration::from_millis(200)).await;
    let flushed_before_its_deadline = trips.seen().len() > 1;
    let late = tokio::time::timeout(Duration::from_secs(5), late.settled())
        .await
        .unwrap();

    assert_eq!(early.settled().await.unwrap(), 1.0);
    assert!(!flushed_before_its_deadline);
    assert_eq!(late.unwrap(), 2.0);
}

#[rstest]
#[tokio::test]
async fn the_deadline_flush_runs_under_the_tag_given_when_its_first_op_is_declared(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, tags) = connect_tagged(&deployment);
    let caller = Arc::new(Mutex::new("the request before it has a span"));
    let asked = caller.clone();
    let request = RequestRedis::with_deadline_tag(
        Handle::current(),
        Duration::from_millis(100),
        Arc::new(move || Some(Arc::new(*asked.lock().unwrap()) as CallTag)),
    );
    *caller.lock().unwrap() = "post_call_redis_batch";

    let declared = tagged(Arc::new("declaring op"), async {
        request
            .post(&client)
            .add(commands::incr_by_float("a", Amount::Int(1), TTL))
    })
    .await;
    tokio::time::timeout(Duration::from_secs(5), declared.settled())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(tags.seen(), vec![Some("post_call_redis_batch")]);
}

#[rstest]
#[tokio::test]
async fn a_follow_up_runs_under_the_tag_it_was_spawned_in(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, tags) = connect_tagged(&deployment);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);

    tagged(Arc::new("async_increment"), async {
        request.spawn(async move {
            client
                .send(commands::incr_by_float("a", Amount::Int(1), TTL))
                .await
                .unwrap();
        });
    })
    .await;
    request.flush_post().await;

    assert_eq!(tags.seen(), vec![Some("async_increment")]);
}
