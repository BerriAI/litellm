mod support;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_redis_client::{Ttl, commands};
use litellm_redis_store::{
    Clock, Number,
    deployment_health::{DeploymentHealth, HealthState, HealthStates},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, remote, within};

const STATES_KEY: &str = "litellm:health_check:deployment_health_state";

fn states() -> HealthStates {
    states_because("timeout")
}

fn states_because(reason: &str) -> HealthStates {
    one("dep-a", false, Number::Float(1.5), reason)
}

fn one(id: &str, is_healthy: bool, timestamp: Number, reason: &str) -> HealthStates {
    HealthStates(vec![(
        id.to_owned(),
        HealthState {
            is_healthy,
            timestamp,
            reason: reason.to_owned(),
        },
    )])
}

#[rstest]
#[tokio::test]
async fn states_one_pod_writes_are_read_by_another_under_mains_key_and_encoding(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let writer = DeploymentHealth::new(Some(remote(connect(&deployment))), None);
    let reader = DeploymentHealth::new(Some(remote(connect(&deployment))), None);

    writer.set(states(), Duration::from_secs(90)).await.unwrap();

    let written: Option<String> = raw.send(commands::get(STATES_KEY)).await.unwrap();
    let Ttl::Expires(remaining) = raw.send(commands::ttl(STATES_KEY)).await.unwrap() else {
        panic!("{STATES_KEY} has no TTL");
    };
    assert_eq!(
        written.as_deref(),
        Some("{'dep-a': {'is_healthy': False, 'timestamp': 1.5, 'reason': 'timeout'}}")
    );
    assert!(within(remaining, 90), "{remaining:?}");
    assert_eq!(reader.get_shared().await, Some(states()));
    assert_eq!(reader.get().await.unwrap(), Some(states()));
}

#[rstest]
#[tokio::test]
async fn without_redis_the_states_stay_in_memory_and_nothing_is_shared() {
    let store = DeploymentHealth::new(None, None);

    store.set(states(), Duration::from_secs(90)).await.unwrap();

    assert_eq!(
        (store.get().await.unwrap(), store.get_shared().await),
        (Some(states()), None)
    );
}

#[rstest]
#[tokio::test]
async fn a_snapshot_read_from_redis_is_read_again_once_the_configured_memory_ttl_passes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let now = Arc::new(Mutex::new(Duration::from_secs(1_730_000_000)));
    let clock: Clock = {
        let now = now.clone();
        Arc::new(move || *now.lock().unwrap())
    };
    let writer = DeploymentHealth::new(Some(remote(connect(&deployment))), None);
    let reader = DeploymentHealth::with_clock(
        Some(remote(connect(&deployment))),
        Some(Duration::from_secs(5)),
        clock,
    );
    writer.set(states(), Duration::from_secs(90)).await.unwrap();
    let first = reader.get().await.unwrap();
    writer
        .set(states_because("rate limited"), Duration::from_secs(90))
        .await
        .unwrap();

    let within_ttl = reader.get().await.unwrap();
    *now.lock().unwrap() += Duration::from_secs(6);
    let after_ttl = reader.get().await.unwrap();

    assert_eq!(
        (first, within_ttl, after_ttl),
        (
            Some(states()),
            Some(states()),
            Some(states_because("rate limited"))
        )
    );
}

#[rstest]
#[case::extra_fields_are_ignored(
    "{'dep-a': {'is_healthy': False, 'timestamp': 1.5, 'reason': 'timeout', 'region': 'eu'}}",
    one("dep-a", false, Number::Float(1.5), "timeout")
)]
#[case::a_falsy_flag_is_unhealthy(
    "{'dep-a': {'is_healthy': 0, 'timestamp': 95, 'reason': 'timeout'}}",
    one("dep-a", false, Number::Int(95), "timeout")
)]
#[case::a_missing_flag_is_healthy(
    "{'dep-a': {'timestamp': 1.5}}",
    one("dep-a", true, Number::Float(1.5), "")
)]
#[case::a_timestamp_that_is_no_number_is_stale(
    "{'dep-a': {'is_healthy': False, 'timestamp': 'soon', 'reason': 'timeout'}}",
    one("dep-a", false, Number::Float(0.0), "timeout")
)]
#[case::a_bool_timestamp_is_no_number(
    "{'dep-a': {'is_healthy': False, 'timestamp': True, 'reason': 'timeout'}}",
    one("dep-a", false, Number::Float(0.0), "timeout")
)]
#[case::what_is_no_record_is_dropped_alone(
    "{'dep-a': 'not a record', 'dep-b': None, 'dep-c': {'is_healthy': False, 'timestamp': 1.5, 'reason': 'timeout'}}",
    one("dep-c", false, Number::Float(1.5), "timeout")
)]
#[tokio::test]
async fn a_record_another_writer_left_is_read_the_way_mains_readers_read_it(
    #[case] written: &str,
    #[case] read: HealthStates,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let reader = DeploymentHealth::new(Some(remote(connect(&deployment))), None);
    raw.send(commands::set(STATES_KEY, written, Duration::from_secs(90)))
        .await
        .unwrap();

    assert_eq!(
        (reader.get_shared().await, reader.get().await.unwrap()),
        (Some(read.clone()), Some(read))
    );
}

#[rstest]
#[tokio::test]
async fn an_entry_that_is_no_dict_reads_as_nothing_shared(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let reader = DeploymentHealth::new(Some(remote(connect(&deployment))), None);
    raw.send(commands::set(
        STATES_KEY,
        "['dep-a']",
        Duration::from_secs(90),
    ))
    .await
    .unwrap();

    assert_eq!(
        (reader.get_shared().await, reader.get().await.unwrap()),
        (None, None)
    );
}
