mod support;

use std::time::Duration;

use litellm_python_compat::Value;
use litellm_redis_client::{Ttl, commands};
use litellm_redis_store::deployment_health::DeploymentHealth;
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, remote, within};

const STATES_KEY: &str = "litellm:health_check:deployment_health_state";

fn states() -> Value {
    let text = |text: &str| Value::Str(text.to_owned());
    Value::Dict(vec![(
        text("dep-a"),
        Value::Dict(vec![
            (text("is_healthy"), Value::Bool(false)),
            (text("timestamp"), Value::Float(1.5)),
            (text("reason"), text("timeout")),
        ]),
    )])
}

#[rstest]
#[tokio::test]
async fn states_one_pod_writes_are_read_by_another_under_mains_key_and_encoding(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let writer = DeploymentHealth::new(Some(remote(connect(&deployment))));
    let reader = DeploymentHealth::new(Some(remote(connect(&deployment))));

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
    let store = DeploymentHealth::new(None);

    store.set(states(), Duration::from_secs(90)).await.unwrap();

    assert_eq!(
        (store.get().await.unwrap(), store.get_shared().await),
        (Some(states()), None)
    );
}
