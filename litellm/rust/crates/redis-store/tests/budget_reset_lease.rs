mod support;

use std::time::Duration;

use litellm_redis_client::commands;
use litellm_redis_store::budget_reset_lease::Lease;
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::connect;

#[rstest]
#[tokio::test]
async fn python_lease_blocks_rust_and_stale_release_leaves_the_new_owner(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    const KEY: &str = "cronjob_lock:reset_budget_job";
    const PYTHON_OWNER: &str = "\"python-pod\"";
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let redis = connect(&deployment);
    let ttl = Duration::from_secs(900);
    assert!(
        redis
            .send(commands::set_nx(KEY, PYTHON_OWNER, ttl))
            .await
            .unwrap()
    );
    assert!(Lease::acquire(&redis).await.unwrap().is_none());
    redis.send(commands::del(&[KEY])).await.unwrap();
    let lease = Lease::acquire(&redis).await.unwrap().unwrap();
    assert!(lease.is_current().await.unwrap());
    assert!(
        !redis
            .send(commands::set_nx(KEY, PYTHON_OWNER, ttl))
            .await
            .unwrap()
    );
    redis
        .send(commands::set(KEY, PYTHON_OWNER, ttl))
        .await
        .unwrap();
    assert!(!lease.is_current().await.unwrap());
    lease.release().await.unwrap();
    assert_eq!(
        redis
            .send(commands::get::<String>(KEY))
            .await
            .unwrap()
            .as_deref(),
        Some(PYTHON_OWNER)
    );
}
