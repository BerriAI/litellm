//! Runs against a real Redis. Set `LITELLM_TEST_REDIS_URL` (for example `redis://127.0.0.1:6379`)
//! to run them; without it every test returns early.

use std::time::Duration;

use litellm_jobs::{Acquire, HolderId, JobName, LeaseStore, Renewal};
use litellm_jobs_redis::{LeaseNaming, PythonLeaseNaming, RedisLeaseStore};
use redis::{AsyncCommands, aio::MultiplexedConnection};
use rstest::rstest;

const TTL: Duration = Duration::from_millis(300);

struct Pods {
    connection: MultiplexedConnection,
    store: RedisLeaseStore<MultiplexedConnection, PythonLeaseNaming>,
    job: JobName,
    first: HolderId,
    second: HolderId,
}

async fn pods() -> Option<Pods> {
    let url = std::env::var("LITELLM_TEST_REDIS_URL").ok()?;
    let connection = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    Some(Pods {
        store: RedisLeaseStore::new(connection.clone(), PythonLeaseNaming),
        connection,
        job: JobName::new(format!("test_job_{}", HolderId::random().as_str())),
        first: HolderId::random(),
        second: HolderId::random(),
    })
}

#[tokio::test]
async fn a_held_lease_is_busy_for_others_and_reentrant_for_its_holder() {
    let Some(pods) = pods().await else { return };

    assert_eq!(
        pods.store
            .try_acquire(&pods.job, &pods.first, TTL)
            .await
            .unwrap(),
        Acquire::Held
    );
    assert_eq!(
        pods.store
            .try_acquire(&pods.job, &pods.second, TTL)
            .await
            .unwrap(),
        Acquire::Busy
    );
    assert_eq!(
        pods.store
            .try_acquire(&pods.job, &pods.first, TTL)
            .await
            .unwrap(),
        Acquire::Held
    );
}

#[tokio::test]
async fn an_expired_lease_is_taken_over_and_the_old_holder_cannot_renew() {
    let Some(pods) = pods().await else { return };
    pods.store
        .try_acquire(&pods.job, &pods.first, TTL)
        .await
        .unwrap();
    tokio::time::sleep(TTL + Duration::from_millis(100)).await;

    assert_eq!(
        pods.store
            .try_acquire(&pods.job, &pods.second, TTL)
            .await
            .unwrap(),
        Acquire::Held
    );
    assert_eq!(
        pods.store.renew(&pods.job, &pods.first, TTL).await.unwrap(),
        Renewal::Lost
    );
    assert_eq!(
        pods.store
            .renew(&pods.job, &pods.second, TTL)
            .await
            .unwrap(),
        Renewal::Extended
    );
}

#[tokio::test]
async fn renewal_pushes_expiry_past_the_original_ttl() {
    let Some(pods) = pods().await else { return };
    pods.store
        .try_acquire(&pods.job, &pods.first, TTL)
        .await
        .unwrap();
    tokio::time::sleep(TTL / 2).await;
    pods.store.renew(&pods.job, &pods.first, TTL).await.unwrap();
    tokio::time::sleep(TTL * 3 / 4).await;

    assert_eq!(
        pods.store
            .try_acquire(&pods.job, &pods.second, TTL)
            .await
            .unwrap(),
        Acquire::Busy
    );
}

#[rstest]
#[case::by_a_stranger(false, Acquire::Busy)]
#[case::by_the_holder(true, Acquire::Held)]
#[tokio::test]
async fn only_the_holder_can_release(
    #[case] released_by_holder: bool,
    #[case] second_pod_gets: Acquire,
) {
    let Some(pods) = pods().await else { return };
    pods.store
        .try_acquire(&pods.job, &pods.first, TTL)
        .await
        .unwrap();
    let releaser = if released_by_holder {
        &pods.first
    } else {
        &pods.second
    };
    pods.store.release(&pods.job, releaser).await.unwrap();

    assert_eq!(
        pods.store
            .try_acquire(&pods.job, &pods.second, TTL)
            .await
            .unwrap(),
        second_pod_gets
    );
}

#[tokio::test]
async fn python_and_rust_pods_exclude_each_other() {
    let Some(mut pods) = pods().await else { return };
    let key = PythonLeaseNaming.key(&pods.job);
    let python_pod = "python-pod";
    let _: () = redis::cmd("SET")
        .arg(&key)
        .arg(serde_json::to_string(python_pod).unwrap())
        .arg("NX")
        .arg("EX")
        .arg(60)
        .query_async(&mut pods.connection)
        .await
        .unwrap();

    assert_eq!(
        pods.store
            .try_acquire(&pods.job, &pods.first, TTL)
            .await
            .unwrap(),
        Acquire::Busy
    );

    let _: () = pods.connection.del(&key).await.unwrap();
    pods.store
        .try_acquire(&pods.job, &pods.first, TTL)
        .await
        .unwrap();
    let held_by: String = pods.connection.get(&key).await.unwrap();
    assert_eq!(
        serde_json::from_str::<String>(&held_by).unwrap(),
        pods.first.as_str()
    );
}
