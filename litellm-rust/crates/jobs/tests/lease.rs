use std::time::Duration;

use litellm_jobs::{Acquire, HolderId, JobName, LeaseStore, MemoryLeaseStore, Renewal};
use rstest::{fixture, rstest};

const TTL: Duration = Duration::from_secs(60);

struct Pods {
    store: MemoryLeaseStore,
    job: JobName,
    first: HolderId,
    second: HolderId,
}

#[fixture]
fn pods() -> Pods {
    Pods {
        store: MemoryLeaseStore::default(),
        job: JobName::new("db_spend_update_job"),
        first: HolderId::random(),
        second: HolderId::random(),
    }
}

#[rstest]
#[tokio::test(start_paused = true)]
async fn a_held_lease_is_busy_for_others_and_reentrant_for_its_holder(pods: Pods) {
    assert_eq!(
        pods.store.try_acquire(&pods.job, &pods.first, TTL).await,
        Ok(Acquire::Held)
    );
    assert_eq!(
        pods.store.try_acquire(&pods.job, &pods.second, TTL).await,
        Ok(Acquire::Busy)
    );
    assert_eq!(
        pods.store.try_acquire(&pods.job, &pods.first, TTL).await,
        Ok(Acquire::Held)
    );
}

#[rstest]
#[tokio::test(start_paused = true)]
async fn an_expired_lease_is_taken_over_and_the_old_holder_cannot_renew(pods: Pods) {
    pods.store
        .try_acquire(&pods.job, &pods.first, TTL)
        .await
        .unwrap();
    tokio::time::advance(TTL + Duration::from_secs(1)).await;

    assert_eq!(
        pods.store.try_acquire(&pods.job, &pods.second, TTL).await,
        Ok(Acquire::Held)
    );
    assert_eq!(
        pods.store.renew(&pods.job, &pods.first, TTL).await,
        Ok(Renewal::Lost)
    );
    assert_eq!(
        pods.store.renew(&pods.job, &pods.second, TTL).await,
        Ok(Renewal::Extended)
    );
}

#[rstest]
#[tokio::test(start_paused = true)]
async fn renewal_pushes_expiry_past_the_original_ttl(pods: Pods) {
    pods.store
        .try_acquire(&pods.job, &pods.first, TTL)
        .await
        .unwrap();
    tokio::time::advance(TTL / 2).await;
    pods.store.renew(&pods.job, &pods.first, TTL).await.unwrap();
    tokio::time::advance(TTL * 3 / 4).await;

    assert_eq!(
        pods.store.try_acquire(&pods.job, &pods.second, TTL).await,
        Ok(Acquire::Busy)
    );
}

#[rstest]
#[case::by_a_stranger(false, Acquire::Busy)]
#[case::by_the_holder(true, Acquire::Held)]
#[tokio::test(start_paused = true)]
async fn only_the_holder_can_release(
    pods: Pods,
    #[case] released_by_holder: bool,
    #[case] second_pod_gets: Acquire,
) {
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
        pods.store.try_acquire(&pods.job, &pods.second, TTL).await,
        Ok(second_pod_gets)
    );
}
