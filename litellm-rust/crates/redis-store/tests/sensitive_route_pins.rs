mod support;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_redis_client::{Ttl, commands};
use litellm_redis_store::{Clock, sensitive_route_pins::SensitiveRoutePins};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, remote, within};

const PIN_KEY: &str = "{sensitive_route:tenant-a:session-1}:model";
const HOUR: Duration = Duration::from_secs(3600);

struct StepClock(Mutex<Duration>);

impl StepClock {
    fn start() -> Arc<Self> {
        Arc::new(Self(Mutex::new(Duration::from_secs(1_000))))
    }

    fn advance(&self, by: Duration) {
        *self.0.lock().unwrap() += by;
    }

    fn clock(self: &Arc<Self>) -> Clock {
        let this = self.clone();
        Arc::new(move || *this.0.lock().unwrap())
    }
}

#[rstest]
#[tokio::test]
async fn a_pin_one_pod_writes_is_read_by_another_under_mains_key_and_encoding(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let writer = SensitiveRoutePins::new(Some(remote(connect(&deployment))));
    let reader = SensitiveRoutePins::new(Some(remote(connect(&deployment))));

    writer
        .set("tenant-a", "session-1", "on-prem-model", HOUR)
        .await
        .unwrap();

    let written: Option<String> = raw.send(commands::get(PIN_KEY)).await.unwrap();
    let Ttl::Expires(remaining) = raw.send(commands::ttl(PIN_KEY)).await.unwrap() else {
        panic!("{PIN_KEY} has no TTL");
    };
    assert_eq!(written.as_deref(), Some("\"on-prem-model\""));
    assert!(within(remaining, 3600), "{remaining:?}");
    assert_eq!(
        reader.get("tenant-a", "session-1", HOUR).await.unwrap(),
        Some("on-prem-model".to_owned())
    );
    assert_eq!(
        reader.get("tenant-b", "session-1", HOUR).await.unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn a_pin_read_from_redis_stays_in_memory_for_its_remaining_ttl_only() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let time = StepClock::start();
    let reader = SensitiveRoutePins::with_clock(Some(remote(connect(&deployment))), time.clock());
    raw.send(commands::set(
        PIN_KEY,
        "\"on-prem-model\"",
        Duration::from_secs(30),
    ))
    .await
    .unwrap();

    let read = reader.get("tenant-a", "session-1", HOUR).await.unwrap();
    deployment.stop().await.unwrap();
    time.advance(Duration::from_secs(20));
    let before_expiry = reader.get("tenant-a", "session-1", HOUR).await.unwrap();
    time.advance(Duration::from_secs(15));
    let after_expiry = reader.get("tenant-a", "session-1", HOUR).await.unwrap();

    assert_eq!(read, Some("on-prem-model".to_owned()));
    assert_eq!(before_expiry, read);
    assert_eq!(after_expiry, None);
}

#[rstest]
#[tokio::test]
async fn a_redis_outage_keeps_pinning_sessions_in_memory() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = SensitiveRoutePins::new(Some(remote(connect(&deployment))));
    store
        .set("tenant-a", "session-1", "on-prem-model", HOUR)
        .await
        .unwrap();
    deployment.stop().await.unwrap();

    let pinned_before = store.get("tenant-a", "session-1", HOUR).await.unwrap();
    store
        .set("tenant-a", "session-2", "other-model", HOUR)
        .await
        .unwrap();
    let pinned_during = store.get("tenant-a", "session-2", HOUR).await.unwrap();

    assert_eq!(pinned_before, Some("on-prem-model".to_owned()));
    assert_eq!(pinned_during, Some("other-model".to_owned()));
}

#[tokio::test]
async fn without_redis_pins_live_in_memory_until_they_expire() {
    let time = StepClock::start();
    let store = SensitiveRoutePins::with_clock(None, time.clock());
    store
        .set(
            "tenant-a",
            "session-1",
            "on-prem-model",
            Duration::from_secs(10),
        )
        .await
        .unwrap();

    let pinned = store.get("tenant-a", "session-1", HOUR).await.unwrap();
    time.advance(Duration::from_secs(11));
    let expired = store.get("tenant-a", "session-1", HOUR).await.unwrap();

    assert_eq!(pinned, Some("on-prem-model".to_owned()));
    assert_eq!(expired, None);
}
