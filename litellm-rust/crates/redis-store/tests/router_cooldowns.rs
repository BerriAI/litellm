mod support;

use std::time::Duration;

use litellm_redis_client::{Expiry, Ttl};
use litellm_redis_store::{
    Number,
    router_cooldowns::{POLICY, RouterCooldowns},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{ALLOWED_FAILS, connect, cooldowns, within};

/// A clock fixed just after the fixture timestamps, so every fixture cooldown is still active.
fn fixture_clock() -> litellm_cache::Clock {
    std::sync::Arc::new(|| Duration::from_secs_f64(1_730_000_000.75))
}

#[rstest]
#[tokio::test]
async fn put_writes_mains_str_dict_bytes_with_whole_second_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterCooldowns::new(Some(connect(&deployment)), POLICY);

    for case in cooldowns() {
        let ttl = Duration::from_secs_f64(case.ttl_seconds);
        store.put(case.model_id, &case.cooldown, ttl).await.unwrap();
        let key = case.key;

        let written: Option<String> = raw.get(key).await.unwrap();
        let Ttl::Expires(remaining) = raw.ttl(key).await.unwrap() else {
            panic!("{key} has no TTL");
        };
        assert_eq!(written.as_deref(), Some(case.str_dict));
        assert!(within(remaining, case.redis_ttl_seconds), "{remaining:?}");
    }
}

#[rstest]
#[tokio::test]
async fn reads_cooldowns_python_wrote_as_str_dict_or_json(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values("str_dict", "json")] format: &str,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), POLICY, fixture_clock());

    for case in cooldowns() {
        let bytes = match format {
            "json" => case.json,
            _ => case.str_dict,
        };
        raw.set(case.key, bytes, Duration::from_secs(60))
            .await
            .unwrap();

        assert_eq!(
            store.active(&[case.model_id.to_owned()]).await.unwrap(),
            vec![(case.model_id.to_owned(), case.cooldown)]
        );
    }
}

#[rstest]
#[tokio::test]
async fn an_ended_cooldown_is_not_active_but_still_counts_toward_the_minimum(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let late = std::sync::Arc::new(|| Duration::from_secs(1_730_000_100));
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), POLICY, late);
    let models = cooldowns()
        .iter()
        .map(|case| case.model_id.to_owned())
        .collect::<Vec<_>>();
    for case in cooldowns() {
        raw.set(case.key, case.str_dict, Duration::from_secs(60))
            .await
            .unwrap();
    }

    assert_eq!(
        store.minimum(&models).await.unwrap(),
        Some(Number::Float(0.5))
    );
    assert_eq!(store.active(&models).await.unwrap(), vec![]);
}

#[rstest]
#[tokio::test]
async fn allowed_fails_count_across_routers_and_keep_the_first_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let first = RouterCooldowns::new(Some(connect(&deployment)), POLICY);
    let second = RouterCooldowns::new(Some(connect(&deployment)), POLICY);

    for case in ALLOWED_FAILS {
        let key = case.key;
        raw.del(&[key]).await.unwrap();
        let suffix = case.suffix;

        let counts = [
            first
                .increment_allowed_fails("dep-a", suffix, Duration::from_secs_f64(30.9))
                .await
                .unwrap(),
            second
                .increment_allowed_fails("dep-a", suffix, Duration::from_secs(120))
                .await
                .unwrap(),
        ];
        let Ttl::Expires(remaining) = raw.ttl(key).await.unwrap() else {
            panic!("{key} has no TTL");
        };

        assert_eq!(counts, [1, 2], "{key}");
        assert!(within(remaining, 30), "{remaining:?}");
    }
}

#[rstest]
#[tokio::test]
async fn python_written_fail_counts_continue_in_rust(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterCooldowns::new(Some(connect(&deployment)), POLICY);
    raw.incr_by(
        "deployment:dep-a:allowed_fails",
        4,
        Expiry::KeepExisting(Duration::from_secs(60)),
    )
    .await
    .unwrap();

    assert_eq!(
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
        5
    );
}

#[rstest]
#[tokio::test]
async fn a_router_without_redis_keeps_its_cooldowns_and_counts_in_memory() {
    let store = RouterCooldowns::with_clock(None, POLICY, fixture_clock());
    let [case, _] = cooldowns();

    store
        .put(case.model_id, &case.cooldown, Duration::from_secs(90))
        .await
        .unwrap();
    let fails = [
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
    ];
    store.flush_local_counters().unwrap();

    assert_eq!(
        store.active(&[case.model_id.to_owned()]).await.unwrap(),
        vec![(case.model_id.to_owned(), case.cooldown)]
    );
    assert_eq!(fails, [1, 2]);
    assert_eq!(
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
        1
    );
}

#[rstest]
#[tokio::test]
async fn a_redis_outage_serves_memory_and_counts_fails_locally() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), POLICY, fixture_clock());
    let [case, _] = cooldowns();
    store
        .put(case.model_id, &case.cooldown, Duration::from_secs(90))
        .await
        .unwrap();
    deployment.stop().await.unwrap();

    let active = store
        .active(&[case.model_id.to_owned(), "dep-missing".to_owned()])
        .await
        .unwrap();
    let fails = store
        .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
        .await
        .unwrap();

    assert_eq!(active, vec![(case.model_id.to_owned(), case.cooldown)]);
    assert_eq!(fails, 1);
}

struct StepClock(std::sync::Mutex<Duration>);

impl StepClock {
    fn at(seconds: f64) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self(std::sync::Mutex::new(Duration::from_secs_f64(
            seconds,
        ))))
    }

    fn advance(&self, by: Duration) {
        *self.0.lock().unwrap() += by;
    }

    fn clock(self: &std::sync::Arc<Self>) -> litellm_cache::Clock {
        let this = self.clone();
        std::sync::Arc::new(move || *this.0.lock().unwrap())
    }
}

#[rstest]
#[tokio::test]
async fn a_miss_is_read_from_redis_again_only_after_the_one_second_refresh(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let time = StepClock::at(1_730_000_000.75);
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), POLICY, time.clock());
    let [case, _] = cooldowns();

    let before = store.active(&[case.model_id.to_owned()]).await.unwrap();
    raw.set(case.key, case.str_dict, Duration::from_secs(60))
        .await
        .unwrap();
    let throttled = store.active(&[case.model_id.to_owned()]).await.unwrap();
    time.advance(Duration::from_secs(1));
    let refreshed = store.active(&[case.model_id.to_owned()]).await.unwrap();

    assert_eq!(before, vec![]);
    assert_eq!(throttled, vec![]);
    assert_eq!(
        refreshed,
        vec![(case.model_id.to_owned(), case.cooldown.clone())]
    );
}

#[rstest]
#[tokio::test]
async fn a_breaker_refusal_does_not_hold_back_the_next_read() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let breaker = litellm_circuit_breaker::Breaker::new(
        litellm_circuit_breaker::Config {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(200),
            ..litellm_circuit_breaker::Config::default()
        },
        Box::new(|_| {}),
    );
    let time = StepClock::at(1_730_000_000.75);
    let guarded = support::connect_guarded(&deployment, breaker);
    let store = RouterCooldowns::with_clock(Some(guarded.clone()), POLICY, time.clock());
    let [case, _] = cooldowns();
    deployment.stop().await.unwrap();
    store.active(&[case.model_id.to_owned()]).await.unwrap();
    time.advance(Duration::from_secs(1));
    let refused = store.active(&[case.model_id.to_owned()]).await.unwrap();
    deployment.resume().await.unwrap();
    litellm_redis_testing::eventually(async || guarded.ping().await.ok()).await;
    raw.set(case.key, case.str_dict, Duration::from_secs(60))
        .await
        .unwrap();

    let recovered = store.active(&[case.model_id.to_owned()]).await.unwrap();

    assert_eq!(refused, vec![]);
    assert_eq!(
        recovered,
        vec![(case.model_id.to_owned(), case.cooldown.clone())]
    );
}
