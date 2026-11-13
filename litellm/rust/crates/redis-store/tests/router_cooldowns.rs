mod support;

use std::time::Duration;

use litellm_redis_client::commands;
use litellm_redis_client::{Operation, Ttl};
use litellm_redis_store::Send as RedisSend;
use litellm_redis_store::{
    Number, RequestRedis,
    router_cooldowns::{Cooldown, RouterCooldowns},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{ALLOWED_FAILS, COOLDOWN_POLICY as POLICY, connect, cooldowns, remote, within};
use tokio::runtime::Handle;

async fn read_active(
    store: &std::sync::Arc<RouterCooldowns>,
    model_ids: &[String],
) -> Vec<(String, Cooldown)> {
    store
        .active(RedisSend::Now, model_ids)
        .unwrap()
        .await
        .unwrap()
}

/// A clock fixed just after the fixture timestamps, so every fixture cooldown is still active.
fn fixture_clock() -> litellm_redis_store::Clock {
    std::sync::Arc::new(|| Duration::from_secs_f64(1_730_000_000.75))
}

#[rstest]
#[tokio::test]
async fn put_writes_the_bytes_mains_set_path_writes_with_whole_second_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values("json", "str_dict")] format: &str,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = std::sync::Arc::new(RouterCooldowns::new(
        Some(remote(connect(&deployment))),
        POLICY,
    ));

    for case in cooldowns() {
        let ttl = Duration::from_secs_f64(case.ttl_seconds);
        let (written_by_put, expected) = match format {
            "json" => (
                store.put(case.model_id, &case.cooldown, ttl).await,
                case.json,
            ),
            _ => (
                store.put_str(case.model_id, &case.cooldown, ttl).await,
                case.str_dict,
            ),
        };
        written_by_put.unwrap();
        let key = case.key;

        let written: Option<String> = raw.send(commands::get(key)).await.unwrap();
        let Ttl::Expires(remaining) = raw.send(commands::ttl(key)).await.unwrap() else {
            panic!("{key} has no TTL");
        };
        assert_eq!(written.as_deref(), Some(expected));
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
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(connect(&deployment))),
        POLICY,
        fixture_clock(),
    ));

    for case in cooldowns() {
        let bytes = match format {
            "json" => case.json,
            _ => case.str_dict,
        };
        raw.send(commands::set(case.key, bytes, Duration::from_secs(60)))
            .await
            .unwrap();

        assert_eq!(
            read_active(&store, &[case.model_id.to_owned()]).await,
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
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(connect(&deployment))),
        POLICY,
        late,
    ));
    let models = cooldowns()
        .iter()
        .map(|case| case.model_id.to_owned())
        .collect::<Vec<_>>();
    for case in cooldowns() {
        raw.send(commands::set(
            case.key,
            case.str_dict,
            Duration::from_secs(60),
        ))
        .await
        .unwrap();
    }

    assert_eq!(
        store.minimum(&models).await.unwrap(),
        Some(Number::Float(0.5))
    );
    assert_eq!(read_active(&store, &models).await, vec![]);
}

#[rstest]
#[tokio::test]
async fn allowed_fails_count_across_async_and_sync_routers_and_keep_the_first_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let first = std::sync::Arc::new(RouterCooldowns::new(
        Some(remote(connect(&deployment))),
        POLICY,
    ));
    let second = std::sync::Arc::new(RouterCooldowns::new(
        Some(remote(connect(&deployment))),
        POLICY,
    ));

    for case in ALLOWED_FAILS {
        let key = case.key;
        raw.send(commands::del(&[key])).await.unwrap();
        let suffix = case.suffix;

        let counts = [
            first
                .increment_allowed_fails("dep-a", suffix, Duration::from_secs_f64(30.9))
                .await
                .unwrap(),
            second
                .increment_allowed_fails_int("dep-a", suffix, Duration::from_secs(120))
                .await
                .unwrap(),
        ];
        let Ttl::Expires(remaining) = raw.send(commands::ttl(key)).await.unwrap() else {
            panic!("{key} has no TTL");
        };

        assert_eq!(counts, [1, 2], "{key}");
        assert!(within(remaining, 30), "{remaining:?}");
    }
}

#[rstest]
#[case::async_increment_cache_sends_incrbyfloat("INCRBYFLOAT")]
#[case::increment_cache_sends_incrby("INCRBY")]
#[tokio::test]
async fn allowed_fails_send_the_increment_mains_path_sends(
    #[case] command: &'static str,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = support::connect_observed(&deployment);
    let store = std::sync::Arc::new(RouterCooldowns::new(Some(remote(client)), POLICY));
    let ttl = Duration::from_secs(60);

    let count = match command {
        "INCRBYFLOAT" => store.increment_allowed_fails("dep-a", None, ttl).await,
        _ => store.increment_allowed_fails_int("dep-a", None, ttl).await,
    };

    assert_eq!(count.unwrap(), 1);
    assert_eq!(trips.seen().first(), Some(&Operation::Command(command)));
}

#[rstest]
#[tokio::test]
async fn python_written_fail_counts_continue_in_rust(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = std::sync::Arc::new(RouterCooldowns::new(
        Some(remote(connect(&deployment))),
        POLICY,
    ));
    raw.send(commands::incr_by(
        "deployment:dep-a:allowed_fails",
        4,
        Duration::from_secs(60),
    ))
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
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(None, POLICY, fixture_clock()));
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
        read_active(&store, &[case.model_id.to_owned()]).await,
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
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(connect(&deployment))),
        POLICY,
        fixture_clock(),
    ));
    let [case, _] = cooldowns();
    store
        .put(case.model_id, &case.cooldown, Duration::from_secs(90))
        .await
        .unwrap();
    deployment.stop().await.unwrap();

    let active = read_active(
        &store,
        &[case.model_id.to_owned(), "dep-missing".to_owned()],
    )
    .await;
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

    fn clock(self: &std::sync::Arc<Self>) -> litellm_redis_store::Clock {
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
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(connect(&deployment))),
        POLICY,
        time.clock(),
    ));
    let [case, _] = cooldowns();

    let before = read_active(&store, &[case.model_id.to_owned()]).await;
    raw.send(commands::set(
        case.key,
        case.str_dict,
        Duration::from_secs(60),
    ))
    .await
    .unwrap();
    let throttled = read_active(&store, &[case.model_id.to_owned()]).await;
    time.advance(Duration::from_secs(1));
    let refreshed = read_active(&store, &[case.model_id.to_owned()]).await;

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
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(guarded.clone())),
        POLICY,
        time.clock(),
    ));
    let [case, _] = cooldowns();
    deployment.stop().await.unwrap();
    read_active(&store, &[case.model_id.to_owned()]).await;
    time.advance(Duration::from_secs(1));
    let refused = read_active(&store, &[case.model_id.to_owned()]).await;
    deployment.resume().await.unwrap();
    litellm_redis_testing::eventually(async || guarded.ping().await.ok()).await;
    raw.send(commands::set(
        case.key,
        case.str_dict,
        Duration::from_secs(60),
    ))
    .await
    .unwrap();

    let recovered = read_active(&store, &[case.model_id.to_owned()]).await;

    assert_eq!(refused, vec![]);
    assert_eq!(
        recovered,
        vec![(case.model_id.to_owned(), case.cooldown.clone())]
    );
}

#[rstest]
#[tokio::test]
async fn a_declared_active_read_rides_the_request_batch_and_answers_what_active_does(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = support::connect_observed(&deployment);
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(client)),
        POLICY,
        fixture_clock(),
    ));
    let models = cooldowns().map(|case| case.model_id.to_owned());
    for case in cooldowns() {
        raw.send(commands::set(
            case.key,
            case.str_dict,
            Duration::from_secs(60),
        ))
        .await
        .unwrap();
    }
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    let declared = store.active(RedisSend::PreCall(&request), &models).unwrap();
    assert!(trips.seen().is_empty());
    request.flush_pre().await;
    let active = declared.await.unwrap();

    let one_mget = match topology {
        Topology::Standalone => Operation::Pipeline { ops: 1 },
        Topology::Cluster => Operation::Command("MGET"),
    };
    assert_eq!(trips.seen(), [one_mget]);
    assert_eq!(
        active,
        cooldowns()
            .map(|case| (case.model_id.to_owned(), case.cooldown))
            .to_vec()
    );
    assert_eq!(read_active(&store, &models).await, active);
    assert_eq!(trips.seen().len(), 1, "the declared read backfilled memory");
}

#[rstest]
#[tokio::test]
async fn a_cooldown_recorded_after_the_declaration_wins_over_the_redis_miss(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(connect(&deployment))),
        POLICY,
        fixture_clock(),
    ));
    let [case, _] = cooldowns();
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    let declared = store
        .active(RedisSend::PreCall(&request), &[case.model_id.to_owned()])
        .unwrap();
    request.flush_pre().await;
    store
        .put(case.model_id, &case.cooldown, Duration::from_secs(90))
        .await
        .unwrap();

    assert_eq!(
        declared.await.unwrap(),
        vec![(case.model_id.to_owned(), case.cooldown)]
    );
}

#[rstest]
#[tokio::test]
async fn a_breaker_refusal_of_a_declared_read_does_not_hold_back_the_next_read() {
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
    let guarded = support::connect_guarded(&deployment, breaker);
    let store = std::sync::Arc::new(RouterCooldowns::with_clock(
        Some(remote(guarded.clone())),
        POLICY,
        fixture_clock(),
    ));
    let [case, _] = cooldowns();
    let models = [case.model_id.to_owned()];
    deployment.stop().await.unwrap();
    let opened = RequestRedis::new(Handle::current(), Duration::from_secs(60));
    let failed = store
        .active(RedisSend::PreCall(&opened), &["dep-other".to_owned()])
        .unwrap();
    opened.flush_pre().await;
    failed.await.unwrap();
    let refused_request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    let refused = store
        .active(RedisSend::PreCall(&refused_request), &models)
        .unwrap();
    refused_request.flush_pre().await;
    let refused = refused.await.unwrap();
    deployment.resume().await.unwrap();
    litellm_redis_testing::eventually(async || guarded.ping().await.ok()).await;
    raw.send(commands::set(
        case.key,
        case.str_dict,
        Duration::from_secs(60),
    ))
    .await
    .unwrap();

    assert_eq!(refused, vec![]);
    assert_eq!(
        read_active(&store, &models).await,
        vec![(case.model_id.to_owned(), case.cooldown.clone())]
    );
}
