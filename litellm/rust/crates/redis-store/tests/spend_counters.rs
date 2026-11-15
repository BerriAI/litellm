mod support;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::commands;
use litellm_redis_client::{Operation, RedisClient, Ttl};
use litellm_redis_store::Send as RedisSend;
use litellm_redis_store::{
    Number, RequestRedis,
    router_cooldowns::RouterCooldowns,
    spend_counters::{LostSeed, Reading, SpendCounter, SpendCounters},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{COOLDOWN_POLICY, connect, connect_observed, remote, within};
use tokio::runtime::Handle;

const TTL: Duration = Duration::from_secs(60);
const COMMITTED_TTL: Duration = Duration::from_secs(300);
const LONG_DEADLINE: Duration = Duration::from_secs(60);

const READS: [(&str, Number); 4] = [
    ("7.25", Number::Float(7.25)),
    ("3", Number::Int(3)),
    ("0.0", Number::Float(0.0)),
    ("1e-05", Number::Float(1e-05)),
];

const SETS: [(Number, &str); 4] = [
    (Number::Float(0.0), "0.0"),
    (Number::Int(12), "12"),
    (Number::Float(1e-05), "1e-05"),
    (Number::Float(0.30000000000000004), "0.30000000000000004"),
];

const SET_MAXES: [(Number, &str); 3] = [
    (Number::Float(2.5), "2.5"),
    (Number::Int(100), "100"),
    (Number::Float(1e-05), "1e-05"),
];

fn team(id: &str) -> SpendCounter {
    SpendCounter::Team {
        team_id: id.to_owned(),
        window: None,
    }
}

fn committed(counter: SpendCounter) -> SpendCounter {
    SpendCounter::Committed(Box::new(counter))
}

fn store(client: Option<RedisClient>) -> Arc<SpendCounters> {
    Arc::new(SpendCounters::new(client.map(remote)))
}

/// Mirrored by `tests/unit/redis/stores/test_spend_counters.py`, which checks the same keys and
/// bytes against Python's own key builders and `RedisCache` codec.
#[rstest]
#[case::redis(Reading::Redis(Some(Number::Int(3))), "redis", Some(Number::Int(3)))]
#[case::redis_miss(Reading::Redis(None), "redis", None)]
#[case::memory(
    Reading::Memory(Some(Number::Float(2.5))),
    "memory",
    Some(Number::Float(2.5))
)]
fn a_reading_names_where_it_was_answered(
    #[case] reading: Reading,
    #[case] source: &str,
    #[case] value: Option<Number>,
) {
    assert_eq!(<&str>::from(reading), source);
    assert_eq!(reading.value(), value);
}

#[rstest]
#[case(
    SpendCounter::Key {
        token: "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b".to_owned(),
        window: None,
    },
    "spend:key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
)]
#[case(
    SpendCounter::Key {
        token: "88dc28d0".to_owned(),
        window: Some("30d".to_owned()),
    },
    "spend:key:88dc28d0:window:30d"
)]
#[case(
    SpendCounter::Team {
        team_id: "team-a".to_owned(),
        window: None,
    },
    "spend:team:team-a"
)]
#[case(
    SpendCounter::Team {
        team_id: "team-a".to_owned(),
        window: Some("1h".to_owned()),
    },
    "spend:team:team-a:window:1h"
)]
#[case(
    SpendCounter::TeamMember {
        user_id: "user:with:colons".to_owned(),
        team_id: "team-a".to_owned(),
    },
    "spend:team_member:user:with:colons:team-a"
)]
#[case(SpendCounter::User("user@example.com".to_owned()), "spend:user:user@example.com")]
#[case(
    SpendCounter::UserWindow {
        user_id: "user:window:1h".to_owned(),
        window: "1mo".to_owned(),
    },
    "spend:user_window:user:window:1h:1mo"
)]
#[case(SpendCounter::EndUser("customer é 1".to_owned()), "spend:end_user:customer é 1")]
#[case(SpendCounter::Org("org-1".to_owned()), "spend:org:org-1")]
#[case(SpendCounter::Tag("prod:eu".to_owned()), "spend:tag:prod:eu")]
#[case(SpendCounter::Project("proj-1".to_owned()), "spend:project:proj-1")]
#[case(
    SpendCounter::ModelAccessGroup("gpt-group".to_owned()),
    "spend:model_access_group:gpt-group"
)]
#[case(SpendCounter::ShadowEval("job-7".to_owned()), "spend:shadow_eval:job-7")]
#[case::committed_key(
    committed(SpendCounter::Key {
        token: "88dc28d0".to_owned(),
        window: None,
    }),
    "spend:committed:key:88dc28d0"
)]
#[case::committed_team(committed(team("team-a")), "spend:committed:team:team-a")]
#[case::committed_user(
    committed(SpendCounter::User("user@example.com".to_owned())),
    "spend:committed:user:user@example.com"
)]
#[case::committed_org(
    committed(SpendCounter::Org("org-1".to_owned())),
    "spend:committed:org:org-1"
)]
#[case::committed_end_user(
    committed(SpendCounter::EndUser("customer é 1".to_owned())),
    "spend:committed:end_user:customer é 1"
)]
fn every_counter_formats_and_parses_back_to_mains_key(
    #[case] counter: SpendCounter,
    #[case] key: &str,
) {
    assert_eq!(counter.key(), key);
    assert_eq!(SpendCounter::parse(key).unwrap(), counter, "{key}");
}

#[rstest]
#[case("spend_db_floor:spend:key:abc")]
#[case("virtual_key_spend:abc:gpt:1d")]
#[case("spend:unknown:abc")]
#[case("spend:team_member:no-team")]
#[case::user_window_without_duration("spend:user_window:no-duration")]
#[case::user_window_without_user("spend:user_window::1h")]
#[case::committed_user_window("spend:committed:user_window:u:1h")]
#[case("spend:")]
#[case::committed_window("spend:committed:key:abc:window:1d")]
#[case::committed_tag("spend:committed:tag:prod")]
#[case::committed_team_member("spend:committed:team_member:user:team-a")]
#[case::committed_twice("spend:committed:committed:key:abc")]
#[case::committed_nothing("spend:committed:")]
fn keys_outside_the_spend_counter_keyspace_are_refused(#[case] key: &str) {
    assert!(SpendCounter::parse(key).is_err(), "{key}");
}

#[rstest]
#[tokio::test]
async fn reads_values_python_wrote_and_a_clean_miss_is_a_redis_answer(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let counters = (0..READS.len())
        .map(|index| team(&format!("t{index}")))
        .collect::<Vec<_>>();
    for ((bytes, _), counter) in READS.iter().zip(&counters) {
        raw.send(commands::set(&counter.key(), *bytes, TTL))
            .await
            .unwrap();
    }
    let expected = READS
        .iter()
        .map(|(_, spend)| Some(*spend))
        .collect::<Vec<_>>();

    for (counter, value) in counters.iter().zip(&expected) {
        assert_eq!(store.read(counter).await.unwrap(), Reading::Redis(*value));
    }
    assert_eq!(
        store
            .read_many(RedisSend::Now, &counters)
            .unwrap()
            .await
            .unwrap(),
        Some(expected)
    );
    assert_eq!(
        store.read(&team("absent")).await.unwrap(),
        Reading::Redis(None)
    );
}

#[rstest]
#[tokio::test]
async fn resets_and_repairs_write_mains_bytes_with_the_redis_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let counter = team("t");

    for (spend, bytes) in SETS {
        store.set(&counter, spend, None).await.unwrap();
        assert_eq!(
            raw.send(commands::get::<String>(&counter.key()))
                .await
                .unwrap(),
            Some(bytes.to_owned())
        );
    }
    for (spend, bytes) in SET_MAXES {
        raw.send(commands::del(&[&counter.key()])).await.unwrap();
        let repaired = store.set_max(&counter, spend).await.unwrap();
        assert_eq!(
            raw.send(commands::get::<String>(&counter.key()))
                .await
                .unwrap(),
            Some(bytes.to_owned())
        );
        assert_eq!(repaired, Some(Number::Float(spend.as_f64())));
    }
    store
        .set(&counter, Number::Int(5), Some(Duration::from_secs(30)))
        .await
        .unwrap();
    let Ttl::Expires(remaining) = raw.send(commands::ttl(&counter.key())).await.unwrap() else {
        panic!("a reset counter has no TTL");
    };
    assert!(within(remaining, 30), "{remaining:?}");
}

#[rstest]
#[tokio::test]
async fn set_max_only_ever_raises_a_counter(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let counter = SpendCounter::Key {
        token: "hashed".to_owned(),
        window: None,
    };
    store.increment(&counter, Number::Float(9.5)).await.unwrap();

    let lower = store.set_max(&counter, Number::Float(4.0)).await.unwrap();
    let higher = store.set_max(&counter, Number::Float(12.0)).await.unwrap();

    assert_eq!(lower, Some(Number::Float(9.5)));
    assert_eq!(higher, Some(Number::Float(12.0)));
    assert_eq!(
        raw.send(commands::get::<String>(&counter.key()))
            .await
            .unwrap(),
        Some("12.0".to_owned())
    );
    let Ttl::Expires(remaining) = raw.send(commands::ttl(&counter.key())).await.unwrap() else {
        panic!("a repaired counter has no TTL");
    };
    assert!(within(remaining, 60), "{remaining:?}");
    assert_eq!(
        store.get_local(&counter).unwrap(),
        Some(Number::Float(12.0))
    );
}

#[rstest]
#[tokio::test]
async fn increments_refresh_the_ttl_and_mirror_the_shared_total_into_memory(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = connect_observed(&deployment);
    let other_pod = store(Some(connect(&deployment)));
    let store = store(Some(client));
    let [first, second] = [team("a"), team("b")];
    other_pod
        .increment(&first, Number::Float(1.25))
        .await
        .unwrap();
    raw.send(commands::expire(&first.key(), Duration::from_secs(5)))
        .await
        .unwrap();

    let totals = store
        .increment_many(&[
            (first.clone(), Number::Float(0.5)),
            (second.clone(), Number::Int(2)),
        ])
        .await
        .unwrap();

    assert_eq!(totals, [Number::Float(1.75), Number::Float(2.0)]);
    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 2 }]);
    let Ttl::Expires(remaining) = raw.send(commands::ttl(&first.key())).await.unwrap() else {
        panic!("an incremented counter has no TTL");
    };
    assert!(within(remaining, 60), "{remaining:?}");
    assert_eq!(
        raw.send(commands::get::<String>(&first.key()))
            .await
            .unwrap(),
        Some("1.75".to_owned())
    );
    assert_eq!(store.get_local(&first).unwrap(), Some(Number::Float(1.75)));
    assert_eq!(store.get_local(&second).unwrap(), Some(Number::Float(2.0)));
}

#[rstest]
#[tokio::test]
async fn a_post_call_increment_from_a_thread_without_a_runtime_lands_in_redis_and_memory(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let counter = team("t");

    let declared = std::thread::spawn({
        let (store, request, counter) = (Arc::clone(&store), request.clone(), counter.clone());
        move || store.increment_many_post_call(&request, &[(counter, Number::Float(3.5))])
    })
    .join()
    .expect("a Python thread has no Tokio context, and declaring there must not panic");
    request.flush_post().await;

    assert!(declared.is_some());
    let mirrored =
        litellm_redis_testing::eventually(async || store.get_local(&counter).unwrap()).await;
    assert_eq!(mirrored, Number::Float(3.5));
    assert_eq!(
        raw.send(commands::get::<String>(&counter.key()))
            .await
            .unwrap(),
        Some("3.5".to_owned())
    );
}

#[rstest]
#[tokio::test]
async fn post_call_increments_wait_for_the_post_batch_then_land_in_memory(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = connect_observed(&deployment);
    let store = store(Some(client));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let counter = team("t");

    let settlement = store
        .increment_many_post_call(&request, &[(counter.clone(), Number::Float(3.5))])
        .unwrap();
    request.flush_pre().await;

    assert!(!settlement.is_settled());
    assert!(trips.seen().is_empty());
    assert_eq!(
        raw.send(commands::get::<String>(&counter.key()))
            .await
            .unwrap(),
        None
    );

    request.flush_post().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(settlement.is_settled());
    let increment = match topology {
        Topology::Standalone => Operation::Pipeline { ops: 1 },
        Topology::Cluster => Operation::Command("INCRBYFLOAT"),
    };
    assert_eq!(trips.seen(), [increment]);
    assert_eq!(
        raw.send(commands::get::<String>(&counter.key()))
            .await
            .unwrap(),
        Some("3.5".to_owned())
    );
    assert_eq!(store.get_local(&counter).unwrap(), Some(Number::Float(3.5)));
}

#[rstest]
#[tokio::test]
async fn the_budget_read_rides_the_same_pre_call_pipeline_as_the_cooldown_read(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = connect_observed(&deployment);
    let cooldowns = std::sync::Arc::new(RouterCooldowns::new(
        Some(remote(client.clone())),
        COOLDOWN_POLICY,
    ));
    let spend = store(Some(client));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let counters = [
        SpendCounter::Key {
            token: "hashed".to_owned(),
            window: None,
        },
        team("t"),
    ];
    raw.send(commands::set(&counters[0].key(), "4.5", TTL))
        .await
        .unwrap();

    let active = cooldowns
        .active(RedisSend::PreCall(&request), &["dep-a".to_owned()])
        .unwrap();
    let budget = spend
        .read_many(RedisSend::PreCall(&request), &counters)
        .unwrap();
    request.flush_pre().await;

    let expected = match topology {
        Topology::Standalone => vec![Operation::Pipeline { ops: 2 }],
        Topology::Cluster => vec![Operation::Command("MGET"); 2],
    };
    assert_eq!(trips.seen(), expected);
    assert_eq!(active.await.unwrap(), vec![]);
    assert_eq!(
        budget.await.unwrap(),
        Some(vec![Some(Number::Float(4.5)), None])
    );
    assert_eq!(trips.seen().len(), expected.len());
}

#[rstest]
#[tokio::test]
async fn a_seed_lets_the_first_pod_win_and_others_answer_with_its_value(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let first = store(Some(connect(&deployment)));
    let second = store(Some(connect(&deployment)));
    let counter = SpendCounter::Org("org".to_owned());

    let won = first
        .seed(&counter, Number::Float(10.0), LostSeed::Answer)
        .await
        .unwrap();
    let lost = second
        .seed(&counter, Number::Float(99.0), LostSeed::Increment)
        .await
        .unwrap();

    assert_eq!([won, lost], [Number::Float(10.0), Number::Float(10.0)]);
    assert_eq!(
        raw.send(commands::get::<String>(&counter.key()))
            .await
            .unwrap(),
        Some("10.0".to_owned())
    );
    let Ttl::Expires(remaining) = raw.send(commands::ttl(&counter.key())).await.unwrap() else {
        panic!("a seeded counter has no TTL");
    };
    assert!(within(remaining, 60), "{remaining:?}");
    assert_eq!(
        second.get_local(&counter).unwrap(),
        Some(Number::Float(10.0))
    );
}

#[rstest]
#[tokio::test]
async fn delete_drops_memory_and_redis_and_refresh_ttl_reports_whether_redis_held_it(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let counters = [team("a"), team("b")];
    for counter in &counters {
        store.increment(counter, Number::Int(1)).await.unwrap();
    }
    raw.send(commands::expire(&counters[0].key(), Duration::from_secs(5)))
        .await
        .unwrap();

    assert!(store.refresh_ttl(&counters[0]).await.unwrap());
    let Ttl::Expires(remaining) = raw.send(commands::ttl(&counters[0].key())).await.unwrap() else {
        panic!("a refreshed counter has no TTL");
    };
    assert!(within(remaining, 60), "{remaining:?}");

    store.delete(&counters).await.unwrap();

    for counter in &counters {
        assert_eq!(
            raw.send(commands::get::<String>(&counter.key()))
                .await
                .unwrap(),
            None
        );
        assert_eq!(store.get_local(counter).unwrap(), None);
    }
    assert!(!store.refresh_ttl(&counters[0]).await.unwrap());
}

// main's read_spend_counter_cache_value: RedisCache.async_get_cache swallows a failed GET and
// returns None, an authoritative miss. Only the breaker guard's refusal raises into the fallback
// that reads this pod's memory
#[rstest]
#[tokio::test]
async fn a_failed_read_serves_the_local_counter_and_a_failed_increment_drops_the_counter() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = store(Some(connect(&deployment)));
    let [kept, failed] = [team("kept"), team("failed")];
    store.increment(&kept, Number::Float(2.0)).await.unwrap();
    store.increment(&failed, Number::Float(3.0)).await.unwrap();
    deployment.stop().await.unwrap();

    let read = store.read(&kept).await.unwrap();
    let increment = store.increment(&failed, Number::Float(1.0)).await;

    assert_eq!(read, Reading::Memory(Some(Number::Float(2.0))));
    assert!(increment.is_err());
    assert_eq!(store.get_local(&failed).unwrap(), None);
    assert_eq!(
        store
            .read_many(RedisSend::Now, &[kept])
            .unwrap()
            .await
            .unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn a_read_with_an_open_breaker_serves_the_local_counter() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let breaker = litellm_circuit_breaker::Breaker::new(
        litellm_circuit_breaker::Config {
            failure_threshold: 1,
            recovery_timeout: Duration::from_secs(60),
            ..litellm_circuit_breaker::Config::default()
        },
        Box::new(|_| {}),
    );
    let store = store(Some(support::connect_guarded(&deployment, breaker)));
    let counter = team("t");
    store.increment(&counter, Number::Float(2.0)).await.unwrap();
    deployment.stop().await.unwrap();

    let failed = store.read(&counter).await.unwrap();
    let refused = store.read(&counter).await.unwrap();

    assert_eq!(failed, Reading::Memory(Some(Number::Float(2.0))));
    assert_eq!(refused, Reading::Memory(Some(Number::Float(2.0))));
}

// increment_in_memory calls async_increment_cache(local_only=True), which hands InMemoryCache no
// TTL, so the entry takes InMemoryCache's own default rather than spend_counter_cache's 60 s
#[test]
fn a_local_increment_outlives_the_default_in_memory_ttl() {
    let seconds = Arc::new(std::sync::atomic::AtomicU64::new(1_000));
    let now = {
        let seconds = seconds.clone();
        Arc::new(move || Duration::from_secs(seconds.load(std::sync::atomic::Ordering::SeqCst)))
    };
    let store = SpendCounters::with_clock(None, now);
    let counter = team("t");

    store.increment_local(&counter, Number::Int(1)).unwrap();
    seconds.fetch_add(61, std::sync::atomic::Ordering::SeqCst);

    assert_eq!(store.get_local(&counter).unwrap(), Some(Number::Int(1)));
}

#[rstest]
#[tokio::test]
async fn a_failed_post_call_increment_drops_the_counter_from_memory() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = store(Some(connect(&deployment)));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let counter = team("t");
    store.increment(&counter, Number::Float(1.0)).await.unwrap();
    let settlement = store
        .increment_many_post_call(&request, &[(counter.clone(), Number::Float(1.0))])
        .unwrap();
    deployment.stop().await.unwrap();

    request.flush_post().await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert_eq!(store.get_local(&counter).unwrap(), None);
    assert!(settlement.is_settled());
}

#[rstest]
#[tokio::test]
async fn a_post_call_settlement_settles_when_the_deadline_sends_the_batch(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let request = RequestRedis::new(Handle::current(), Duration::from_millis(50));
    let counter = committed(team("t"));

    let settlement = store
        .increment_many_post_call(&request, &[(counter.clone(), Number::Float(2.5))])
        .unwrap();

    assert!(!settlement.is_settled());
    litellm_redis_testing::eventually(async || settlement.is_settled().then_some(())).await;
    assert_eq!(
        raw.send(commands::get::<String>(&counter.key()))
            .await
            .unwrap(),
        Some("2.5".to_owned())
    );
}

#[rstest]
#[case::committed_twin(committed, 300)]
#[case::enforcement_counter(std::convert::identity, 60)]
#[tokio::test]
async fn every_write_gives_a_committed_twin_the_committed_ttl_and_other_counters_the_redis_ttl(
    #[case] counter_of: fn(SpendCounter) -> SpendCounter,
    #[case] ttl_seconds: u64,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    store.set_committed_ttl(COMMITTED_TTL);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let [
        incremented,
        pipelined,
        post_call,
        seeded,
        repaired,
        refreshed,
    ] = ["a", "b", "c", "d", "e", "f"].map(|id| counter_of(team(id)));
    raw.send(commands::set(
        &refreshed.key(),
        "1.0",
        Duration::from_secs(5),
    ))
    .await
    .unwrap();

    store
        .increment(&incremented, Number::Float(1.0))
        .await
        .unwrap();
    store
        .increment_many(&[(pipelined.clone(), Number::Float(1.0))])
        .await
        .unwrap();
    let settlement = store
        .increment_many_post_call(&request, &[(post_call.clone(), Number::Float(1.0))])
        .unwrap();
    request.flush_post().await;
    store
        .seed(&seeded, Number::Float(1.0), LostSeed::Answer)
        .await
        .unwrap();
    store.set_max(&repaired, Number::Float(1.0)).await.unwrap();
    assert!(store.refresh_ttl(&refreshed).await.unwrap());

    assert!(settlement.is_settled());
    let remaining =
        async |counter: &SpendCounter| match raw.send(commands::ttl(&counter.key())).await.unwrap()
        {
            Ttl::Expires(remaining) => remaining,
            other => panic!("{} has no TTL: {other:?}", counter.key()),
        };
    assert!(within(remaining(&incremented).await, ttl_seconds));
    assert!(within(remaining(&pipelined).await, ttl_seconds));
    assert!(within(remaining(&post_call).await, ttl_seconds));
    assert!(within(remaining(&seeded).await, ttl_seconds));
    assert!(within(remaining(&repaired).await, ttl_seconds));
    assert!(within(remaining(&refreshed).await, ttl_seconds));
}

#[rstest]
#[tokio::test]
async fn a_post_call_charge_moves_a_committed_twins_expiry_out_to_the_committed_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    store.set_committed_ttl(COMMITTED_TTL);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let counter = committed(team("t"));
    raw.send(commands::set(&counter.key(), "1.0", Duration::from_secs(5)))
        .await
        .unwrap();

    let settlement = store
        .increment_many_post_call(&request, &[(counter.clone(), Number::Float(1.0))])
        .unwrap();
    request.flush_post().await;

    assert!(settlement.is_settled());
    match raw.send(commands::ttl(&counter.key())).await.unwrap() {
        Ttl::Expires(remaining) => assert!(within(remaining, COMMITTED_TTL.as_secs())),
        other => panic!("{} has no TTL: {other:?}", counter.key()),
    }
}

#[tokio::test]
async fn without_redis_counters_live_in_memory() {
    let store = store(None);
    let counter = team("t");
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);

    let totals = store
        .increment_many(&[
            (counter.clone(), Number::Int(2)),
            (counter.clone(), Number::Float(0.5)),
        ])
        .await
        .unwrap();
    let seeded = store
        .seed(&counter, Number::Float(1.0), LostSeed::Answer)
        .await
        .unwrap();
    let raised = store.set_max(&counter, Number::Float(7.0)).await.unwrap();

    assert_eq!(totals, [Number::Int(2), Number::Float(2.5)]);
    assert_eq!(seeded, Number::Float(2.5));
    assert_eq!(raised, None);
    assert_eq!(
        store.read(&counter).await.unwrap(),
        Reading::Memory(Some(Number::Float(7.0)))
    );
    assert!(
        store
            .increment_many_post_call(&request, &[(counter.clone(), Number::Int(1))])
            .is_none()
    );
    assert!(
        store
            .read_many(RedisSend::PreCall(&request), &[counter])
            .is_none()
    );
}

// main's RedisCache: async_increment(refresh_ttl=True), MGET for a batched read, async_set_max as
// a plain EVAL, SET EX NX for a seed and one multi-key DEL per chunk
#[rstest]
#[tokio::test]
async fn spend_counters_send_the_commands_main_sends(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let store = store(Some(connect(&deployment)));
    let counters = [team("a"), team("b")];
    deployment
        .cli_on_every_node(&["CONFIG", "RESETSTAT"])
        .await
        .unwrap();

    store
        .increment(&counters[0], Number::Float(1.5))
        .await
        .unwrap();
    store
        .read_many(RedisSend::Now, &counters)
        .unwrap()
        .await
        .unwrap()
        .unwrap();
    store
        .set_max(&counters[0], Number::Float(9.0))
        .await
        .unwrap();
    store
        .seed(&counters[1], Number::Float(2.0), LostSeed::Answer)
        .await
        .unwrap();
    store.delete(&counters).await.unwrap();

    // commandstats also counts the GET, SET and EXPIRE the set_max script runs inside Redis
    let calls = async |command| support::calls(&deployment, command).await;
    assert_eq!(calls("incrbyfloat").await, 1);
    assert_eq!(calls("expire").await, 2);
    assert_eq!(calls("eval").await, 1);
    assert_eq!(calls("evalsha").await, 0);
    assert_eq!(calls("set").await, 2);
    assert!(calls("mget").await > 0);
    assert_eq!(calls("get").await, 1);
    assert!(calls("del").await > 0);
}
