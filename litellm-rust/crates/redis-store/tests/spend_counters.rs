mod support;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::{Operation, RedisClient, Ttl};
use litellm_redis_store::{
    Number, RequestRedis,
    router_cooldowns::{self, RouterCooldowns},
    spend_counters::{LostSeed, Reading, SpendCounter, SpendCounters},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, connect_observed, within};
use tokio::runtime::Handle;

const TTL: Duration = Duration::from_secs(60);
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

fn store(client: Option<RedisClient>) -> Arc<SpendCounters> {
    Arc::new(SpendCounters::new(client, TTL))
}

/// Mirrored by `tests/unit/redis/stores/test_spend_counters.py`, which checks the same keys and
/// bytes against Python's own key builders and `RedisCache` codec.
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
#[case(SpendCounter::EndUser("customer é 1".to_owned()), "spend:end_user:customer é 1")]
#[case(SpendCounter::Org("org-1".to_owned()), "spend:org:org-1")]
#[case(SpendCounter::Tag("prod:eu".to_owned()), "spend:tag:prod:eu")]
#[case(SpendCounter::Project("proj-1".to_owned()), "spend:project:proj-1")]
#[case(
    SpendCounter::ModelAccessGroup("gpt-group".to_owned()),
    "spend:model_access_group:gpt-group"
)]
#[case(SpendCounter::ShadowEval("job-7".to_owned()), "spend:shadow_eval:job-7")]
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
#[case("spend:")]
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
        raw.set(&counter.key(), *bytes, TTL).await.unwrap();
    }
    let expected = READS
        .iter()
        .map(|(_, spend)| Some(*spend))
        .collect::<Vec<_>>();

    for (counter, value) in counters.iter().zip(&expected) {
        assert_eq!(store.read(counter).await.unwrap(), Reading::Redis(*value));
    }
    assert_eq!(store.read_many(&counters).await, Some(expected));
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
            raw.get::<String>(&counter.key()).await.unwrap(),
            Some(bytes.to_owned())
        );
    }
    for (spend, bytes) in SET_MAXES {
        raw.del(&[&counter.key()]).await.unwrap();
        let repaired = store.set_max(&counter, spend).await.unwrap();
        assert_eq!(
            raw.get::<String>(&counter.key()).await.unwrap(),
            Some(bytes.to_owned())
        );
        assert_eq!(repaired, Some(Number::Float(spend.as_f64())));
    }
    store
        .set(&counter, Number::Int(5), Some(Duration::from_secs(30)))
        .await
        .unwrap();
    let Ttl::Expires(remaining) = raw.ttl(&counter.key()).await.unwrap() else {
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
        raw.get::<String>(&counter.key()).await.unwrap(),
        Some("12.0".to_owned())
    );
    let Ttl::Expires(remaining) = raw.ttl(&counter.key()).await.unwrap() else {
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
    raw.expire(&first.key(), Duration::from_secs(5))
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
    let Ttl::Expires(remaining) = raw.ttl(&first.key()).await.unwrap() else {
        panic!("an incremented counter has no TTL");
    };
    assert!(within(remaining, 60), "{remaining:?}");
    assert_eq!(
        raw.get::<String>(&first.key()).await.unwrap(),
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

    assert!(declared);
    let mirrored =
        litellm_redis_testing::eventually(async || store.get_local(&counter).unwrap()).await;
    assert_eq!(mirrored, Number::Float(3.5));
    assert_eq!(
        raw.get::<String>(&counter.key()).await.unwrap(),
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

    assert!(store.increment_many_post_call(&request, &[(counter.clone(), Number::Float(3.5))]));
    request.flush_pre().await;

    assert!(trips.seen().is_empty());
    assert_eq!(raw.get::<String>(&counter.key()).await.unwrap(), None);

    request.flush_post().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let increment = match topology {
        Topology::Standalone => Operation::Pipeline { ops: 1 },
        Topology::Cluster => Operation::Command("INCRBYFLOAT"),
    };
    assert_eq!(trips.seen(), [increment]);
    assert_eq!(
        raw.get::<String>(&counter.key()).await.unwrap(),
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
    let cooldowns = RouterCooldowns::new(Some(client.clone()), router_cooldowns::POLICY);
    let spend = store(Some(client));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let counters = [
        SpendCounter::Key {
            token: "hashed".to_owned(),
            window: None,
        },
        team("t"),
    ];
    raw.set(&counters[0].key(), "4.5", TTL).await.unwrap();

    let active = cooldowns
        .declare_active(&request, &["dep-a".to_owned()])
        .unwrap();
    let budget = spend
        .declare_read_many_pre_call(&request, &counters)
        .unwrap();
    request.flush_pre().await;

    let expected = match topology {
        Topology::Standalone => vec![Operation::Pipeline { ops: 2 }],
        Topology::Cluster => vec![Operation::Command("MGET"); 2],
    };
    assert_eq!(trips.seen(), expected);
    assert_eq!(cooldowns.resolve_active(active).await.unwrap(), vec![]);
    assert_eq!(
        budget.resolve().await,
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
        raw.get::<String>(&counter.key()).await.unwrap(),
        Some("10.0".to_owned())
    );
    let Ttl::Expires(remaining) = raw.ttl(&counter.key()).await.unwrap() else {
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
    raw.expire(&counters[0].key(), Duration::from_secs(5))
        .await
        .unwrap();

    assert!(store.refresh_ttl(&counters[0]).await.unwrap());
    let Ttl::Expires(remaining) = raw.ttl(&counters[0].key()).await.unwrap() else {
        panic!("a refreshed counter has no TTL");
    };
    assert!(within(remaining, 60), "{remaining:?}");

    store.delete(&counters).await.unwrap();

    for counter in &counters {
        assert_eq!(raw.get::<String>(&counter.key()).await.unwrap(), None);
        assert_eq!(store.get_local(counter).unwrap(), None);
    }
    assert!(!store.refresh_ttl(&counters[0]).await.unwrap());
}

#[rstest]
#[tokio::test]
async fn a_redis_outage_reads_memory_and_a_failed_increment_drops_the_counter() {
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
    assert_eq!(store.read_many(&[kept]).await, None);
}

#[rstest]
#[tokio::test]
async fn a_failed_post_call_increment_drops_the_counter_from_memory() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = store(Some(connect(&deployment)));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let counter = team("t");
    store.increment(&counter, Number::Float(1.0)).await.unwrap();
    assert!(store.increment_many_post_call(&request, &[(counter.clone(), Number::Float(1.0))]));
    deployment.stop().await.unwrap();

    request.flush_post().await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert_eq!(store.get_local(&counter).unwrap(), None);
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
    assert!(!store.increment_many_post_call(&request, &[(counter.clone(), Number::Int(1))]));
    assert!(
        store
            .declare_read_many_pre_call(&request, &[counter])
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
    store.read_many(&counters).await.unwrap();
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
