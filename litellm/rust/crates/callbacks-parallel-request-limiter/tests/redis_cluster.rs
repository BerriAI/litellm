use std::{sync::Arc, time::Duration};

use litellm_callbacks_parallel_request_limiter::{
    Code, Ctx, Descriptor, Engine, Policy, Settings, SlotAcquisition,
};
use litellm_redis_client::{
    Cluster, Credentials, Namespace, Node, RedisClient, RedisConfig, Target, Timeouts,
};
use litellm_redis_store::{
    Remote, RequestRedis,
    rate_limits::{RateLimits, Scope},
};
use litellm_redis_testing::RedisDeployment;
use litellm_redis_testing::Topology;
use rstest::{fixture, rstest};
use serde_json::json;
use tokio::runtime::Handle;

const TEAM: &str = "team-a";

fn pod(deployment: &RedisDeployment) -> Engine {
    let RedisDeployment::Cluster(cluster) = deployment else {
        unreachable!("these tests run against a cluster")
    };
    let client = RedisClient::new(RedisConfig {
        target: Target::Cluster(Cluster {
            nodes: cluster
                .nodes()
                .map(|(host, port)| Node {
                    host: host.to_owned(),
                    port,
                })
                .collect(),
            tls: None,
        }),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(500), None),
        breaker: None,
        events: None,
        health_check: None,
        keepalive: true,
    })
    .unwrap();
    let store = Arc::new(RateLimits::new(
        Some(Remote {
            client,
            default_ttl: Duration::from_secs(60),
        }),
        None,
    ));
    Engine::new(
        store,
        Settings {
            window_size: 60,
            tpm_reservation_enabled: false,
        },
        Arc::new(|| 1_700_000_000.5),
    )
}

#[fixture]
async fn cluster() -> RedisDeployment {
    RedisDeployment::start(Topology::Cluster).await.unwrap()
}

fn key_and_team(key: &str, key_limit: i64, team_limit: i64) -> Vec<Descriptor> {
    serde_json::from_value(json!([
        {"key": "api_key", "value": key, "rate_limit": {"max_parallel_requests": key_limit}},
        {"key": "team", "value": TEAM, "rate_limit": {"max_parallel_requests": team_limit}},
    ]))
    .unwrap()
}

fn scopes(key: &str) -> Vec<Scope> {
    [("api_key", key), ("team", TEAM)]
        .map(|(descriptor, value)| Scope {
            descriptor: descriptor.into(),
            value: value.into(),
        })
        .to_vec()
}

fn ctx(fail_closed: bool) -> Ctx<'static> {
    Ctx {
        policy: Policy {
            fail_closed,
            force_grouping: false,
        },
        request: None,
    }
}

async fn acquire(
    engine: &Engine,
    descriptors: &[Descriptor],
    slot: &str,
    fail_closed: bool,
) -> Code {
    engine
        .should_rate_limit(
            descriptors,
            false,
            true,
            Some(slot),
            false,
            ctx(fail_closed),
        )
        .await
        .expect("the gauges are verifiable against a healthy cluster")
        .overall
}

#[rstest]
#[tokio::test]
async fn the_key_and_team_gauges_land_in_different_cluster_slots(
    #[future(awt)] cluster: RedisDeployment,
) {
    assert_eq!(
        pod(&cluster)
            .store()
            .slot_groups(&scopes("key-1"), false)
            .len(),
        2
    );
}

#[rstest]
#[case::fail_open(false)]
#[case::fail_closed(true)]
#[tokio::test]
async fn a_team_slot_held_on_one_pod_refuses_another_key_of_the_team_on_another_pod(
    #[future(awt)] cluster: RedisDeployment,
    #[case] fail_closed: bool,
) {
    let (first, second) = (pod(&cluster), pod(&cluster));

    assert_eq!(
        acquire(&first, &key_and_team("key-1", 5, 1), "slot-1", fail_closed).await,
        Code::Ok
    );
    assert_eq!(
        acquire(&second, &key_and_team("key-2", 5, 1), "slot-2", fail_closed).await,
        Code::OverLimit
    );
}

#[rstest]
#[tokio::test]
async fn releasing_the_team_slot_on_one_pod_admits_the_team_on_another(
    #[future(awt)] cluster: RedisDeployment,
) {
    let (first, second) = (pod(&cluster), pod(&cluster));
    assert_eq!(
        acquire(&first, &key_and_team("key-1", 5, 1), "slot-1", false).await,
        Code::Ok
    );

    first
        .release_slots(
            &SlotAcquisition {
                slot_id: "slot-1".into(),
                gauges: scopes("key-1"),
            },
            ctx(false).policy,
        )
        .await;

    assert_eq!(
        acquire(&second, &key_and_team("key-2", 5, 1), "slot-2", false).await,
        Code::Ok
    );
}

#[rstest]
#[tokio::test]
async fn a_refusal_by_the_team_gives_back_the_key_slot_it_already_took(
    #[future(awt)] cluster: RedisDeployment,
) {
    let holder = pod(&cluster);
    assert_eq!(
        acquire(&holder, &key_and_team("key-1", 5, 1), "slot-1", false).await,
        Code::Ok
    );
    assert_eq!(
        acquire(
            &pod(&cluster),
            &key_and_team("key-2", 1, 1),
            "slot-2",
            false
        )
        .await,
        Code::OverLimit
    );

    holder
        .release_slots(
            &SlotAcquisition {
                slot_id: "slot-1".into(),
                gauges: scopes("key-1"),
            },
            ctx(false).policy,
        )
        .await;

    assert_eq!(
        acquire(
            &pod(&cluster),
            &key_and_team("key-2", 1, 1),
            "slot-3",
            false
        )
        .await,
        Code::Ok
    );
}

#[rstest]
#[tokio::test]
async fn a_deferred_release_frees_the_team_slot_for_another_pod(
    #[future(awt)] cluster: RedisDeployment,
) {
    let (first, second) = (pod(&cluster), pod(&cluster));
    assert_eq!(
        acquire(&first, &key_and_team("key-1", 5, 1), "slot-1", false).await,
        Code::Ok
    );
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    let deferred = first
        .defer_release(
            &SlotAcquisition {
                slot_id: "slot-1".into(),
                gauges: scopes("key-1"),
            },
            Ctx {
                request: Some(&request),
                ..ctx(false)
            },
        )
        .await;
    request.flush_post().await;

    assert!(deferred);
    let mut attempt = 0;
    litellm_redis_testing::eventually(async || {
        attempt += 1;
        let slot = format!("slot-retry-{attempt}");
        (acquire(&second, &key_and_team("key-2", 5, 1), &slot, false).await == Code::Ok)
            .then_some(())
    })
    .await;
}
