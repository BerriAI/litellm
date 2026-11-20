mod support;

use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_redis_client::{Operation, Ttl, commands};
use litellm_redis_store::{
    Number, RouterStores, RouterTiers,
    deployment_health::{DeploymentHealth, HealthState as WrittenHealth, HealthStates},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use litellm_router_core::{
    HealthState, Outcomes, RouterCache, UsageCounts, UsageWindow, health::CooldownValue,
};
use rstest::rstest;
use support::{
    ALLOWED_FAILS, COOLDOWN_POLICY, USAGE_POLICY, connect, connect_observed, cooldowns, remote,
    within,
};

const TIERS: RouterTiers = RouterTiers {
    cooldowns: COOLDOWN_POLICY,
    usage: USAGE_POLICY,
    default_in_memory_ttl: None,
};

const TPM_KEY: &str = "global_router:dep:anthropic/claude-sonnet-5-5:tpm:12-34";
const RPM_KEY: &str = "global_router:dep:anthropic/claude-sonnet-5-5:rpm:12-34";

fn router(deployment: &RedisDeployment) -> RouterStores {
    RouterStores::new(Some(remote(connect(deployment))), TIERS)
}

fn window() -> UsageWindow {
    UsageWindow {
        deployment_id: "dep".to_owned(),
        model: "anthropic/claude-sonnet-5-5".to_owned(),
        minute: "12-34".to_owned(),
    }
}

fn cooling_down_now() -> CooldownValue {
    CooldownValue {
        exception_received: "RateLimitError".to_owned(),
        status_code: "429".to_owned(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64(),
        cooldown_time: 30.0,
    }
}

fn unhealthy(id: &str) -> HealthStates {
    HealthStates(vec![(
        id.to_owned(),
        WrittenHealth {
            is_healthy: false,
            timestamp: Number::Float(1.5),
            reason: "timeout".to_owned(),
        },
    )])
}

fn read_unhealthy(id: &str) -> HashMap<String, HealthState> {
    HashMap::from([(
        id.to_owned(),
        HealthState {
            is_healthy: false,
            timestamp: 1.5,
        },
    )])
}

#[rstest]
#[case::whole_seconds_as_an_int(0)]
#[case::fractional_seconds_as_a_float(1)]
#[tokio::test]
async fn a_cooldown_is_stored_as_the_json_main_writes(
    #[case] case: usize,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let case = cooldowns().into_iter().nth(case).unwrap();
    let cooldown = CooldownValue {
        exception_received: case.cooldown.exception_received.clone(),
        status_code: case.cooldown.status_code.clone(),
        timestamp: case.cooldown.timestamp.as_f64(),
        cooldown_time: case.cooldown.cooldown_time.as_f64(),
    };

    router(&deployment)
        .put_cooldown(
            case.model_id,
            &cooldown,
            Duration::from_secs_f64(case.ttl_seconds),
        )
        .await
        .unwrap();

    let written: Option<String> = raw.send(commands::get(case.key)).await.unwrap();
    let Ttl::Expires(remaining) = raw.send(commands::ttl(case.key)).await.unwrap() else {
        panic!("{} has no TTL", case.key);
    };
    assert_eq!(written.as_deref(), Some(case.json));
    assert!(within(remaining, case.redis_ttl_seconds), "{remaining:?}");
}

#[rstest]
#[tokio::test]
async fn another_router_reads_a_cooldown_in_the_order_it_asked(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (first, second) = (router(&deployment), router(&deployment));
    let cooldown = cooling_down_now();

    first
        .put_cooldown("dep-a", &cooldown, Duration::from_secs(30))
        .await
        .unwrap();

    assert_eq!(
        second
            .cooldowns(&["dep-b".to_owned(), "dep-a".to_owned()])
            .await
            .unwrap(),
        [None, Some(cooldown)]
    );
}

#[rstest]
#[tokio::test]
async fn allowed_fails_count_across_routers_under_mains_key(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = connect_observed(&deployment);
    let first = RouterStores::new(Some(remote(client)), TIERS);
    let second = router(&deployment);
    let [_, _, case] = ALLOWED_FAILS;
    let ttl = Duration::from_secs(30);

    let counts = (
        first
            .increment_allowed_fails("dep-a", case.suffix, ttl)
            .await
            .unwrap(),
        second
            .increment_allowed_fails("dep-a", case.suffix, ttl)
            .await
            .unwrap(),
    );

    let stored: Option<String> = raw.send(commands::get(case.key)).await.unwrap();
    assert_eq!(counts, (1.0, 2.0));
    assert_eq!(stored.as_deref(), Some("2"));
    assert_eq!(
        trips.seen().first(),
        Some(&Operation::Command("INCRBYFLOAT"))
    );
}

#[rstest]
#[tokio::test]
async fn usage_one_router_adds_is_read_by_another_from_the_global_router_counters(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (first, second) = (router(&deployment), router(&deployment));

    first.add_usage(&window(), 7.5, 1.0).await.unwrap();
    first.add_request(&window()).await.unwrap();

    let stored: [Option<String>; 2] = [
        raw.send(commands::get(TPM_KEY)).await.unwrap(),
        raw.send(commands::get(RPM_KEY)).await.unwrap(),
    ];
    assert_eq!(stored, [Some("7.5".to_owned()), Some("2".to_owned())]);
    assert_eq!(
        second.usage(&window()).await.unwrap(),
        UsageCounts {
            tokens: Some(7.5),
            requests: Some(2.0),
        }
    );
}

#[rstest]
#[tokio::test]
async fn health_states_the_health_checks_wrote_are_read_without_their_reason(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    DeploymentHealth::new(Some(remote(connect(&deployment))), None)
        .set(unhealthy("dep-a"), Duration::from_secs(60))
        .await
        .unwrap();

    assert_eq!(
        router(&deployment).health_states().await.unwrap(),
        read_unhealthy("dep-a")
    );
}

#[rstest]
#[tokio::test]
async fn success_and_failure_counts_stay_in_the_router_that_counted_them(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (first, second) = (router(&deployment), router(&deployment));

    first.count_success("dep-a").await.unwrap();
    first.count_failure("dep-a").await.unwrap();
    first.count_failure("dep-a").await.unwrap();

    assert_eq!(
        (
            first.outcomes("dep-a").await.unwrap(),
            second.outcomes("dep-a").await.unwrap(),
        ),
        (
            Outcomes {
                successes: 1,
                fails: 2,
            },
            Outcomes::default(),
        )
    );
}

#[rstest]
#[tokio::test]
async fn a_router_without_redis_answers_from_memory() {
    let router = RouterStores::new(None, TIERS);
    let cooldown = cooling_down_now();
    let ttl = Duration::from_secs(30);

    router.put_cooldown("dep-a", &cooldown, ttl).await.unwrap();
    let fails = (
        router
            .increment_allowed_fails("dep-a", None, ttl)
            .await
            .unwrap(),
        router
            .increment_allowed_fails("dep-a", None, ttl)
            .await
            .unwrap(),
    );
    router.add_usage(&window(), 7.5, 1.0).await.unwrap();
    router.add_request(&window()).await.unwrap();

    assert_eq!(
        router.cooldowns(&["dep-a".to_owned()]).await.unwrap(),
        [Some(cooldown)]
    );
    assert_eq!(fails, (1.0, 2.0));
    assert_eq!(
        router.usage(&window()).await.unwrap(),
        UsageCounts {
            tokens: Some(7.5),
            requests: Some(2.0),
        }
    );
    assert_eq!(router.health_states().await.unwrap(), HashMap::new());
}

#[rstest]
#[tokio::test]
async fn attaching_redis_shares_every_store_and_happens_once(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let attached = RouterStores::new(None, TIERS);
    let other = router(&deployment);
    let cooldown = cooling_down_now();

    let attaches = (
        attached.attach(remote(connect(&deployment))),
        attached.attach(remote(connect(&deployment))),
    );
    attached
        .put_cooldown("dep-a", &cooldown, Duration::from_secs(30))
        .await
        .unwrap();
    attached.add_usage(&window(), 7.5, 1.0).await.unwrap();
    DeploymentHealth::new(Some(remote(connect(&deployment))), None)
        .set(unhealthy("dep-a"), Duration::from_secs(60))
        .await
        .unwrap();

    assert_eq!(attaches, (true, false));
    assert_eq!(
        other.cooldowns(&["dep-a".to_owned()]).await.unwrap(),
        [Some(cooldown)]
    );
    assert_eq!(
        other.usage(&window()).await.unwrap(),
        UsageCounts {
            tokens: Some(7.5),
            requests: Some(1.0),
        }
    );
    assert_eq!(
        attached.health_states().await.unwrap(),
        read_unhealthy("dep-a")
    );
}
