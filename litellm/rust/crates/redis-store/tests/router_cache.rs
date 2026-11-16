mod support;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_redis_client::commands;
use litellm_redis_store::{RouterStores, RouterTiers};
use litellm_redis_testing::{RedisDeployment, Topology};
use litellm_router_core::{
    LimitCounter, LimitKind, RouterCache, UsageCounts, UsageWindow, health::CooldownValue,
};
use rstest::rstest;
use support::{COOLDOWN_POLICY, USAGE_POLICY, connect, cooldowns, remote};

const TIERS: RouterTiers = RouterTiers {
    cooldowns: COOLDOWN_POLICY,
    usage: USAGE_POLICY,
    default_in_memory_ttl: None,
};

/// Two workers' stores on one Redis.
fn workers(deployment: &RedisDeployment) -> (RouterStores, RouterStores) {
    let worker = || RouterStores::new(Some(remote(connect(deployment))), TIERS);
    (worker(), worker())
}

fn window() -> UsageWindow {
    UsageWindow {
        deployment_id: "dep".to_owned(),
        model: "anthropic/claude-sonnet-5-5".to_owned(),
        minute: "12-34".to_owned(),
    }
}

#[rstest]
#[tokio::test]
async fn a_cooldown_router_core_writes_is_stored_as_mains_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (stores, _) = workers(&deployment);

    for case in cooldowns() {
        let cooldown = CooldownValue {
            exception_received: case.cooldown.exception_received.clone(),
            status_code: case.cooldown.status_code.clone(),
            timestamp: case.cooldown.timestamp.as_f64(),
            cooldown_time: case.cooldown.cooldown_time.as_f64(),
        };
        stores
            .put_cooldown(
                case.model_id,
                &cooldown,
                Duration::from_secs_f64(case.ttl_seconds),
            )
            .await
            .unwrap();

        let written: Option<String> = raw.send(commands::get(case.key)).await.unwrap();
        assert_eq!(written.as_deref(), Some(case.json));
    }
}

#[rstest]
#[tokio::test]
async fn another_worker_sees_a_cooldown_and_the_shared_fail_count(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (first, second) = workers(&deployment);
    let cooldown = CooldownValue {
        exception_received: "RateLimitError".to_owned(),
        status_code: "429".to_owned(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64(),
        cooldown_time: 30.0,
    };

    first
        .put_cooldown("dep-a", &cooldown, Duration::from_secs(30))
        .await
        .unwrap();
    let counts = (
        first
            .increment_allowed_fails("dep-a", None, Duration::from_secs(30))
            .await
            .unwrap(),
        second
            .increment_allowed_fails("dep-a", None, Duration::from_secs(30))
            .await
            .unwrap(),
    );

    assert_eq!(
        second
            .cooldowns(&["dep-b".to_owned(), "dep-a".to_owned()])
            .await
            .unwrap(),
        [None, Some(cooldown)]
    );
    assert_eq!(counts, (1.0, 2.0));
}

#[rstest]
#[tokio::test]
async fn usage_one_worker_adds_is_read_by_another(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (first, second) = workers(&deployment);

    first.add_usage(&window(), 7.5, 1.0).await.unwrap();
    first.add_request(&window()).await.unwrap();

    let rpm: Option<String> = raw
        .send(commands::get(
            "global_router:dep:anthropic/claude-sonnet-5-5:rpm:12-34",
        ))
        .await
        .unwrap();
    assert_eq!(rpm.as_deref(), Some("2"));
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
async fn success_and_failure_counts_stay_in_the_worker_that_saw_them() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let (first, second) = workers(&deployment);

    first.count_success("dep-a").await.unwrap();
    first.count_failure("dep-a").await.unwrap();

    let seen = first.outcomes("dep-a").await.unwrap();
    let unseen = second.outcomes("dep-a").await.unwrap();
    assert_eq!(
        (
            (seen.successes, seen.fails),
            (unseen.successes, unseen.fails)
        ),
        ((1, 1), (0, 0))
    );
}

fn limit(kind: LimitKind, window: &str) -> LimitCounter {
    LimitCounter {
        deployment_id: "dep".to_owned(),
        model: "azure/gpt-5.5".to_owned(),
        kind,
        window: window.to_owned(),
    }
}

#[rstest]
#[tokio::test]
async fn model_rate_limit_counters_are_shared_under_mains_keys(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (first, second) = workers(&deployment);
    let charged = [
        (limit(LimitKind::Rpm, "12-34"), 1, Duration::from_secs(60)),
        (
            limit(LimitKind::Rpd, "2026-11-15"),
            1,
            Duration::from_secs(86_400),
        ),
    ];

    let counted = (
        first.add_limit_usages(&charged).await.unwrap(),
        second.add_limit_usages(&charged).await.unwrap(),
    );
    second
        .add_limit_usage(&limit(LimitKind::Tpm, "12-34"), 7, Duration::from_secs(60))
        .await
        .unwrap();

    let rpd: Option<String> = raw
        .send(commands::get("dep:azure/gpt-5.5:rpd:2026-11-15"))
        .await
        .unwrap();
    assert_eq!(
        (
            counted,
            rpd.as_deref(),
            first
                .limit_usage(&limit(LimitKind::Tpm, "12-34"), 100.0)
                .await
                .unwrap(),
        ),
        (
            (Some(vec![1.0, 1.0]), Some(vec![2.0, 2.0])),
            Some("2"),
            Some(7.0)
        )
    );
}
