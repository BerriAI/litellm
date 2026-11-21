use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_gateway_inference::{RouterTiers, RoutingCache};
use litellm_redis_client::{
    Cluster, Credentials, Namespace, Node, RedisClient, RedisConfig, RedisUrl, Target, Timeouts,
    Ttl, commands,
};
use litellm_redis_store::{
    Number, Remote,
    deployment_health::{DeploymentHealth, HealthState as WrittenHealth, HealthStates},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use litellm_router_core::{
    HealthState, Outcomes, RouterCache, UsageCounts, UsageWindow, health::CooldownValue,
};
use rstest::rstest;

const TPM_KEY: &str = "global_router:dep:anthropic/claude-sonnet-5-5:tpm:12-34";
const RPM_KEY: &str = "global_router:dep:anthropic/claude-sonnet-5-5:rpm:12-34";

fn connect(deployment: &RedisDeployment) -> RedisClient {
    let target = match deployment {
        RedisDeployment::Standalone(server) => Target::Url(RedisUrl {
            url: litellm_secret_value::SecretValue::new(server.url().to_owned()),
            tls: None,
            db: None,
            username: None,
            password: None,
        }),
        RedisDeployment::Cluster(cluster) => Target::Cluster(Cluster {
            nodes: cluster
                .nodes()
                .map(|(host, port)| Node {
                    host: host.to_owned(),
                    port,
                })
                .collect(),
            tls: None,
        }),
    };
    RedisClient::new(RedisConfig {
        target,
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(500), None),
        breaker: None,
        events: None,
        health_check: None,
        keepalive: true,
    })
    .unwrap()
}

fn remote(deployment: &RedisDeployment) -> Remote {
    Remote {
        client: connect(deployment),
        default_ttl: Duration::from_secs(60),
    }
}

fn routing(deployment: &RedisDeployment) -> RoutingCache {
    RoutingCache::new(Some(remote(deployment)), RouterTiers::default())
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

#[rstest]
#[case::whole_seconds_as_an_int(
    60.0,
    "{\"exception_received\": \"RateLimitError\", \"status_code\": \"429\", \"timestamp\": 1730000000.25, \"cooldown_time\": 60}"
)]
#[case::fractional_seconds_as_a_float(
    0.5,
    "{\"exception_received\": \"RateLimitError\", \"status_code\": \"429\", \"timestamp\": 1730000000.25, \"cooldown_time\": 0.5}"
)]
#[tokio::test]
async fn a_cooldown_is_stored_as_the_json_python_writes(
    #[case] cooldown_time: f64,
    #[case] json: &str,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let cooldown = CooldownValue {
        exception_received: "RateLimitError".to_owned(),
        status_code: "429".to_owned(),
        timestamp: 1_730_000_000.25,
        cooldown_time,
    };

    routing(&deployment)
        .put_cooldown("dep-a", &cooldown, Duration::from_secs_f64(90.5))
        .await
        .unwrap();

    let key = "deployment:dep-a:cooldown";
    let written: Option<String> = raw.send(commands::get(key)).await.unwrap();
    let Ttl::Expires(remaining) = raw.send(commands::ttl(key)).await.unwrap() else {
        panic!("{key} has no TTL");
    };
    assert_eq!(written.as_deref(), Some(json));
    assert!(
        remaining <= Duration::from_secs(90) && remaining > Duration::from_secs(85),
        "{remaining:?}"
    );
}

#[rstest]
#[tokio::test]
async fn another_gateway_reads_a_cooldown_in_the_order_it_asked(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (first, second) = (routing(&deployment), routing(&deployment));
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
async fn allowed_fails_count_across_gateways_under_pythons_key(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (first, second) = (routing(&deployment), routing(&deployment));
    let ttl = Duration::from_secs(30);

    let counts = (
        first
            .increment_allowed_fails("dep-a", Some("timeout"), ttl)
            .await
            .unwrap(),
        second
            .increment_allowed_fails("dep-a", Some("timeout"), ttl)
            .await
            .unwrap(),
    );

    let stored: Option<String> = raw
        .send(commands::get("deployment:dep-a:allowed_fails:timeout"))
        .await
        .unwrap();
    assert_eq!(counts, (1.0, 2.0));
    assert_eq!(stored.as_deref(), Some("2"));
}

#[rstest]
#[tokio::test]
async fn usage_one_gateway_adds_is_read_by_another_from_the_global_router_counters(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (first, second) = (routing(&deployment), routing(&deployment));

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
    DeploymentHealth::new(Some(remote(&deployment)), None)
        .set(
            HealthStates(vec![(
                "dep-a".to_owned(),
                WrittenHealth {
                    is_healthy: false,
                    timestamp: Number::Float(1.5),
                    reason: "timeout".to_owned(),
                },
            )]),
            Duration::from_secs(60),
        )
        .await
        .unwrap();

    assert_eq!(
        routing(&deployment).health_states().await.unwrap(),
        HashMap::from([(
            "dep-a".to_owned(),
            HealthState {
                is_healthy: false,
                timestamp: 1.5,
            },
        )])
    );
}

#[rstest]
#[tokio::test]
async fn success_and_failure_counts_stay_in_the_gateway_that_counted_them() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let (first, second) = (routing(&deployment), routing(&deployment));

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
async fn a_gateway_without_redis_answers_from_memory() {
    let routing = RoutingCache::new(None, RouterTiers::default());
    let cooldown = cooling_down_now();
    let ttl = Duration::from_secs(30);

    routing.put_cooldown("dep-a", &cooldown, ttl).await.unwrap();
    let fails = (
        routing
            .increment_allowed_fails("dep-a", None, ttl)
            .await
            .unwrap(),
        routing
            .increment_allowed_fails("dep-a", None, ttl)
            .await
            .unwrap(),
    );
    routing.add_usage(&window(), 7.5, 1.0).await.unwrap();
    routing.add_request(&window()).await.unwrap();

    assert_eq!(
        routing.cooldowns(&["dep-a".to_owned()]).await.unwrap(),
        [Some(cooldown)]
    );
    assert_eq!(fails, (1.0, 2.0));
    assert_eq!(
        routing.usage(&window()).await.unwrap(),
        UsageCounts {
            tokens: Some(7.5),
            requests: Some(2.0),
        }
    );
    assert_eq!(routing.health_states().await.unwrap(), HashMap::new());
}
