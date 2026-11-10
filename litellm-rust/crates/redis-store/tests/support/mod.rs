#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_circuit_breaker::{State, Transition};
use litellm_redis_client::{
    CallEvent, Cluster, Credentials, Namespace, Node, Operation, RedisClient, RedisConfig,
    RedisEvents, RedisUrl, Target, Timeouts, current_tag,
};
use litellm_redis_store::{
    Number, Remote, TierPolicy,
    router_cooldowns::Cooldown,
    router_usage::{Counter, Kind},
};
use litellm_redis_testing::RedisDeployment;

/// `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS` and `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
pub const COOLDOWN_POLICY: TierPolicy = TierPolicy {
    refresh_after: Duration::from_secs(1),
    tracked_reads: 1000,
    default_in_memory_ttl: None,
};

/// `DualCache`'s default `redis_batch_cache_expiry` and `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
pub const USAGE_POLICY: TierPolicy = TierPolicy {
    refresh_after: Duration::from_secs(10),
    tracked_reads: 1000,
    default_in_memory_ttl: None,
};

pub fn connect(deployment: &RedisDeployment) -> RedisClient {
    connect_with(deployment, None, None)
}

/// `client` with `RedisCache`'s default TTL when `litellm.default_redis_ttl` is unset.
pub fn remote(client: RedisClient) -> Remote {
    Remote {
        client,
        default_ttl: Duration::from_secs(60),
    }
}

/// Every round trip a client made, as the client reports it.
#[derive(Default)]
pub struct RoundTrips(Mutex<Vec<Operation>>);

impl RoundTrips {
    pub fn seen(&self) -> Vec<Operation> {
        self.0.lock().unwrap().clone()
    }
}

impl RedisEvents for RoundTrips {
    fn call(&self, event: &CallEvent) {
        self.0.lock().unwrap().push(event.operation);
    }

    fn breaker(&self, _transition: &Transition<State>) {}
}

pub fn connect_observed(deployment: &RedisDeployment) -> (RedisClient, Arc<RoundTrips>) {
    let trips = Arc::new(RoundTrips::default());
    (connect_with(deployment, None, Some(trips.clone())), trips)
}

/// The `&'static str` tag each call ran under, `None` for an untagged call.
#[derive(Default)]
pub struct CallTags(Mutex<Vec<Option<&'static str>>>);

impl CallTags {
    pub fn seen(&self) -> Vec<Option<&'static str>> {
        self.0.lock().unwrap().clone()
    }
}

impl RedisEvents for CallTags {
    fn call(&self, _event: &CallEvent) {
        let tag = current_tag().and_then(|tag| tag.downcast_ref::<&'static str>().copied());
        self.0.lock().unwrap().push(tag);
    }

    fn breaker(&self, _transition: &Transition<State>) {}
}

pub fn connect_tagged(deployment: &RedisDeployment) -> (RedisClient, Arc<CallTags>) {
    let tags = Arc::new(CallTags::default());
    (connect_with(deployment, None, Some(tags.clone())), tags)
}

pub fn connect_guarded(
    deployment: &RedisDeployment,
    breaker: litellm_circuit_breaker::Breaker,
) -> RedisClient {
    connect_with(deployment, Some(breaker), None)
}

fn connect_with(
    deployment: &RedisDeployment,
    breaker: Option<litellm_circuit_breaker::Breaker>,
    events: Option<Arc<dyn RedisEvents>>,
) -> RedisClient {
    let endpoint = match deployment {
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
        target: endpoint,
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(500), None),
        breaker,
        events,
        health_check: None,
        keepalive: true,
    })
    .unwrap()
}

/// What main's `RedisCache` writes for a cooldown: `str(value)` with `EX int(ttl)`, read back with
/// `json.loads` then `ast.literal_eval` (`litellm/caching/redis_cache.py` `set_cache` and
/// `_get_cache_logic`). Keys come from main's `CooldownCache`. The same literals sit in
/// `tests/unit/redis/stores/test_router_cooldowns.py`, checked against main's codec.
pub struct CooldownCase {
    pub model_id: &'static str,
    pub cooldown: Cooldown,
    pub ttl_seconds: f64,
    pub key: &'static str,
    pub redis_ttl_seconds: u64,
    pub str_dict: &'static str,
    pub json: &'static str,
}

pub fn cooldowns() -> [CooldownCase; 2] {
    [
        CooldownCase {
            model_id: "dep-a",
            cooldown: Cooldown {
                exception_received: "RateLimitError: limited é 'quoted'".to_owned(),
                status_code: "429".to_owned(),
                timestamp: Number::Float(1_730_000_000.25),
                cooldown_time: Number::Int(60),
            },
            ttl_seconds: 90.5,
            key: "deployment:dep-a:cooldown",
            redis_ttl_seconds: 90,
            str_dict: "{'exception_received': \"RateLimitError: limited é 'quoted'\", 'status_code': '429', 'timestamp': 1730000000.25, 'cooldown_time': 60}",
            json: "{\"exception_received\": \"RateLimitError: limited \\u00e9 'quoted'\", \"status_code\": \"429\", \"timestamp\": 1730000000.25, \"cooldown_time\": 60}",
        },
        CooldownCase {
            model_id: "dep-b",
            cooldown: Cooldown {
                exception_received: "Timeout".to_owned(),
                status_code: "408".to_owned(),
                timestamp: Number::Float(1_730_000_000.5),
                cooldown_time: Number::Float(0.5),
            },
            ttl_seconds: 30.0,
            key: "deployment:dep-b:cooldown",
            redis_ttl_seconds: 30,
            str_dict: "{'exception_received': 'Timeout', 'status_code': '408', 'timestamp': 1730000000.5, 'cooldown_time': 0.5}",
            json: "{\"exception_received\": \"Timeout\", \"status_code\": \"408\", \"timestamp\": 1730000000.5, \"cooldown_time\": 0.5}",
        },
    ]
}

/// main's allowed-fails keys for `dep-a`: an empty suffix is the same key as none.
pub struct AllowedFailsCase {
    pub suffix: Option<&'static str>,
    pub key: &'static str,
}

pub const ALLOWED_FAILS: [AllowedFailsCase; 3] = [
    AllowedFailsCase {
        suffix: None,
        key: "deployment:dep-a:allowed_fails",
    },
    AllowedFailsCase {
        suffix: Some(""),
        key: "deployment:dep-a:allowed_fails",
    },
    AllowedFailsCase {
        suffix: Some("timeout"),
        key: "deployment:dep-a:allowed_fails:timeout",
    },
];

/// main's `RouterCacheEnum` usage keys, including the `None` a missing model prints into the key.
pub struct UsageCase {
    pub counter: Counter,
    pub key: &'static str,
}

pub fn usage() -> [UsageCase; 2] {
    [
        UsageCase {
            counter: Counter {
                deployment_id: "dep".to_owned(),
                model: Some("anthropic/claude-sonnet-5-5".to_owned()),
                minute: "12-34".to_owned(),
                kind: Kind::Tpm,
            },
            key: "global_router:dep:anthropic/claude-sonnet-5-5:tpm:12-34",
        },
        UsageCase {
            counter: Counter {
                deployment_id: "dep".to_owned(),
                model: None,
                minute: "09-05".to_owned(),
                kind: Kind::Rpm,
            },
            key: "global_router:dep:None:rpm:09-05",
        },
    ]
}

/// A TTL Redis reports for a key written `whole_seconds` ago at most a few seconds back.
pub fn within(ttl: Duration, whole_seconds: u64) -> bool {
    ttl <= Duration::from_secs(whole_seconds)
        && ttl > Duration::from_secs(whole_seconds.saturating_sub(5))
}

/// How many times each command reached the servers since the last `CONFIG RESETSTAT`, so a test
/// can pin the commands a store sends against the ones main's Python sends.
pub async fn calls(deployment: &RedisDeployment, command: &str) -> u64 {
    let stats = match deployment {
        RedisDeployment::Standalone(server) => server.cli(&["INFO", "commandstats"]).await.unwrap(),
        RedisDeployment::Cluster(cluster) => {
            let mut stats = String::new();
            for (_, port) in cluster.nodes() {
                stats.push_str(
                    &cluster
                        .cli_on_node(port, &["INFO", "commandstats"])
                        .await
                        .unwrap(),
                );
            }
            stats
        }
    };
    let prefix = format!("cmdstat_{command}:calls=");
    stats
        .lines()
        .filter_map(|line| line.trim().strip_prefix(&prefix))
        .filter_map(|rest| rest.split(',').next()?.parse::<u64>().ok())
        .sum()
}
