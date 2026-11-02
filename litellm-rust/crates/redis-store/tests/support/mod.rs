#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::time::Duration;

use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, Credentials, Endpoint, Namespace, RedisClient, RedisConfig,
    Timeouts, TlsOptions,
};
use litellm_redis_store::{
    Number,
    router_cooldowns::Cooldown,
    router_usage::{Counter, Kind},
};
use litellm_redis_testing::RedisDeployment;

pub fn connect(deployment: &RedisDeployment) -> RedisClient {
    connect_with(deployment, None)
}

pub fn connect_guarded(
    deployment: &RedisDeployment,
    breaker: litellm_circuit_breaker::Breaker,
) -> RedisClient {
    connect_with(deployment, Some(breaker))
}

fn connect_with(
    deployment: &RedisDeployment,
    breaker: Option<litellm_circuit_breaker::Breaker>,
) -> RedisClient {
    let endpoint = match deployment {
        RedisDeployment::Standalone(server) => Endpoint::Url {
            url: litellm_secret_value::SecretValue::new(server.url().to_owned()),
            tls: TlsOptions::verified(),
        },
        RedisDeployment::Cluster(cluster) => Endpoint::Cluster(ClusterEndpoint {
            nodes: cluster
                .nodes()
                .map(|(host, port)| ClusterNode {
                    host: host.to_owned(),
                    port,
                })
                .collect(),
            tls: None,
        }),
    };
    RedisClient::new(RedisConfig {
        endpoint,
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(500), None),
        breaker,
        events: None,
        health_check: None,
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
