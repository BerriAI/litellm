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
use litellm_redis_store::Remote;
use litellm_redis_testing::RedisDeployment;

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
