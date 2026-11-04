#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::time::Duration;

use litellm_circuit_breaker::Breaker;
use litellm_redis_client::{
    Cluster, Credentials, Expiry, Namespace, Node, RedisClient, RedisConfig, RedisUrl, Target,
    Timeouts,
};
use litellm_redis_testing::{ClientConnection, RedisDeployment};
use litellm_secret_value::SecretValue;

pub const KEEP: Expiry = Expiry::KeepExisting(Duration::from_secs(60));

pub fn endpoint(deployment: &RedisDeployment) -> Target {
    match deployment {
        RedisDeployment::Standalone(server) => Target::Url(RedisUrl {
            url: SecretValue::new(server.url()),
            tls: None,
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
    }
}

pub fn connect(deployment: &RedisDeployment, namespace: &str, timeout: Duration) -> RedisClient {
    connect_guarded(deployment, namespace, timeout, None)
}

pub fn connect_guarded(
    deployment: &RedisDeployment,
    namespace: &str,
    timeout: Duration,
    breaker: Option<Breaker>,
) -> RedisClient {
    RedisClient::new(RedisConfig {
        target: endpoint(deployment),
        credentials: Credentials::NONE,
        namespace: Namespace::new(namespace),
        timeouts: Timeouts::new(timeout, None),
        breaker,
        events: None,
    })
    .unwrap()
}

pub fn connect_as(deployment: &RedisDeployment, credentials: Credentials) -> RedisClient {
    RedisClient::new(RedisConfig {
        target: endpoint(deployment),
        credentials,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_secs(2), None),
        breaker: None,
        events: None,
    })
    .unwrap()
}

pub async fn connections(deployment: &RedisDeployment) -> Vec<ClientConnection> {
    deployment.clients().await.unwrap()
}

/// `INFO commandstats` from every node, so a test can count what was actually sent.
pub async fn command_stats(deployment: &RedisDeployment) -> String {
    match deployment {
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
    }
}

/// How many times `command` reached a server, summed across the nodes in `stats`.
pub fn calls(stats: &str, command: &str) -> u64 {
    let prefix = format!("cmdstat_{command}:calls=");
    stats
        .lines()
        .filter_map(|line| line.trim().strip_prefix(&prefix))
        .filter_map(|rest| rest.split(',').next()?.parse::<u64>().ok())
        .sum()
}
