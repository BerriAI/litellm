#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::time::Duration;

use litellm_circuit_breaker::Breaker;
use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, Endpoint, Expiry, Namespace, RedisClient, RedisConfig, Timeouts,
};
use litellm_redis_testing::{ClientConnection, RedisDeployment};
use litellm_secret_value::SecretValue;

pub const KEEP: Expiry = Expiry::KeepExisting(Duration::from_secs(60));

pub fn endpoint(deployment: &RedisDeployment) -> Endpoint {
    match deployment {
        RedisDeployment::Standalone(server) => Endpoint::Url(SecretValue::new(server.url())),
        RedisDeployment::Cluster(cluster) => Endpoint::Cluster(ClusterEndpoint {
            nodes: cluster
                .nodes()
                .map(|(host, port)| ClusterNode {
                    host: host.to_owned(),
                    port,
                })
                .collect(),
            username: None,
            password: None,
            tls: false,
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
    connect_to(endpoint(deployment), namespace, timeout, breaker)
}

pub fn connect_to(
    endpoint: Endpoint,
    namespace: &str,
    timeout: Duration,
    breaker: Option<Breaker>,
) -> RedisClient {
    RedisClient::new(RedisConfig {
        endpoint,
        namespace: Namespace::new(namespace),
        timeouts: Timeouts::new(timeout, None),
        breaker,
        events: None,
    })
    .unwrap()
}

pub async fn connections(deployment: &RedisDeployment) -> Vec<ClientConnection> {
    deployment.clients().await.unwrap()
}
