#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::{pin::Pin, time::Duration};

use futures_util::Stream;
use litellm_circuit_breaker::Breaker;
use litellm_redis_client::{
    BasicAuth, ClusterEndpoint, ClusterNode, Credentials, Endpoint, Expiry, Namespace, RedisClient,
    RedisConfig, RedisError, StreamingCredentialsProvider, Timeouts, TlsOptions,
};
use litellm_redis_testing::{ClientConnection, RedisDeployment};
use litellm_secret_value::SecretValue;
use tokio::sync::watch;

pub const KEEP: Expiry = Expiry::KeepExisting(Duration::from_secs(60));

pub fn endpoint(deployment: &RedisDeployment) -> Endpoint {
    match deployment {
        RedisDeployment::Standalone(server) => Endpoint::Url {
            url: SecretValue::new(server.url()),
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
        endpoint: endpoint(deployment),
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
        endpoint: endpoint(deployment),
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

pub struct Rotating(watch::Sender<BasicAuth>);

impl Rotating {
    pub fn new(username: &str, password: &str) -> Self {
        Self(watch::Sender::new(BasicAuth::new(
            username.to_owned(),
            password.to_owned(),
        )))
    }

    pub fn rotate(&self, username: &str, password: &str) {
        self.0
            .send_replace(BasicAuth::new(username.to_owned(), password.to_owned()));
    }
}

impl StreamingCredentialsProvider for Rotating {
    fn subscribe(
        &self,
    ) -> Pin<Box<dyn Stream<Item = Result<BasicAuth, RedisError>> + Send + 'static>> {
        let receiver = self.0.subscribe();
        Box::pin(futures_util::stream::unfold(
            (receiver, true),
            |(mut receiver, first)| async move {
                if !first {
                    receiver.changed().await.ok()?;
                }
                let credentials = receiver.borrow_and_update().clone();
                Some((Ok(credentials), (receiver, false)))
            },
        ))
    }
}
