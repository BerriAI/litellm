use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, DEFAULT_TIMEOUT, Endpoint, Namespace, RedisClient, RedisConfig,
};
use litellm_redis_testing::{RedisCluster, RedisServer};
use litellm_secret_value::SecretValue;

#[derive(Clone, Copy, Debug)]
pub enum Topology {
    Standalone,
    Cluster,
}

pub enum Server {
    Standalone(RedisServer),
    Cluster(RedisCluster),
}

impl Server {
    pub async fn start(topology: Topology) -> Self {
        match topology {
            Topology::Standalone => Self::Standalone(RedisServer::start().await.unwrap()),
            Topology::Cluster => Self::Cluster(RedisCluster::start().await.unwrap()),
        }
    }

    pub fn client(&self) -> RedisClient {
        let endpoint = match self {
            Self::Standalone(server) => Endpoint::Url(SecretValue::new(server.url().to_owned())),
            Self::Cluster(cluster) => Endpoint::Cluster(ClusterEndpoint {
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
        };
        RedisClient::new(RedisConfig {
            endpoint,
            namespace: Namespace::default(),
            timeout: DEFAULT_TIMEOUT,
            breaker: None,
        })
        .unwrap()
    }
}
