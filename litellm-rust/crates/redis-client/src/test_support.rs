use std::time::Duration;

use litellm_circuit_breaker::Breaker;
use litellm_redis_testing::{RedisCluster, RedisServer};
use redis::cluster_routing::{Route, RoutingInfo, SingleNodeRoutingInfo, SlotAddr};

use crate::{
    ClusterEndpoint, ClusterNode, Endpoint, Namespace, RedisClient, RedisConfig,
    connection::Connection,
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Topology {
    Standalone,
    Cluster,
}

pub(crate) enum Server {
    Standalone(RedisServer),
    Cluster(RedisCluster),
}

impl Server {
    pub(crate) async fn start(topology: Topology) -> Self {
        match topology {
            Topology::Standalone => Self::Standalone(RedisServer::start().await.unwrap()),
            Topology::Cluster => Self::Cluster(RedisCluster::start().await.unwrap()),
        }
    }

    pub(crate) fn endpoint(&self) -> Endpoint {
        match self {
            Self::Standalone(server) => Endpoint::Url(server.url().to_owned()),
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
        }
    }

    pub(crate) async fn client(&self, namespace: &str, timeout: Duration) -> RedisClient {
        self.guarded_client(namespace, timeout, None).await
    }

    pub(crate) async fn guarded_client(
        &self,
        namespace: &str,
        timeout: Duration,
        breaker: Option<Breaker>,
    ) -> RedisClient {
        RedisClient::new(RedisConfig {
            endpoint: self.endpoint(),
            namespace: Namespace::new(namespace),
            timeout,
            breaker,
        })
        .unwrap()
    }

    pub(crate) async fn restart(&self) {
        match self {
            Self::Standalone(server) => server.restart().await.unwrap(),
            Self::Cluster(cluster) => cluster.restart().await.unwrap(),
        }
    }

    pub(crate) async fn stop(&self) {
        match self {
            Self::Standalone(server) => server.stop().await.unwrap(),
            Self::Cluster(cluster) => cluster.stop().await.unwrap(),
        }
    }

    pub(crate) async fn resume(&self) {
        match self {
            Self::Standalone(server) => server.resume().await.unwrap(),
            Self::Cluster(cluster) => cluster.resume().await.unwrap(),
        }
    }

    pub(crate) async fn freeze(&self) {
        match self {
            Self::Standalone(server) => server.freeze().await.unwrap(),
            Self::Cluster(cluster) => cluster.freeze().await.unwrap(),
        }
    }

    pub(crate) async fn thaw(&self) {
        match self {
            Self::Standalone(server) => server.thaw().await.unwrap(),
            Self::Cluster(cluster) => cluster.thaw().await.unwrap(),
        }
    }
}

pub(crate) async fn client_id(client: &RedisClient, key: &str) -> i64 {
    let command = redis::cmd("CLIENT").arg("ID").to_owned();
    match client.connection.clone() {
        Connection::Standalone(mut connection) => {
            command.query_async(&mut connection).await.unwrap()
        }
        Connection::Cluster(mut connection) => {
            let node = Route::with_key(key, SlotAddr::Master);
            let routing = RoutingInfo::SingleNode(SingleNodeRoutingInfo::SpecificNode(node));
            redis::from_redis_value(connection.route_command(command, routing).await.unwrap())
                .unwrap()
        }
    }
}

pub(crate) async fn flush_scripts(client: &RedisClient) {
    redis::cmd("SCRIPT")
        .arg("FLUSH")
        .query_async::<()>(&mut client.connection.clone())
        .await
        .unwrap();
}
