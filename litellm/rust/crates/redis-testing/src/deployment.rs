use std::time::{Duration, Instant};

use crate::{Error, RedisCluster, RedisServer};

const EVENTUALLY_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug)]
pub enum Topology {
    Standalone,
    Cluster,
}

pub enum RedisDeployment {
    Standalone(RedisServer),
    Cluster(RedisCluster),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientConnection {
    pub node: u16,
    pub id: u64,
    pub user: String,
}

impl RedisDeployment {
    pub async fn start(topology: Topology) -> Result<Self, Error> {
        Ok(match topology {
            Topology::Standalone => Self::Standalone(RedisServer::start().await?),
            Topology::Cluster => Self::Cluster(RedisCluster::start().await?),
        })
    }

    pub async fn cli_on_every_node(&self, args: &[&str]) -> Result<(), Error> {
        match self {
            Self::Standalone(server) => server.cli(args).await.map(drop),
            Self::Cluster(cluster) => cluster.cli_on_every_node(args).await,
        }
    }

    pub async fn clients(&self) -> Result<Vec<ClientConnection>, Error> {
        match self {
            Self::Standalone(server) => server.clients().await,
            Self::Cluster(cluster) => cluster.clients().await,
        }
    }

    pub async fn restart(&self) -> Result<(), Error> {
        match self {
            Self::Standalone(server) => server.restart().await,
            Self::Cluster(cluster) => cluster.restart().await,
        }
    }

    pub async fn stop(&self) -> Result<(), Error> {
        match self {
            Self::Standalone(server) => server.stop().await,
            Self::Cluster(cluster) => cluster.stop().await,
        }
    }

    pub async fn resume(&self) -> Result<(), Error> {
        match self {
            Self::Standalone(server) => server.resume().await,
            Self::Cluster(cluster) => cluster.resume().await,
        }
    }

    pub async fn freeze(&self) -> Result<(), Error> {
        match self {
            Self::Standalone(server) => server.freeze().await,
            Self::Cluster(cluster) => cluster.freeze().await,
        }
    }

    pub async fn thaw(&self) -> Result<(), Error> {
        match self {
            Self::Standalone(server) => server.thaw().await,
            Self::Cluster(cluster) => cluster.thaw().await,
        }
    }
}

pub(crate) fn parse_client_list(node: u16, reply: &str) -> Vec<ClientConnection> {
    reply
        .lines()
        .filter(|line| !line.split(' ').any(|field| field == "cmd=client|list"))
        .filter_map(|line| {
            let field = |name: &str| {
                line.split(' ')
                    .find_map(|field| field.strip_prefix(name)?.strip_prefix('='))
            };
            Some(ClientConnection {
                node,
                id: field("id")?.parse().ok()?,
                user: field("user")?.to_owned(),
            })
        })
        .collect()
}

pub async fn eventually<T>(mut attempt: impl AsyncFnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + EVENTUALLY_TIMEOUT;
    loop {
        if let Some(value) = attempt().await {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "condition not met within {EVENTUALLY_TIMEOUT:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
