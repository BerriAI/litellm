#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::{pin::Pin, time::Duration};

use futures_util::Stream;
use litellm_circuit_breaker::Breaker;
use litellm_redis_client::{
    BasicAuth, Cluster, Credentials, Expiry, Namespace, Node, RedisClient, RedisConfig, RedisError,
    RedisUrl, Standalone, StreamingCredentialsProvider, Target, Timeouts,
};
use litellm_redis_testing::{ClientConnection, RedisDeployment, TcpProxy};
use litellm_secret_value::SecretValue;
use tokio::sync::watch;

pub const KEEP: Expiry = Expiry::KeepExisting(Duration::from_secs(60));

pub fn endpoint(deployment: &RedisDeployment) -> Target {
    match deployment {
        RedisDeployment::Standalone(server) => Target::Url(RedisUrl {
            url: SecretValue::new(server.url()),
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
        health_check: None,
        keepalive: true,
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
        health_check: None,
        keepalive: true,
    })
    .unwrap()
}

pub async fn connections(deployment: &RedisDeployment) -> Vec<ClientConnection> {
    deployment.clients().await.unwrap()
}

/// A `TcpProxy` in front of every node, and a target that reaches the nodes only through them. A
/// cluster announces the proxy ports, so its slot map and redirects lead there too.
pub struct Proxied {
    proxies: Vec<TcpProxy>,
    pub target: Target,
}

impl Proxied {
    pub fn blackhole_open_connections(&self) {
        self.proxies
            .iter()
            .for_each(TcpProxy::blackhole_open_connections);
    }
}

pub async fn proxied(deployment: &RedisDeployment) -> Proxied {
    match deployment {
        RedisDeployment::Standalone(server) => {
            let proxy = TcpProxy::start(server.port()).await.unwrap();
            let target = Target::Standalone(Standalone {
                host: "127.0.0.1".to_owned(),
                port: proxy.port(),
                tls: None,
                db: 0,
            });
            Proxied {
                proxies: vec![proxy],
                target,
            }
        }
        RedisDeployment::Cluster(cluster) => {
            let mut proxies = Vec::new();
            for (_, port) in cluster.nodes() {
                proxies.push(TcpProxy::start(port).await.unwrap());
            }
            let ports: [u16; 3] = std::array::from_fn(|node| proxies[node].port());
            cluster.announce_ports(ports).await.unwrap();
            let target = Target::Cluster(Cluster {
                nodes: ports
                    .iter()
                    .map(|&port| Node {
                        host: "127.0.0.1".to_owned(),
                        port,
                    })
                    .collect(),
                tls: None,
            });
            Proxied { proxies, target }
        }
    }
}

/// `INFO <section>` from every node, concatenated.
async fn info(deployment: &RedisDeployment, section: &str) -> String {
    match deployment {
        RedisDeployment::Standalone(server) => server.cli(&["INFO", section]).await.unwrap(),
        RedisDeployment::Cluster(cluster) => {
            let mut stats = String::new();
            for (_, port) in cluster.nodes() {
                stats.push_str(&cluster.cli_on_node(port, &["INFO", section]).await.unwrap());
            }
            stats
        }
    }
}

/// `INFO commandstats` from every node, so a test can count what was actually sent.
pub async fn command_stats(deployment: &RedisDeployment) -> String {
    info(deployment, "commandstats").await
}

/// Keys looked up on every node since the last `CONFIG RESETSTAT`, found or not.
pub async fn keys_read(deployment: &RedisDeployment) -> u64 {
    let stats = info(deployment, "stats").await;
    ["keyspace_hits:", "keyspace_misses:"]
        .iter()
        .flat_map(|field| {
            stats
                .lines()
                .filter_map(move |line| line.trim().strip_prefix(field)?.parse::<u64>().ok())
        })
        .sum()
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
