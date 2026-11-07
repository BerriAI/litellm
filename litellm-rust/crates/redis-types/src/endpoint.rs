use std::{collections::BTreeSet, path::PathBuf};

use crate::{Node, Target};

const DEFAULT_PORT: u16 = 6379;

/// The Redis server a target reaches, for state that belongs to the server rather than to one
/// connection, like its circuit breaker. Targets that differ only in db, credentials or TLS reach
/// one server. Different names for one host, like `localhost` and `127.0.0.1`, stay apart.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Endpoint {
    Tcp {
        host: String,
        port: u16,
    },
    Unix(PathBuf),
    Cluster(BTreeSet<(String, u16)>),
    Sentinel {
        monitors: BTreeSet<(String, u16)>,
        service_name: String,
    },
}

impl Target {
    /// None for a url whose server cannot be read from it.
    pub fn endpoint(&self) -> Option<Endpoint> {
        match self {
            Target::Standalone(host) => Some(Endpoint::tcp(&host.host, host.port)),
            Target::Unix(socket) => Some(Endpoint::Unix(socket.path.clone())),
            Target::Url(url) => url_endpoint(url.url.expose()),
            Target::Cluster(cluster) => Some(Endpoint::Cluster(nodes(&cluster.nodes))),
            Target::Sentinel(sentinel) => Some(Endpoint::Sentinel {
                monitors: nodes(&sentinel.monitors),
                service_name: sentinel.service_name.clone(),
            }),
        }
    }
}

impl Endpoint {
    fn tcp(host: &str, port: u16) -> Self {
        Self::Tcp {
            host: host.to_ascii_lowercase(),
            port,
        }
    }
}

fn nodes(nodes: &[Node]) -> BTreeSet<(String, u16)> {
    nodes
        .iter()
        .map(|node| (node.host.to_ascii_lowercase(), node.port))
        .collect()
}

// redis-py's parse_url schemes. host() rather than host_str(), which keeps an IPv6 address's brackets
fn url_endpoint(url: &str) -> Option<Endpoint> {
    let url = url::Url::parse(url).ok()?;
    match url.scheme() {
        "redis" | "rediss" => {
            let host = match url.host()? {
                url::Host::Domain(domain) => domain.to_owned(),
                url::Host::Ipv4(ip) => ip.to_string(),
                url::Host::Ipv6(ip) => ip.to_string(),
            };
            Some(Endpoint::tcp(&host, url.port().unwrap_or(DEFAULT_PORT)))
        }
        "unix" => Some(Endpoint::Unix(url.path().into())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use litellm_secret_value::SecretValue;
    use rstest::rstest;

    use crate::{Cluster, Node, RedisUrl, Sentinel, Standalone, Target, TlsOptions, UnixSocket};

    fn host(host: &str, port: u16, db: i64) -> Target {
        Target::Standalone(Standalone {
            host: host.to_owned(),
            port,
            tls: None,
            db,
        })
    }

    fn url(url: &str) -> Target {
        Target::Url(RedisUrl {
            url: SecretValue::new(url.to_owned()),
            tls: None,
            db: None,
            username: None,
            password: None,
        })
    }

    fn cluster(nodes: &[(&str, u16)]) -> Target {
        Target::Cluster(Cluster {
            nodes: nodes
                .iter()
                .map(|(host, port)| Node {
                    host: (*host).to_owned(),
                    port: *port,
                })
                .collect(),
            tls: None,
        })
    }

    fn sentinel(service_name: &str, db: i64) -> Target {
        Target::Sentinel(Sentinel {
            monitors: vec![Node {
                host: "s1".to_owned(),
                port: 26379,
            }],
            service_name: service_name.to_owned(),
            monitor_auth: None,
            tls: None,
            db,
        })
    }

    #[rstest]
    #[case::another_db(host("r1", 6379, 0), host("r1", 6379, 1))]
    #[case::host_case(host("r1", 6379, 0), host("R1", 6379, 0))]
    #[case::tls(
        host("r1", 6379, 0),
        Target::Standalone(Standalone {
            host: "r1".to_owned(),
            port: 6379,
            tls: Some(TlsOptions::verified()),
            db: 0,
        })
    )]
    #[case::url_and_host(host("r1", 6379, 0), url("redis://r1:6379/0"))]
    #[case::url_credentials_and_db(url("redis://r1:6379/0"), url("redis://user:pw@r1/2?db=3"))]
    #[case::rediss_url(host("r1", 6380, 0), url("rediss://r1:6380"))]
    #[case::ipv6_url(host("::1", 6379, 0), url("redis://[::1]:6379"))]
    #[case::unix_url(
        Target::Unix(UnixSocket { path: "/tmp/redis.sock".into(), db: 0 }),
        url("unix:///tmp/redis.sock?db=2")
    )]
    #[case::cluster_node_order(cluster(&[("a", 1), ("b", 2)]), cluster(&[("b", 2), ("a", 1)]))]
    #[case::sentinel_db(sentinel("mymaster", 0), sentinel("mymaster", 1))]
    fn targets_on_one_server_share_an_endpoint(#[case] left: Target, #[case] right: Target) {
        assert_eq!(left.endpoint(), right.endpoint());
        assert!(left.endpoint().is_some());
    }

    #[rstest]
    #[case::another_port(host("r1", 6379, 0), url("redis://r1:6380"))]
    #[case::another_host(host("r1", 6379, 0), host("r2", 6379, 0))]
    #[case::another_cluster(cluster(&[("a", 1)]), cluster(&[("a", 1), ("b", 2)]))]
    #[case::another_service(sentinel("mymaster", 0), sentinel("other", 0))]
    #[case::tcp_and_unix(
        host("localhost", 6379, 0),
        Target::Unix(UnixSocket { path: "/tmp/redis.sock".into(), db: 0 })
    )]
    fn targets_on_different_servers_do_not(#[case] left: Target, #[case] right: Target) {
        assert_ne!(left.endpoint(), right.endpoint());
    }

    #[rstest]
    fn a_url_with_no_server_has_no_endpoint() {
        assert_eq!(url("http://r1:6379").endpoint(), None);
    }
}
