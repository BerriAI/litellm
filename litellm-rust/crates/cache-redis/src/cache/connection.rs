use std::collections::HashMap;

use litellm_cache::Error;
use redis::{
    ConnectionAddr, ConnectionInfo, ConnectionLike, IntoConnectionInfo,
    cluster::{ClusterClient, ClusterClientBuilder, ClusterConnection, NodeAddress},
    cluster_routing::{
        MultipleNodeRoutingInfo, ResponsePolicy, RoutingInfo, SingleNodeRoutingInfo, Slot,
    },
};

use super::REDIS_TIMEOUT;
use crate::topology::RedisNode;

pub struct PooledConnection<C> {
    pub(super) connection: C,
    pub(super) failed: bool,
}

/// Pools connections without a checkout PING, which would double every operation's round trips.
/// A timed-out command leaves its reply on the socket while redis still reports the connection
/// open, so any connection whose operation failed is discarded instead of being reused.
pub struct ConnectionManager(redis::Client);

impl ConnectionManager {
    pub(super) fn open(url: &str) -> Result<Self, Error> {
        redis::Client::open(url)
            .map(Self)
            .map_err(|_| Error::Unavailable)
    }
}

impl r2d2::ManageConnection for ConnectionManager {
    type Connection = PooledConnection<redis::Connection>;
    type Error = redis::RedisError;

    fn connect(&self) -> Result<Self::Connection, redis::RedisError> {
        let connection = self.0.get_connection()?;
        connection.set_read_timeout(Some(REDIS_TIMEOUT))?;
        connection.set_write_timeout(Some(REDIS_TIMEOUT))?;
        Ok(PooledConnection {
            connection,
            failed: false,
        })
    }

    fn is_valid(&self, connection: &mut Self::Connection) -> Result<(), redis::RedisError> {
        redis::cmd("PING").query::<String>(&mut connection.connection)?;
        Ok(())
    }

    fn has_broken(&self, connection: &mut Self::Connection) -> bool {
        connection.failed || !redis::ConnectionLike::is_open(&connection.connection)
    }
}

pub struct ClusterConnectionManager(ClusterClient);

impl ClusterConnectionManager {
    pub(super) fn open(url: &str, startup_nodes: &[RedisNode]) -> Result<Self, Error> {
        if startup_nodes.is_empty() {
            return Err(Error::Unavailable);
        }
        let info = url.into_connection_info().map_err(|_| Error::Unavailable)?;
        let nodes = startup_nodes
            .iter()
            .map(|node| node_info(&info, node))
            .collect::<Result<Vec<_>, _>>()?;
        ClusterClientBuilder::new(nodes)
            .connection_timeout(REDIS_TIMEOUT)
            .response_timeout(REDIS_TIMEOUT)
            .build()
            .map(Self)
            .map_err(|_| Error::Unavailable)
    }
}

fn node_info(info: &ConnectionInfo, node: &RedisNode) -> Result<ConnectionInfo, Error> {
    let addr = match info.addr() {
        ConnectionAddr::Tcp(..) => ConnectionAddr::Tcp(node.host.clone(), node.port),
        ConnectionAddr::TcpTls {
            insecure,
            tls_params,
            ..
        } => ConnectionAddr::TcpTls {
            host: node.host.clone(),
            port: node.port,
            insecure: *insecure,
            tls_params: tls_params.clone(),
        },
        _ => return Err(Error::Unavailable),
    };
    Ok(info.clone().set_addr(addr))
}

impl r2d2::ManageConnection for ClusterConnectionManager {
    type Connection = PooledConnection<ClusterConnection>;
    type Error = redis::RedisError;

    fn connect(&self) -> Result<Self::Connection, redis::RedisError> {
        let connection = self.0.get_connection()?;
        connection.set_read_timeout(Some(REDIS_TIMEOUT))?;
        connection.set_write_timeout(Some(REDIS_TIMEOUT))?;
        Ok(PooledConnection {
            connection,
            failed: false,
        })
    }

    fn is_valid(&self, connection: &mut Self::Connection) -> Result<(), redis::RedisError> {
        redis::cmd("PING").query::<String>(&mut connection.connection)?;
        Ok(())
    }

    fn has_broken(&self, connection: &mut Self::Connection) -> bool {
        connection.failed || !redis::ConnectionLike::is_open(&connection.connection)
    }
}

pub enum ConnectionRef<'a> {
    Node(&'a mut dyn redis::ConnectionLike),
    Cluster(&'a mut ClusterConnection),
}

impl redis::ConnectionLike for ConnectionRef<'_> {
    fn req_packed_command(&mut self, cmd: &[u8]) -> redis::RedisResult<redis::Value> {
        match self {
            Self::Node(connection) => connection.req_packed_command(cmd),
            Self::Cluster(connection) => connection.req_packed_command(cmd),
        }
    }

    fn req_packed_commands(
        &mut self,
        cmd: &[u8],
        offset: usize,
        count: usize,
    ) -> redis::RedisResult<Vec<redis::Value>> {
        match self {
            Self::Node(connection) => connection.req_packed_commands(cmd, offset, count),
            Self::Cluster(connection) => connection.req_packed_commands(cmd, offset, count),
        }
    }

    fn get_db(&self) -> i64 {
        match self {
            Self::Node(connection) => connection.get_db(),
            Self::Cluster(connection) => redis::ConnectionLike::get_db(*connection),
        }
    }

    fn supports_pipelining(&self) -> bool {
        match self {
            Self::Node(connection) => connection.supports_pipelining(),
            Self::Cluster(connection) => redis::ConnectionLike::supports_pipelining(*connection),
        }
    }

    fn check_connection(&mut self) -> bool {
        match self {
            Self::Node(connection) => connection.check_connection(),
            Self::Cluster(connection) => connection.check_connection(),
        }
    }

    fn is_open(&self) -> bool {
        match self {
            Self::Node(connection) => connection.is_open(),
            Self::Cluster(connection) => redis::ConnectionLike::is_open(*connection),
        }
    }
}

impl ConnectionRef<'_> {
    pub(crate) fn pipeline(
        &mut self,
        commands: Vec<redis::Cmd>,
    ) -> Result<Vec<redis::Value>, Error> {
        match self {
            Self::Node(connection) => {
                let mut pipeline = redis::pipe();
                for command in &commands {
                    pipeline.add_command(command.clone());
                }
                pipeline
                    .query::<Vec<redis::Value>>(*connection)
                    .map_err(|_| Error::Unavailable)
            }
            Self::Cluster(connection) => {
                let mut replies: Vec<Option<redis::Value>> = vec![None; commands.len()];
                for indices in slot_groups(&commands).into_values() {
                    let mut pipeline = redis::pipe();
                    for index in &indices {
                        pipeline.add_command(commands[*index].clone());
                    }
                    let values = connection
                        .req_packed_commands(&pipeline.get_packed_pipeline(), 0, indices.len())
                        .map_err(|_| Error::Unavailable)?;
                    if values.len() != indices.len() {
                        return Err(Error::Unavailable);
                    }
                    for (index, value) in indices.into_iter().zip(values) {
                        replies[index] = Some(value);
                    }
                }
                replies
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .ok_or(Error::Unavailable)
            }
        }
    }

    pub(crate) fn scan(
        &mut self,
        pattern: &str,
        count: usize,
        mut visit: impl FnMut(&mut Self, Vec<String>) -> Result<bool, Error>,
    ) -> Result<(), Error> {
        let pages = match self {
            Self::Node(connection) => {
                let page = scan_command(0, pattern, count)
                    .query::<ScanPage>(*connection)
                    .map_err(|_| Error::Unavailable)?;
                vec![(None, page)]
            }
            Self::Cluster(connection) => connection
                .route_command(
                    &scan_command(0, pattern, count),
                    RoutingInfo::MultiNode((
                        MultipleNodeRoutingInfo::AllMasters,
                        Some(ResponsePolicy::Special),
                    )),
                )
                .map_err(|_| Error::Unavailable)
                .and_then(primary_pages)?
                .into_iter()
                .map(|(node, page)| (Some(node), page))
                .collect(),
        };
        for (node, (mut cursor, mut keys)) in pages {
            loop {
                if !visit(self, keys)? {
                    return Ok(());
                }
                if cursor == 0 {
                    break;
                }
                (cursor, keys) = self.scan_page(node.as_ref(), cursor, pattern, count)?;
            }
        }
        Ok(())
    }

    pub(crate) fn ping(&mut self) -> Result<bool, redis::RedisError> {
        let command = redis::cmd("PING");
        match self {
            Self::Node(connection) => command
                .query::<String>(*connection)
                .map(|response| response == "PONG"),
            Self::Cluster(connection) => connection
                .route_command(
                    &command,
                    RoutingInfo::MultiNode((
                        MultipleNodeRoutingInfo::AllNodes,
                        Some(ResponsePolicy::AllSucceeded),
                    )),
                )
                .map(|_| true),
        }
    }

    pub(crate) fn node_text(&mut self, command: &redis::Cmd) -> Result<String, Error> {
        match self {
            Self::Node(connection) => command.query(*connection).map_err(|_| Error::Unavailable),
            Self::Cluster(connection) => {
                let value = connection
                    .route_command(
                        command,
                        RoutingInfo::MultiNode((
                            MultipleNodeRoutingInfo::AllNodes,
                            Some(ResponsePolicy::Special),
                        )),
                    )
                    .map_err(|_| Error::Unavailable)?;
                let redis::Value::Map(entries) = value else {
                    return Err(Error::Unavailable);
                };
                let mut replies = entries
                    .into_iter()
                    .map(|(node, reply)| {
                        Ok((
                            redis::from_redis_value::<String>(node)
                                .map_err(|_| Error::Unavailable)?,
                            redis::from_redis_value::<String>(reply)
                                .map_err(|_| Error::Unavailable)?,
                        ))
                    })
                    .collect::<Result<Vec<(String, String)>, Error>>()?;
                replies.sort();
                Ok(replies
                    .into_iter()
                    .map(|(_, reply)| reply)
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
    }

    pub(crate) fn flushall(&mut self) -> Result<(), Error> {
        let command = redis::cmd("FLUSHALL");
        match self {
            Self::Node(connection) => command.query(*connection).map_err(|_| Error::Unavailable),
            Self::Cluster(connection) => connection
                .route_command(
                    &command,
                    RoutingInfo::MultiNode((
                        MultipleNodeRoutingInfo::AllMasters,
                        Some(ResponsePolicy::AllSucceeded),
                    )),
                )
                .map(|_| ())
                .map_err(|_| Error::Unavailable),
        }
    }

    fn scan_page(
        &mut self,
        node: Option<&NodeAddress>,
        cursor: u64,
        pattern: &str,
        count: usize,
    ) -> Result<ScanPage, Error> {
        let command = scan_command(cursor, pattern, count);
        match (self, node) {
            (Self::Node(connection), None) => {
                command.query(*connection).map_err(|_| Error::Unavailable)
            }
            (Self::Cluster(connection), Some(node)) => connection
                .route_command(
                    &command,
                    RoutingInfo::SingleNode(SingleNodeRoutingInfo::ByAddress {
                        host: node.host().to_string(),
                        port: node.port(),
                    }),
                )
                .map_err(|_| Error::Unavailable)
                .and_then(|value| redis::from_redis_value(value).map_err(|_| Error::Unavailable)),
            _ => Err(Error::Unavailable),
        }
    }
}

type ScanPage = (u64, Vec<String>);

fn primary_pages(value: redis::Value) -> Result<Vec<(NodeAddress, ScanPage)>, Error> {
    let redis::Value::Map(entries) = value else {
        return Err(Error::Unavailable);
    };
    entries
        .into_iter()
        .map(|(node, page)| {
            let node = redis::from_redis_value::<String>(node).map_err(|_| Error::Unavailable)?;
            let node = NodeAddress::try_from(node.as_str()).map_err(|_| Error::Unavailable)?;
            let page = redis::from_redis_value::<ScanPage>(page).map_err(|_| Error::Unavailable)?;
            Ok((node, page))
        })
        .collect()
}

fn scan_command(cursor: u64, pattern: &str, count: usize) -> redis::Cmd {
    let mut command = redis::cmd("SCAN");
    command
        .cursor_arg(cursor)
        .arg("MATCH")
        .arg(pattern)
        .arg("COUNT")
        .arg(count);
    command
}

fn slot_groups(commands: &[redis::Cmd]) -> HashMap<Slot, Vec<usize>> {
    let mut groups: HashMap<Slot, Vec<usize>> = HashMap::new();
    for (index, command) in commands.iter().enumerate() {
        let key = match command.args_iter().nth(1) {
            Some(redis::Arg::Simple(key)) => key,
            _ => b"",
        };
        groups.entry(Slot::for_key(key)).or_default().push(index);
    }
    groups
}
