use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::Error;
use redis::{
    ConnectionAddr, ConnectionInfo, IntoConnectionInfo,
    cluster::{
        ClusterClient, ClusterClientBuilder, ClusterConnection, ClusterPipeline, NodeAddress,
    },
    cluster_routing::{
        MultipleNodeRoutingInfo, ResponsePolicy, RoutingInfo, SingleNodeRoutingInfo,
    },
};

use crate::topology::{RedisNode, RedisTopology};

pub(crate) const REDIS_TIMEOUT: Duration = Duration::from_secs(5);
const REDIS_POOL_SIZE: u32 = 16;

#[allow(private_interfaces)]
pub enum Connections<C> {
    Pool(r2d2::Pool<ConnectionManager>),
    Cluster(r2d2::Pool<ClusterConnectionManager>),
    Fixed(Mutex<C>),
}

impl<C> Connections<C>
where
    C: redis::ConnectionLike + Send + 'static,
{
    pub fn execute<T>(
        &self,
        operation: impl FnOnce(&mut ConnectionRef<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        match self {
            Self::Pool(pool) => {
                let mut pooled = pool.get().map_err(|_| Error::Unavailable)?;
                let result = operation(&mut ConnectionRef::Node(&mut pooled.connection));
                pooled.failed = matches!(result, Err(Error::Unavailable));
                result
            }
            Self::Cluster(pool) => {
                let mut pooled = pool.get().map_err(|_| Error::Unavailable)?;
                let result = operation(&mut ConnectionRef::Cluster(&mut pooled.connection));
                pooled.failed = matches!(result, Err(Error::Unavailable));
                result
            }
            Self::Fixed(connection) => {
                let mut connection = connection.lock().map_err(|_| Error::Unavailable)?;
                operation(&mut ConnectionRef::Node(&mut *connection))
            }
        }
    }

    pub async fn run_blocking<T, F>(connections: Arc<Self>, operation: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&mut ConnectionRef<'_>) -> Result<T, Error> + Send + 'static,
    {
        tokio::task::spawn_blocking(move || connections.execute(operation))
            .await
            .map_err(|_| Error::Unavailable)?
    }

    pub fn fixed(connection: C) -> Self {
        Self::Fixed(Mutex::new(connection))
    }

    pub fn open(url: &str, topology: &RedisTopology) -> Result<Self, Error> {
        match topology {
            RedisTopology::Standalone => Ok(Self::Pool(pool(ConnectionManager::open(url)?)?)),
            RedisTopology::Cluster { startup_nodes } => Ok(Self::Cluster(pool(
                ClusterConnectionManager::open(url, startup_nodes)?,
            )?)),
        }
    }

    /// Closes every idle pooled connection; the next operation opens a fresh one. Connections
    /// checked out right now return to the pool, and a caller-owned connection stays open.
    pub fn disconnect(&self) {
        match self {
            Self::Pool(pool) => close_idle(pool),
            Self::Cluster(pool) => close_idle(pool),
            Self::Fixed(_) => {}
        }
    }
}

fn pool<M: r2d2::ManageConnection>(manager: M) -> Result<r2d2::Pool<M>, Error> {
    r2d2::Pool::builder()
        .max_size(REDIS_POOL_SIZE)
        .min_idle(Some(0))
        .connection_timeout(REDIS_TIMEOUT)
        .test_on_check_out(false)
        .build(manager)
        .map_err(|_| Error::Unavailable)
}

fn close_idle<M, T>(pool: &r2d2::Pool<M>)
where
    M: r2d2::ManageConnection<Connection = PooledConnection<T>>,
{
    let mut idle = Vec::new();
    while let Some(mut connection) = pool.try_get() {
        connection.failed = true;
        idle.push(connection);
    }
}

pub(crate) struct PooledConnection<C> {
    connection: C,
    failed: bool,
}

/// Pools connections without a checkout PING, which would double every operation's round trips.
/// A timed-out command leaves its reply on the socket while redis still reports the connection
/// open, so any connection whose operation failed is discarded instead of being reused.
pub(crate) struct ConnectionManager(redis::Client);

impl ConnectionManager {
    fn open(url: &str) -> Result<Self, Error> {
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

pub(crate) struct ClusterConnectionManager(ClusterClient);

impl ClusterConnectionManager {
    fn open(url: &str, startup_nodes: &[RedisNode]) -> Result<Self, Error> {
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
    /// Runs `pipeline` and decodes its non-ignored replies as `T`. A cluster connection refuses
    /// `Pipeline::query`, so there a transaction goes to its keys' slot as one MULTI/EXEC and
    /// anything else is split per node by `ClusterPipeline`; either way the raw replies are
    /// handed back to `pipeline` to decode.
    pub(crate) fn query_pipeline<T: redis::FromRedisValue>(
        &mut self,
        pipeline: &redis::Pipeline,
    ) -> Result<T, Error> {
        match self {
            Self::Node(connection) => pipeline.query(*connection),
            Self::Cluster(connection) if pipeline.is_transaction() => {
                redis::ConnectionLike::req_packed_commands(
                    *connection,
                    &pipeline.get_packed_pipeline(),
                    pipeline.len() + 1,
                    1,
                )
                .and_then(|replies| pipeline.query(&mut Replies(Some(replies))))
            }
            Self::Cluster(connection) => {
                let mut cluster = ClusterPipeline::with_capacity(pipeline.len());
                for command in pipeline.cmd_iter() {
                    cluster.add_command(command.clone());
                }
                cluster
                    .query(connection)
                    .and_then(|replies| pipeline.query(&mut Replies(Some(replies))))
            }
        }
        .map_err(|_| Error::Unavailable)
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

/// Hands already received pipeline replies to `Pipeline::query`, so it applies its own
/// ignore and error handling to replies a cluster pipeline gathered from several nodes.
struct Replies(Option<Vec<redis::Value>>);

impl redis::ConnectionLike for Replies {
    fn req_packed_command(&mut self, _: &[u8]) -> redis::RedisResult<redis::Value> {
        Err((redis::ErrorKind::Client, "replies hold a pipeline only").into())
    }

    fn req_packed_commands(
        &mut self,
        _: &[u8],
        _: usize,
        _: usize,
    ) -> redis::RedisResult<Vec<redis::Value>> {
        self.0
            .take()
            .ok_or_else(|| (redis::ErrorKind::Client, "replies were already read").into())
    }

    fn get_db(&self) -> i64 {
        0
    }

    fn check_connection(&mut self) -> bool {
        true
    }

    fn is_open(&self) -> bool {
        true
    }
}
