use redis::{
    Cmd, ConnectionInfo, Pipeline, RedisFuture, RedisResult, Value,
    aio::{ConnectionLike, ConnectionManager, ConnectionManagerConfig},
    cluster::{ClusterClientBuilder, ClusterConfig},
    cluster_async::ClusterConnection,
    io::tcp::TcpSettings,
};

use crate::{ClusterEndpoint, Timeouts, error::Deployment};

// redis-py 5.3.1 redis/asyncio/connection.py: without retry options a connection gets Retry(NoBackoff(), 0)
const RECONNECT_RETRIES: usize = 0;
// redis-py 5.3.1 redis/asyncio/cluster.py: cluster_error_retry_attempts=3
const CLUSTER_RETRIES: u32 = 3;
// redis-py 5.3.1 redis/asyncio/cluster.py sleeps a flat 0.25s before retrying after CLUSTERDOWN.
// redis-rs draws the wait from [min, max), so a 1ms window pins it to 250ms
const CLUSTER_RETRY_WAIT_MS: u64 = 250;

#[derive(Clone)]
pub(crate) enum Connection {
    Standalone(ConnectionManager),
    Cluster(ClusterConnection),
}

impl Connection {
    pub(crate) fn deployment(&self) -> Deployment {
        match self {
            Self::Standalone(_) => Deployment::Standalone,
            Self::Cluster(_) => Deployment::Cluster,
        }
    }

    pub(crate) fn standalone(info: ConnectionInfo, timeouts: Timeouts) -> RedisResult<Self> {
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(timeouts.connect))
            .set_response_timeout(Some(timeouts.response))
            .set_number_of_retries(RECONNECT_RETRIES);
        let info = info.set_tcp_settings(TcpSettings::default().set_nodelay(true));
        let manager = ConnectionManager::new_lazy_with_config(redis::Client::open(info)?, config)?;
        Ok(Self::Standalone(manager))
    }

    pub(crate) fn cluster(endpoint: &ClusterEndpoint, timeouts: Timeouts) -> RedisResult<Self> {
        let builder = ClusterClientBuilder::new(endpoint.addrs())
            .tcp_settings(
                TcpSettings::default()
                    .set_nodelay(true)
                    .set_keepalive(socket2::TcpKeepalive::new()),
            )
            .connection_timeout(timeouts.connect)
            .response_timeout(timeouts.response)
            .retries(CLUSTER_RETRIES)
            .retry_wait_formula(0, 2)
            .min_retry_wait(CLUSTER_RETRY_WAIT_MS)
            .max_retry_wait(CLUSTER_RETRY_WAIT_MS + 1);
        let builder = match &endpoint.username {
            Some(username) => builder.username(username.expose()),
            None => builder,
        };
        let builder = match &endpoint.password {
            Some(password) => builder.password(password.expose()),
            None => builder,
        };
        // The builder's response timeout bounds the whole request, but with_config overwrites each node's
        // response timeout with this one even when unset, so it has to be repeated here
        let config = ClusterConfig::new().set_response_timeout(timeouts.response);
        Ok(Self::Cluster(
            builder
                .build()?
                .get_pending_async_connection_with_config(config),
        ))
    }
}

impl ConnectionLike for Connection {
    fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> RedisFuture<'a, Value> {
        match self {
            Self::Standalone(connection) => connection.req_packed_command(cmd),
            Self::Cluster(connection) => connection.req_packed_command(cmd),
        }
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        pipeline: &'a Pipeline,
        offset: usize,
        count: usize,
    ) -> RedisFuture<'a, Vec<Value>> {
        match self {
            Self::Standalone(connection) => connection.req_packed_commands(pipeline, offset, count),
            Self::Cluster(connection) => connection.req_packed_commands(pipeline, offset, count),
        }
    }

    fn get_db(&self) -> i64 {
        match self {
            Self::Standalone(connection) => connection.get_db(),
            Self::Cluster(connection) => connection.get_db(),
        }
    }
}
