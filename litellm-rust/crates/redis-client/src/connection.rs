use std::time::Duration;

use redis::{
    Cmd, ConnectionInfo, Pipeline, RedisFuture, RedisResult, Value,
    aio::{ConnectionLike, ConnectionManager, ConnectionManagerConfig},
    cluster::{ClusterClientBuilder, ClusterConfig},
    cluster_async::ClusterConnection,
};

use crate::ClusterEndpoint;

const RETRIES: u32 = 3;

#[derive(Clone)]
pub(crate) enum Connection {
    Standalone(ConnectionManager),
    Cluster(ClusterConnection),
}

impl Connection {
    pub(crate) fn standalone(info: ConnectionInfo, timeout: Duration) -> RedisResult<Self> {
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(timeout))
            .set_response_timeout(Some(timeout))
            .set_number_of_retries(RETRIES as usize);
        let manager = ConnectionManager::new_lazy_with_config(redis::Client::open(info)?, config)?;
        Ok(Self::Standalone(manager))
    }

    pub(crate) fn cluster(endpoint: &ClusterEndpoint, timeout: Duration) -> RedisResult<Self> {
        let builder = ClusterClientBuilder::new(endpoint.addrs())
            .connection_timeout(timeout)
            .response_timeout(timeout)
            .retries(RETRIES);
        let builder = match &endpoint.username {
            Some(username) => builder.username(username),
            None => builder,
        };
        let builder = match &endpoint.password {
            Some(password) => builder.password(password),
            None => builder,
        };
        let config = ClusterConfig::new()
            .set_connection_timeout(timeout)
            .set_response_timeout(timeout);
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
