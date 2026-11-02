use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use redis::{
    AsyncConnectionConfig, Cmd, ConnectionAddr, ConnectionInfo, ErrorKind, Pipeline, RedisError,
    RedisFuture, RedisResult, ServerErrorKind, Value,
    aio::{ConnectionLike, ConnectionManager, ConnectionManagerConfig, MultiplexedConnection},
    cluster::{ClusterClientBuilder, ClusterConfig},
    cluster_async::ClusterConnection,
    io::tcp::TcpSettings,
    sentinel::{SentinelClient, SentinelClientBuilder, SentinelServerType},
};

use crate::{
    ClusterEndpoint, ConnectError, Credentials, Endpoint, SentinelEndpoint, Timeouts,
    config::{self, addr},
    error::Deployment,
};

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
    Sentinel(SentinelConnection),
}

impl Connection {
    pub(crate) fn new(
        endpoint: &Endpoint,
        credentials: &Credentials,
        timeouts: Timeouts,
    ) -> Result<Self, ConnectError> {
        match endpoint {
            Endpoint::Url { .. } | Endpoint::Host(_) => Self::standalone(
                config::connection_info(endpoint, credentials)?,
                credentials,
                timeouts,
            ),
            Endpoint::Cluster(cluster) => Self::cluster(cluster, credentials, timeouts),
            Endpoint::Sentinel(sentinel) => Ok(Self::Sentinel(SentinelConnection::new(
                sentinel,
                credentials,
                timeouts,
            )?)),
        }
    }

    pub(crate) fn deployment(&self) -> Deployment {
        match self {
            Self::Standalone(_) => Deployment::Standalone,
            Self::Cluster(_) => Deployment::Cluster,
            Self::Sentinel(_) => Deployment::Sentinel,
        }
    }

    fn standalone(
        info: ConnectionInfo,
        credentials: &Credentials,
        timeouts: Timeouts,
    ) -> Result<Self, ConnectError> {
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(timeouts.connect))
            .set_response_timeout(Some(timeouts.response))
            .set_number_of_retries(RECONNECT_RETRIES);
        let config = match credentials {
            Credentials::Provider(provider) => config.set_credentials_provider(provider.clone()),
            Credentials::Static { .. } => config,
        };
        let manager = ConnectionManager::new_lazy_with_config(redis::Client::open(info)?, config)?;
        Ok(Self::Standalone(manager))
    }

    fn cluster(
        endpoint: &ClusterEndpoint,
        credentials: &Credentials,
        timeouts: Timeouts,
    ) -> Result<Self, ConnectError> {
        let builder = ClusterClientBuilder::new(config::cluster_addrs(endpoint))
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
        let builder = match &endpoint.tls {
            Some(tls) => builder
                .tls(config::tls_mode(tls))
                .certs(config::tls_certificates(tls)?)
                .danger_accept_invalid_hostnames(!tls.check_hostname),
            None => builder,
        };
        let builder = match credentials {
            Credentials::Static { username, password } => {
                let builder = match username {
                    Some(username) => builder.username(username.expose()),
                    None => builder,
                };
                match password {
                    Some(password) => builder.password(password.expose()),
                    None => builder,
                }
            }
            Credentials::Provider(provider) => builder.set_credentials_provider(provider.clone()),
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
            Self::Sentinel(connection) => connection.req_packed_command(cmd),
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
            Self::Sentinel(connection) => connection.req_packed_commands(pipeline, offset, count),
        }
    }

    fn get_db(&self) -> i64 {
        match self {
            Self::Standalone(connection) => connection.get_db(),
            Self::Cluster(connection) => connection.get_db(),
            Self::Sentinel(connection) => connection.get_db(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct SentinelConnection(Arc<SentinelState>);

struct SentinelState {
    discovery: tokio::sync::Mutex<SentinelClient>,
    current: Mutex<Option<(u64, MultiplexedConnection)>>,
    generations: AtomicU64,
    endpoint: SentinelEndpoint,
    credentials: Credentials,
    config: AsyncConnectionConfig,
    connect_timeout: Duration,
}

impl SentinelConnection {
    fn new(
        endpoint: &SentinelEndpoint,
        credentials: &Credentials,
        timeouts: Timeouts,
    ) -> Result<Self, ConnectError> {
        let sentinels = endpoint
            .sentinels
            .iter()
            .map(|node| addr(&node.host, node.port, endpoint.tls.as_ref()));
        let builder = SentinelClientBuilder::new(
            sentinels,
            &endpoint.service_name,
            SentinelServerType::Master,
        )?;
        let builder = match (&endpoint.sentinel_password, credentials) {
            (
                Some(password),
                Credentials::Static {
                    username: Some(username),
                    ..
                },
            ) => builder
                .set_client_to_sentinel_username(username.expose())
                .set_client_to_sentinel_password(password.expose()),
            (Some(password), _) => builder.set_client_to_sentinel_password(password.expose()),
            (None, _) => builder,
        };
        let builder = match &endpoint.tls {
            Some(tls) => builder
                .set_client_to_sentinel_tls_mode(config::tls_mode(tls))
                .set_client_to_sentinel_certificates(config::tls_certificates(tls)?),
            None => builder,
        };
        let config = AsyncConnectionConfig::new()
            .set_connection_timeout(Some(timeouts.connect))
            .set_response_timeout(Some(timeouts.response));
        let config = match credentials {
            Credentials::Provider(provider) => config.set_credentials_provider(provider.clone()),
            Credentials::Static { .. } => config,
        };
        Ok(Self(Arc::new(SentinelState {
            discovery: tokio::sync::Mutex::new(builder.build()?),
            current: Mutex::new(None),
            generations: AtomicU64::new(0),
            endpoint: endpoint.clone(),
            credentials: credentials.clone(),
            config,
            connect_timeout: timeouts.connect,
        })))
    }

    fn cached(&self) -> Option<(u64, MultiplexedConnection)> {
        self.0
            .current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    async fn connection(&self) -> RedisResult<(u64, MultiplexedConnection)> {
        if let Some(current) = self.cached() {
            return Ok(current);
        }
        let mut discovery = self.0.discovery.lock().await;
        if let Some(current) = self.cached() {
            return Ok(current);
        }
        let master = tokio::time::timeout(self.0.connect_timeout, discovery.async_get_client())
            .await
            .map_err(|_| timed_out("asking the Sentinels for the master timed out"))??;
        let connection = self
            .master_client(master.get_connection_info().addr())?
            .get_multiplexed_async_connection_with_config(&self.0.config)
            .await?;
        let mut current = self
            .0
            .current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let generation = self.0.generations.fetch_add(1, Ordering::Relaxed);
        *current = Some((generation, connection.clone()));
        Ok((generation, connection))
    }

    fn master_client(&self, master: &ConnectionAddr) -> Result<redis::Client, RedisError> {
        let (host, port) = match master {
            ConnectionAddr::Tcp(host, port) | ConnectionAddr::TcpTls { host, port, .. } => {
                (host.as_str(), *port)
            }
            _ => {
                return Err(RedisError::from((
                    ErrorKind::InvalidClientConfig,
                    "Sentinel named a master that is not a TCP address",
                )));
            }
        };
        let endpoint = Endpoint::Host(crate::HostEndpoint {
            host: host.to_owned(),
            port,
            tls: self.0.endpoint.tls.clone(),
            db: self.0.endpoint.db,
        });
        let info =
            config::connection_info(&endpoint, &self.0.credentials).map_err(
                |error| match error {
                    ConnectError::Redis(error) => error,
                    ConnectError::TlsFile { path, .. } => RedisError::from((
                        ErrorKind::InvalidClientConfig,
                        "reading a Redis TLS file",
                        path,
                    )),
                },
            )?;
        redis::Client::open(info)
    }

    fn forget(&self, generation: u64) {
        let mut current = self
            .0
            .current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if current
            .as_ref()
            .is_some_and(|(cached, _)| *cached == generation)
        {
            *current = None;
        }
    }
}

fn timed_out(message: &'static str) -> RedisError {
    RedisError::from(std::io::Error::new(std::io::ErrorKind::TimedOut, message))
}

fn needs_rediscovery(error: &RedisError) -> bool {
    matches!(
        crate::Error::classify(error.clone(), Deployment::Sentinel),
        crate::Error::Transport(_)
    )
}

fn replied_read_only(values: &[Value]) -> bool {
    values.iter().any(|value| {
        matches!(value, Value::ServerError(error) if error.kind() == Some(ServerErrorKind::ReadOnly))
    })
}

impl ConnectionLike for SentinelConnection {
    fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> RedisFuture<'a, Value> {
        Box::pin(async move {
            let (generation, mut connection) = self.connection().await?;
            let reply = connection.req_packed_command(cmd).await;
            if reply.as_ref().is_err_and(needs_rediscovery) {
                self.forget(generation);
            }
            reply
        })
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        pipeline: &'a Pipeline,
        offset: usize,
        count: usize,
    ) -> RedisFuture<'a, Vec<Value>> {
        Box::pin(async move {
            let (generation, mut connection) = self.connection().await?;
            let replies = connection
                .req_packed_commands(pipeline, offset, count)
                .await;
            let rediscover = match &replies {
                Ok(values) => replied_read_only(values),
                Err(error) => needs_rediscovery(error),
            };
            if rediscover {
                self.forget(generation);
            }
            replies
        })
    }

    fn get_db(&self) -> i64 {
        self.0.endpoint.db
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClusterNode;
    use litellm_redis_testing::{RedisSentinel, SERVICE_NAME};

    #[rstest::rstest]
    #[tokio::test]
    async fn a_late_failure_does_not_discard_the_reconnected_sentinel_master() {
        let server = RedisSentinel::start().await.unwrap();
        let (host, port) = server.sentinel();
        let endpoint = SentinelEndpoint {
            sentinels: vec![ClusterNode {
                host: host.to_owned(),
                port,
            }],
            service_name: SERVICE_NAME.to_owned(),
            sentinel_password: None,
            tls: None,
            db: 0,
        };
        let mut client =
            SentinelConnection::new(&endpoint, &Credentials::NONE, Timeouts::default()).unwrap();
        let (old_generation, _) = client.connection().await.unwrap();
        let original: i64 = redis::cmd("CLIENT")
            .arg("ID")
            .query_async(&mut client)
            .await
            .unwrap();
        client.forget(old_generation);
        let replacement: i64 = redis::cmd("CLIENT")
            .arg("ID")
            .query_async(&mut client)
            .await
            .unwrap();
        assert_ne!(original, replacement);

        client.forget(old_generation);
        let after_late_failure: i64 = redis::cmd("CLIENT")
            .arg("ID")
            .query_async(&mut client)
            .await
            .unwrap();

        assert_eq!(replacement, after_late_failure);
    }
}
