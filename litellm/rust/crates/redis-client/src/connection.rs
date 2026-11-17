use std::{
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use redis::{
    AsyncConnectionConfig, Cmd, ConnectionAddr, ConnectionInfo, ErrorKind, Pipeline,
    RedisConnectionInfo, RedisError, RedisFuture, RedisResult, ServerErrorKind, TlsCertificates,
    TlsMode, Value,
    aio::{ConnectionLike, ConnectionManager, ConnectionManagerConfig, MultiplexedConnection},
    cluster::{ClusterClientBuilder, ClusterConfig},
    cluster_async::ClusterConnection,
    cluster_routing::{MultipleNodeRoutingInfo, ResponsePolicy, RoutingInfo},
    sentinel::{SentinelClient, SentinelClientBuilder, SentinelServerType},
};

use crate::{
    Cluster, ConnectError, Credentials, Sentinel, Target, Timeouts,
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
    Standalone(Renewing<ConnectionManager>),
    Cluster(Renewing<ClusterConnection>),
    Sentinel(SentinelConnection),
}

impl Connection {
    pub(crate) fn new(
        target: &Target,
        credentials: &Credentials,
        timeouts: Timeouts,
        keepalive: bool,
    ) -> Result<Self, ConnectError> {
        match target {
            Target::Url(_) | Target::Standalone(_) | Target::Unix(_) => Self::standalone(
                config::connection_info(target, credentials, keepalive)?,
                credentials,
                timeouts,
            ),
            Target::Cluster(cluster) => Self::cluster(cluster, credentials, timeouts, keepalive),
            Target::Sentinel(sentinel) => Ok(Self::Sentinel(SentinelConnection::new(
                sentinel,
                credentials,
                timeouts,
                keepalive,
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
        let client = redis::Client::open(info)?;
        Ok(Self::Standalone(Renewing::new(move || {
            ConnectionManager::new_lazy_with_config(client.clone(), config.clone())
        })?))
    }

    fn cluster(
        endpoint: &Cluster,
        credentials: &Credentials,
        timeouts: Timeouts,
        keepalive: bool,
    ) -> Result<Self, ConnectError> {
        let builder = ClusterClientBuilder::new(config::cluster_addrs(endpoint))
            .tcp_settings(config::tcp_settings(keepalive))
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
        let client = builder.build()?;
        Ok(Self::Cluster(Renewing::new(move || {
            Ok(client.get_pending_async_connection_with_config(config.clone()))
        })?))
    }
}

impl Connection {
    /// `PING` on the one multiplexed connection, or on every cluster node, so a dead socket is
    /// replaced before a real command uses it.
    pub(crate) async fn ping_every_node(&self) -> RedisResult<()> {
        let ping = redis::cmd("PING");
        match self {
            Self::Cluster(renewing) => {
                let (generation, mut connection) = renewing.current();
                let reply = connection
                    .route_command(
                        ping,
                        RoutingInfo::MultiNode((
                            MultipleNodeRoutingInfo::AllNodes,
                            Some(ResponsePolicy::AllSucceeded),
                        )),
                    )
                    .await;
                renewing.settle(generation, &reply);
                reply.map(drop)
            }
            Self::Standalone(_) | Self::Sentinel(_) => {
                ping.query_async::<Value>(&mut self.clone()).await.map(drop)
            }
        }
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

/// A connection swapped for a fresh one once a command on it times out, like redis-py disconnecting
/// a connection whose read timed out. redis-rs keeps the socket after a timeout and the cluster
/// client retries on it, so a socket that went silent would time out every command until TCP gives
/// up on it.
#[derive(Clone)]
pub(crate) struct Renewing<C>(Arc<RenewingState<C>>);

type Open<C> = Box<dyn Fn() -> RedisResult<C> + Send + Sync>;

struct RenewingState<C> {
    open: Open<C>,
    current: Mutex<(u64, C)>,
}

impl<C: Clone> Renewing<C> {
    fn new(open: impl Fn() -> RedisResult<C> + Send + Sync + 'static) -> RedisResult<Self> {
        let first = open()?;
        Ok(Self(Arc::new(RenewingState {
            open: Box::new(open),
            current: Mutex::new((0, first)),
        })))
    }

    fn current(&self) -> (u64, C) {
        self.0
            .current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Replaces the connection of `generation` when its reply timed out, unless another timeout
    /// already replaced it.
    fn settle<T>(&self, generation: u64, reply: &RedisResult<T>) {
        if !reply.as_ref().is_err_and(RedisError::is_timeout) {
            return;
        }
        let mut current = self
            .0
            .current
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if current.0 != generation {
            return;
        }
        match (self.0.open)() {
            Ok(fresh) => *current = (generation + 1, fresh),
            Err(error) => {
                tracing::warn!(%error, "replacing a timed-out Redis connection failed")
            }
        }
    }
}

impl<C: ConnectionLike + Clone + Send> ConnectionLike for Renewing<C> {
    fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> RedisFuture<'a, Value> {
        Box::pin(async move {
            let (generation, mut connection) = self.current();
            let reply = connection.req_packed_command(cmd).await;
            self.settle(generation, &reply);
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
            let (generation, mut connection) = self.current();
            let replies = connection
                .req_packed_commands(pipeline, offset, count)
                .await;
            self.settle(generation, &replies);
            replies
        })
    }

    fn get_db(&self) -> i64 {
        self.current().1.get_db()
    }
}

#[derive(Clone)]
pub(crate) struct SentinelConnection(Arc<SentinelState>);

struct SentinelState {
    discovery: tokio::sync::Mutex<()>,
    current: Mutex<Option<(u64, MultiplexedConnection)>>,
    generations: AtomicU64,
    endpoint: Sentinel,
    credentials: Credentials,
    tls: Option<(TlsMode, TlsCertificates)>,
    config: AsyncConnectionConfig,
    connect_timeout: Duration,
    keepalive: bool,
}

impl SentinelConnection {
    fn new(
        endpoint: &Sentinel,
        credentials: &Credentials,
        timeouts: Timeouts,
        keepalive: bool,
    ) -> Result<Self, ConnectError> {
        let tls = match &endpoint.tls {
            Some(tls) => Some((config::tls_mode(tls), config::tls_certificates(tls)?)),
            None => None,
        };
        let state = SentinelState {
            discovery: tokio::sync::Mutex::new(()),
            current: Mutex::new(None),
            generations: AtomicU64::new(0),
            endpoint: endpoint.clone(),
            credentials: credentials.clone(),
            tls,
            config: async_config(credentials, timeouts),
            connect_timeout: timeouts.connect,
            keepalive,
        };
        state.sentinel_client(&RedisConnectionInfo::default())?;
        Ok(Self(Arc::new(state)))
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
        let _discovering = self.0.discovery.lock().await;
        if let Some(current) = self.cached() {
            return Ok(current);
        }
        let connection = self
            .discover()
            .await?
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

    pub(crate) async fn master(&self) -> RedisResult<redis::Client> {
        self.discover().await
    }

    async fn discover(&self) -> RedisResult<redis::Client> {
        let master = tokio::time::timeout(self.0.connect_timeout, self.ask_sentinels())
            .await
            .map_err(|_| timed_out("asking the Sentinels for the master timed out"))??;
        self.master_client(master.get_connection_info().addr())
    }

    /// redis-rs sends `ROLE` to the master a Sentinel names before trusting it, so that probe gets
    /// the data credentials and TLS a connection opened now would use. The Sentinel client is built
    /// per discovery since a provider's credentials change after `new`.
    async fn ask_sentinels(&self) -> RedisResult<redis::Client> {
        let data_auth =
            config::authenticated(RedisConnectionInfo::default(), &self.0.credentials).await?;
        self.0.sentinel_client(&data_auth)?.async_get_client().await
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
        let endpoint = Target::Standalone(crate::Standalone {
            host: host.to_owned(),
            port,
            tls: self.0.endpoint.tls.clone(),
            db: self.0.endpoint.db,
        });
        let info = config::connection_info(&endpoint, &self.0.credentials, self.0.keepalive)
            .map_err(|error| match error {
                ConnectError::Redis(error) => error,
                ConnectError::TlsFile { path, .. } => RedisError::from((
                    ErrorKind::InvalidClientConfig,
                    "reading a Redis TLS file",
                    path,
                )),
                ConnectError::TlsKey { path, reason } => RedisError::from((
                    ErrorKind::InvalidClientConfig,
                    "reading the Redis TLS key",
                    format!("{path}: {reason}"),
                )),
            })?;
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

impl SentinelState {
    fn sentinel_client(&self, data_auth: &RedisConnectionInfo) -> RedisResult<SentinelClient> {
        let endpoint = &self.endpoint;
        let sentinels = endpoint
            .monitors
            .iter()
            .map(|node| addr(&node.host, node.port, endpoint.tls.as_ref()));
        let builder = SentinelClientBuilder::new(
            sentinels,
            &endpoint.service_name,
            SentinelServerType::Master,
        )?;
        let (username, password) = match &endpoint.monitor_auth {
            Some(auth) => (auth.username.as_ref(), auth.password.as_ref()),
            None => (None, None),
        };
        let builder = match username {
            Some(username) => builder.set_client_to_sentinel_username(username.expose()),
            None => builder,
        };
        let builder = match password {
            Some(password) => builder.set_client_to_sentinel_password(password.expose()),
            None => builder,
        };
        let builder = match data_auth.username() {
            Some(username) => builder.set_client_to_redis_username(username),
            None => builder,
        };
        let builder = match data_auth.password() {
            Some(password) => builder.set_client_to_redis_password(password),
            None => builder,
        };
        let builder = match &self.tls {
            Some((mode, certificates)) => builder
                .set_client_to_sentinel_tls_mode(*mode)
                .set_client_to_sentinel_certificates(certificates.clone())
                .set_client_to_redis_tls_mode(*mode)
                .set_client_to_redis_certificates(certificates.clone()),
            None => builder,
        };
        builder.build()
    }
}

pub(crate) fn async_config(credentials: &Credentials, timeouts: Timeouts) -> AsyncConnectionConfig {
    let config = AsyncConnectionConfig::new()
        .set_connection_timeout(Some(timeouts.connect))
        .set_response_timeout(Some(timeouts.response));
    match credentials {
        Credentials::Provider(provider) => config.set_credentials_provider(provider.clone()),
        Credentials::Static { .. } => config,
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
            let rediscover = match &reply {
                Ok(value) => replied_read_only(std::slice::from_ref(value)),
                Err(error) => needs_rediscovery(error),
            };
            if rediscover {
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
    use crate::Node;
    use litellm_redis_testing::{RedisSentinel, SERVICE_NAME};

    fn endpoint(server: &RedisSentinel) -> Sentinel {
        let (host, port) = server.sentinel();
        Sentinel {
            monitors: vec![Node {
                host: host.to_owned(),
                port,
            }],
            service_name: SERVICE_NAME.to_owned(),
            monitor_auth: None,
            tls: None,
            db: 0,
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn a_late_failure_does_not_discard_the_reconnected_sentinel_master() {
        let server = RedisSentinel::start().await.unwrap();
        let mut client = SentinelConnection::new(
            &endpoint(&server),
            &Credentials::NONE,
            Timeouts::default(),
            true,
        )
        .unwrap();
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

    #[rstest::rstest]
    #[tokio::test]
    async fn a_read_only_reply_to_one_command_rediscovers_the_master() {
        let server = RedisSentinel::start().await.unwrap();
        let mut client = SentinelConnection::new(
            &endpoint(&server),
            &Credentials::NONE,
            Timeouts::default(),
            true,
        )
        .unwrap();
        let replica = redis::Client::open(format!("redis://127.0.0.1:{}", server.replica_port()))
            .unwrap()
            .get_multiplexed_async_connection()
            .await
            .unwrap();
        let stale = client.0.generations.fetch_add(1, Ordering::Relaxed);
        *client.0.current.lock().unwrap() = Some((stale, replica));

        let on_replica = redis::cmd("SET")
            .arg("k")
            .arg("1")
            .exec_async(&mut client)
            .await;
        let on_master = redis::cmd("SET")
            .arg("k")
            .arg("1")
            .exec_async(&mut client)
            .await;

        assert_eq!(
            on_replica.unwrap_err().kind(),
            ErrorKind::Server(ServerErrorKind::ReadOnly)
        );
        on_master.unwrap();
    }
}
