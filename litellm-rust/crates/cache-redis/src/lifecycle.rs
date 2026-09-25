use litellm_cache::{
    CacheCodec, CacheConnectionResult, CacheConnectionStatus, ClientInfoCache, ConnectionCache,
    DisconnectCache, Error, PingCache,
};

use crate::{cache::RedisCache, topology::RedisTopology};

impl<S, C> PingCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    fn sync_ping(&self) -> Result<bool, Error> {
        self.execute(|connection| connection.ping().map_err(|_| Error::Unavailable))
    }

    async fn ping(&self) -> Result<bool, Error> {
        self.run(|connection| connection.ping().map_err(|_| Error::Unavailable))
            .await
    }
}

impl<S, C> ConnectionCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    /// Python `RedisCache.test_connection`, or `RedisClusterCache.test_connection` for a
    /// cluster topology, which differs only in its messages.
    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        let label = match self.topology {
            RedisTopology::Standalone => "Redis",
            RedisTopology::Cluster { .. } => "Redis Cluster",
        };
        let ping = self
            .run(|connection| Ok(connection.ping().map_err(|error| error.to_string())))
            .await
            .unwrap_or_else(|error| Err(error.to_string()));
        Ok(match ping {
            Ok(true) => CacheConnectionResult {
                status: CacheConnectionStatus::Success,
                message: format!("{label} connection test successful"),
                error: None,
            },
            Ok(false) => CacheConnectionResult {
                status: CacheConnectionStatus::Failed,
                message: format!("{label} ping returned False"),
                error: None,
            },
            Err(error) => CacheConnectionResult {
                status: CacheConnectionStatus::Failed,
                message: format!("{label} connection failed: {error}"),
                error: Some(error),
            },
        })
    }
}

impl<S, C> DisconnectCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    async fn disconnect(&self) -> Result<(), Error> {
        self.connections.disconnect();
        Ok(())
    }
}

impl<S, C> ClientInfoCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type ClientList = String;
    type Info = String;

    fn client_list(&self) -> Result<Self::ClientList, Error> {
        self.execute(|connection| connection.node_text(redis::cmd("CLIENT").arg("LIST")))
    }

    fn info(&self) -> Result<Self::Info, Error> {
        self.execute(|connection| connection.node_text(&redis::cmd("INFO")))
    }
}
