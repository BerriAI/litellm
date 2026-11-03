use std::{sync::Arc, time::Duration};

use futures_util::{StreamExt, stream::BoxStream};
use redis::{ConnectionInfo, RedisResult, StreamingCredentialsProvider, aio::PubSubStream};

use crate::{
    ConnectError, Credentials, Standalone, Target, Timeouts, config,
    connection::{Connection, SentinelConnection},
};

// litellm/proxy/common_utils/auth_cache_invalidation_pubsub.py: _BACKOFF_INITIAL_SECONDS, _BACKOFF_MAX_SECONDS
const BACKOFF_INITIAL: Duration = Duration::from_secs(5);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// A subscription the way redis-py's `PubSub` holds one: a dedicated RESP2 connection that sends
/// `SUBSCRIBE`, authenticated once when it connects.
#[derive(Clone)]
pub(crate) struct Subscriber {
    source: Source,
    credentials: Credentials,
    connect_timeout: Duration,
}

#[derive(Clone)]
enum Source {
    Nodes(Arc<[ConnectionInfo]>),
    Sentinel(SentinelConnection),
}

impl Subscriber {
    pub(crate) fn new(
        endpoint: &Target,
        credentials: &Credentials,
        timeouts: Timeouts,
        connection: &Connection,
    ) -> Result<Self, ConnectError> {
        let source = match (endpoint, connection) {
            (_, Connection::Sentinel(sentinel)) => Source::Sentinel(sentinel.clone()),
            (Target::Cluster(cluster), _) => Source::Nodes(
                cluster
                    .nodes
                    .iter()
                    .map(|node| {
                        let node = Target::Standalone(Standalone {
                            host: node.host.clone(),
                            port: node.port,
                            tls: cluster.tls.clone(),
                            db: 0,
                        });
                        config::connection_info(&node, credentials)
                    })
                    .collect::<Result<_, ConnectError>>()?,
            ),
            _ => Source::Nodes(Arc::new([config::connection_info(endpoint, credentials)?])),
        };
        Ok(Self {
            source,
            credentials: credentials.clone(),
            connect_timeout: timeouts.connect,
        })
    }

    pub(crate) fn messages(self, channel: String) -> BoxStream<'static, Vec<u8>> {
        futures_util::stream::unfold(
            (self, channel, None::<PubSubStream>, BACKOFF_INITIAL),
            |(subscriber, channel, live, backoff)| async move {
                let mut live = live;
                let mut backoff = backoff;
                loop {
                    let Some(current) = live.as_mut() else {
                        match subscriber.subscribe(&channel).await {
                            Ok(subscribed) => {
                                live = Some(subscribed);
                                backoff = BACKOFF_INITIAL;
                            }
                            Err(error) => {
                                tracing::warn!(%error, ?backoff, "Redis subscribe failed, retrying");
                                tokio::time::sleep(backoff).await;
                                backoff = (backoff * 2).min(BACKOFF_MAX);
                            }
                        }
                        continue;
                    };
                    match current.next().await {
                        Some(message) => {
                            let payload = message.get_payload_bytes().to_vec();
                            return Some((payload, (subscriber, channel, live, backoff)));
                        }
                        None => live = None,
                    }
                    tracing::warn!(?backoff, "Redis subscription connection lost, resubscribing");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(BACKOFF_MAX);
                }
            },
        )
        .boxed()
    }

    async fn subscribe(&self, channel: &str) -> RedisResult<PubSubStream> {
        match &self.source {
            Source::Sentinel(sentinel) => {
                let master = sentinel.master().await?;
                self.subscribe_on(master.get_connection_info().clone(), channel)
                    .await
            }
            Source::Nodes(nodes) => {
                let mut last = None;
                for node in nodes.iter() {
                    match self.subscribe_on(node.clone(), channel).await {
                        Ok(live) => return Ok(live),
                        Err(error) => last = Some(error),
                    }
                }
                Err(last.expect("an endpoint always has at least one node"))
            }
        }
    }

    async fn subscribe_on(&self, info: ConnectionInfo, channel: &str) -> RedisResult<PubSubStream> {
        let info = self.authenticated(info).await?;
        let client = redis::Client::open(info)?;
        let mut pubsub = tokio::time::timeout(self.connect_timeout, client.get_async_pubsub())
            .await
            .map_err(|_| {
                redis::RedisError::from(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "connecting the Redis subscription timed out",
                ))
            })??;
        pubsub.subscribe(channel).await?;
        Ok(pubsub.into_on_message())
    }

    // redis-py's pubsub connection asks the credential provider once, on connect
    async fn authenticated(&self, info: ConnectionInfo) -> RedisResult<ConnectionInfo> {
        let Credentials::Provider(provider) = &self.credentials else {
            return Ok(info);
        };
        let current = provider.subscribe().next().await.ok_or_else(|| {
            redis::RedisError::from((
                redis::ErrorKind::AuthenticationFailed,
                "the Redis credential provider ended before giving credentials",
            ))
        })??;
        let settings = info
            .redis_settings()
            .clone()
            .set_username(current.username())
            .set_password(current.password());
        Ok(info.set_redis_settings(settings))
    }
}
