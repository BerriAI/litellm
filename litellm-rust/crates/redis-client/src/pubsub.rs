use std::{sync::Arc, time::Duration};

use futures_util::{StreamExt, stream::BoxStream};
use redis::{
    AsyncConnectionConfig, ConnectionInfo, Msg, ProtocolVersion, PushInfo, PushKind, RedisResult,
    aio::MultiplexedConnection,
};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use crate::{
    ConnectError, Credentials, Endpoint, HostEndpoint, Timeouts, config,
    connection::{self, Connection, SentinelConnection},
};

// litellm/proxy/common_utils/auth_cache_invalidation_pubsub.py: _BACKOFF_INITIAL_SECONDS, _BACKOFF_MAX_SECONDS
const BACKOFF_INITIAL: Duration = Duration::from_secs(5);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub(crate) struct Subscriber {
    source: Source,
    config: AsyncConnectionConfig,
}

#[derive(Clone)]
enum Source {
    Nodes(Arc<[redis::Client]>),
    Sentinel(SentinelConnection),
}

struct Live {
    _connection: MultiplexedConnection,
    pushes: UnboundedReceiver<PushInfo>,
}

impl Subscriber {
    pub(crate) fn new(
        endpoint: &Endpoint,
        credentials: &Credentials,
        timeouts: Timeouts,
        connection: &Connection,
    ) -> Result<Self, ConnectError> {
        let source = match (endpoint, connection) {
            (_, Connection::Sentinel(sentinel)) => Source::Sentinel(sentinel.clone()),
            (Endpoint::Cluster(cluster), _) => Source::Nodes(
                cluster
                    .nodes
                    .iter()
                    .map(|node| {
                        let node = Endpoint::Host(HostEndpoint {
                            host: node.host.clone(),
                            port: node.port,
                            tls: cluster.tls.clone(),
                            db: 0,
                            unix_socket_path: None,
                        });
                        Ok(resp3_client(config::connection_info(&node, credentials)?)?)
                    })
                    .collect::<Result<_, ConnectError>>()?,
            ),
            _ => Source::Nodes(Arc::new([resp3_client(config::connection_info(
                endpoint,
                credentials,
            )?)?])),
        };
        Ok(Self {
            source,
            config: connection::async_config(credentials, timeouts),
        })
    }

    pub(crate) fn messages(self, channel: String) -> BoxStream<'static, Vec<u8>> {
        futures_util::stream::unfold(
            (self, channel, None::<Live>, BACKOFF_INITIAL),
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
                    match current.pushes.recv().await {
                        Some(push) if push.kind == PushKind::Disconnection => live = None,
                        None => live = None,
                        Some(push) => {
                            if let Some(message) = Msg::from_push_info(push) {
                                let payload = message.get_payload_bytes().to_vec();
                                return Some((payload, (subscriber, channel, live, backoff)));
                            }
                            continue;
                        }
                    }
                    tracing::warn!(?backoff, "Redis subscription connection lost, resubscribing");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(BACKOFF_MAX);
                }
            },
        )
        .boxed()
    }

    async fn subscribe(&self, channel: &str) -> RedisResult<Live> {
        match &self.source {
            Source::Sentinel(sentinel) => {
                let master = sentinel.master().await?;
                self.subscribe_on(
                    &resp3_client(master.get_connection_info().clone())?,
                    channel,
                )
                .await
            }
            Source::Nodes(nodes) => {
                let mut last = None;
                for node in nodes.iter() {
                    match self.subscribe_on(node, channel).await {
                        Ok(live) => return Ok(live),
                        Err(error) => last = Some(error),
                    }
                }
                Err(last.expect("an endpoint always has at least one node"))
            }
        }
    }

    async fn subscribe_on(&self, client: &redis::Client, channel: &str) -> RedisResult<Live> {
        let (sender, pushes) = unbounded_channel();
        let config = self.config.clone().set_push_sender(sender);
        let mut connection = client
            .get_multiplexed_async_connection_with_config(&config)
            .await?;
        connection.subscribe(channel).await?;
        Ok(Live {
            _connection: connection,
            pushes,
        })
    }
}

fn resp3_client(info: ConnectionInfo) -> RedisResult<redis::Client> {
    let settings = info
        .redis_settings()
        .clone()
        .set_protocol(ProtocolVersion::RESP3);
    redis::Client::open(info.set_redis_settings(settings))
}
