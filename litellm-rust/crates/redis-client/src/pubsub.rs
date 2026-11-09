use std::{sync::Arc, time::Duration};

use futures_util::{StreamExt, stream::BoxStream};
use redis::{
    ConnectionInfo, RedisResult,
    aio::{PubSubSink, PubSubStream},
};

use crate::{
    ConnectError, Credentials, Standalone, Target, Timeouts, config,
    connection::{Connection, SentinelConnection},
};

// litellm/proxy/common_utils/auth_cache_invalidation_pubsub.py: _BACKOFF_INITIAL_SECONDS, _BACKOFF_MAX_SECONDS
const BACKOFF_INITIAL: Duration = Duration::from_secs(5);
const BACKOFF_MAX: Duration = Duration::from_secs(60);
const IDLE_PROBE: Duration = Duration::from_secs(15);

pub enum SubscriptionEvent {
    Connected,
    Disconnected,
    Message(Vec<u8>),
}

struct Live {
    sink: PubSubSink,
    stream: PubSubStream,
}

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
        keepalive: bool,
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
                        config::connection_info(&node, credentials, keepalive)
                    })
                    .collect::<Result<_, ConnectError>>()?,
            ),
            _ => Source::Nodes(Arc::new([config::connection_info(
                endpoint,
                credentials,
                keepalive,
            )?])),
        };
        Ok(Self {
            source,
            credentials: credentials.clone(),
            connect_timeout: timeouts.connect,
        })
    }

    pub(crate) fn messages(self, channel: String) -> BoxStream<'static, Vec<u8>> {
        self.events(channel)
            .filter_map(|event| async move {
                match event {
                    SubscriptionEvent::Message(payload) => Some(payload),
                    _ => None,
                }
            })
            .boxed()
    }

    pub(crate) fn events(self, channel: String) -> BoxStream<'static, SubscriptionEvent> {
        futures_util::stream::unfold(
            (self, channel, None::<Live>, BACKOFF_INITIAL, false),
            |(subscriber, channel, live, backoff, retry)| async move {
                let mut live = live;
                let mut backoff = backoff;
                if retry {
                    tokio::time::sleep(backoff).await;
                }
                loop {
                    let Some(current) = live.as_mut() else {
                        let event = match subscriber.subscribe(&channel).await {
                            Ok(subscribed) => {
                                live = Some(subscribed);
                                backoff = BACKOFF_INITIAL;
                                SubscriptionEvent::Connected
                            }
                            Err(_) => {
                                if retry {
                                    backoff = (backoff * 2).min(BACKOFF_MAX);
                                }
                                SubscriptionEvent::Disconnected
                            }
                        };
                        let retry = live.is_none();
                        return Some((event, (subscriber, channel, live, backoff, retry)));
                    };
                    match tokio::time::timeout(IDLE_PROBE, current.stream.next()).await {
                        Ok(Some(message)) => {
                            let payload = message.get_payload_bytes().to_vec();
                            return Some((
                                SubscriptionEvent::Message(payload),
                                (subscriber, channel, live, backoff, false),
                            ));
                        }
                        Err(_) => {
                            if matches!(
                                tokio::time::timeout(
                                    subscriber.connect_timeout,
                                    current.sink.ping::<String>()
                                )
                                .await,
                                Ok(Ok(_))
                            ) {
                                continue;
                            }
                        }
                        Ok(None) => {}
                    }
                    return Some((
                        SubscriptionEvent::Disconnected,
                        (subscriber, channel, None, backoff, true),
                    ));
                }
            },
        )
        .boxed()
    }

    async fn subscribe(&self, channel: &str) -> RedisResult<Live> {
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

    async fn subscribe_on(&self, info: ConnectionInfo, channel: &str) -> RedisResult<Live> {
        let settings =
            config::authenticated(info.redis_settings().clone(), &self.credentials).await?;
        let client = redis::Client::open(info.set_redis_settings(settings))?;
        let mut pubsub = tokio::time::timeout(self.connect_timeout, client.get_async_pubsub())
            .await
            .map_err(|_| {
                redis::RedisError::from(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "connecting the Redis subscription timed out",
                ))
            })??;
        subscribe_connected(&mut pubsub, channel, self.connect_timeout).await?;
        let (sink, stream) = pubsub.split();
        Ok(Live { sink, stream })
    }
}

async fn subscribe_connected(
    pubsub: &mut redis::aio::PubSub,
    channel: &str,
    connect_timeout: Duration,
) -> RedisResult<()> {
    tokio::time::timeout(connect_timeout, pubsub.subscribe(channel))
        .await
        .map_err(|_| {
            redis::RedisError::from(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "subscribing to the Redis channel timed out",
            ))
        })?
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::StreamExt;
    use litellm_redis_testing::{RedisServer, TcpProxy};
    use rstest::rstest;

    use super::subscribe_connected;

    #[rstest]
    #[tokio::test]
    async fn a_stalled_pubsub_subscribe_times_out_and_a_fresh_pubsub_recovers() {
        let server = RedisServer::start().await.unwrap();
        let proxy = TcpProxy::start(server.port()).await.unwrap();
        let client = redis::Client::open(format!("redis://127.0.0.1:{}", proxy.port())).unwrap();
        let mut pubsub = tokio::time::timeout(Duration::from_secs(2), client.get_async_pubsub())
            .await
            .expect("connecting the initial pubsub should finish")
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            subscribe_connected(&mut pubsub, "existing", Duration::from_secs(1)),
        )
        .await
        .expect("the initial subscription acknowledgement should arrive")
        .unwrap();

        proxy.blackhole_open_connections();
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            subscribe_connected(&mut pubsub, "stalled", Duration::from_millis(100)),
        )
        .await
        .expect("the stalled subscription should return before the safety deadline");
        assert!(
            matches!(&result, Err(error) if error.is_timeout()),
            "{result:?}"
        );
        drop(pubsub);

        let mut recovered = tokio::time::timeout(Duration::from_secs(2), client.get_async_pubsub())
            .await
            .expect("a fresh pubsub connection should finish")
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            subscribe_connected(&mut recovered, "recovered", Duration::from_secs(1)),
        )
        .await
        .expect("the fresh subscription acknowledgement should arrive")
        .unwrap();

        let publisher = redis::Client::open(server.url()).unwrap();
        let mut connection = publisher.get_multiplexed_async_connection().await.unwrap();
        let subscriber_count: i64 = redis::cmd("PUBLISH")
            .arg("recovered")
            .arg("payload")
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(subscriber_count, 1);
        let message = tokio::time::timeout(Duration::from_secs(2), recovered.on_message().next())
            .await
            .expect("the fresh subscription should receive the published payload")
            .unwrap();
        assert_eq!(message.get_payload::<String>().unwrap(), "payload");
    }
}
