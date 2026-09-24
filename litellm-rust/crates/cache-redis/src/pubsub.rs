use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use litellm_cache::{CacheCodec, Error, Message, MessageStream, PubSubCache};
use tokio::sync::{mpsc, oneshot};

use crate::{cache::RedisCache, topology::RedisTopology};

const POLL_INTERVAL: Duration = Duration::from_secs(1);
const QUEUE_CAPACITY: usize = 1024;

/// Messages arrive on a reader thread that owns the subscribed connection and forwards them
/// into a bounded queue; the thread polls with `POLL_INTERVAL` so `close` and drop can stop it.
pub struct RedisSubscription {
    messages: mpsc::Receiver<Result<Message, Error>>,
    stop: Arc<AtomicBool>,
    reader: Option<thread::JoinHandle<()>>,
}

impl MessageStream for RedisSubscription {
    async fn next_message(&mut self, timeout: Option<Duration>) -> Result<Option<Message>, Error> {
        let next = match timeout {
            Some(limit) if limit.is_zero() => match self.messages.try_recv() {
                Ok(message) => Some(message),
                Err(mpsc::error::TryRecvError::Empty) => return Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => None,
            },
            Some(limit) => match tokio::time::timeout(limit, self.messages.recv()).await {
                Ok(message) => message,
                Err(_) => return Ok(None),
            },
            None => self.messages.recv().await,
        };
        match next {
            Some(Ok(message)) => Ok(Some(message)),
            Some(Err(error)) => Err(error),
            None => Err(Error::Unavailable),
        }
    }

    async fn close(mut self) -> Result<(), Error> {
        self.stop.store(true, Ordering::Release);
        let Some(reader) = self.reader.take() else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || reader.join())
            .await
            .map_err(|_| Error::Unavailable)?
            .map_err(|_| Error::Unavailable)
    }
}

impl Drop for RedisSubscription {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

impl<S, C> PubSubCache for RedisCache<S, C>
where
    S: CacheCodec,
    C: redis::ConnectionLike + Send + 'static,
{
    type Subscription = RedisSubscription;

    async fn async_publish(&self, channel: &str, payload: &[u8]) -> Result<usize, Error> {
        if !matches!(self.topology, RedisTopology::Standalone) {
            return Err(Error::UnsupportedOperation);
        }
        let channel = channel.to_owned();
        let payload = payload.to_vec();
        self.run(move |connection| {
            redis::cmd("PUBLISH")
                .arg(channel)
                .arg(payload)
                .query::<usize>(connection)
                .map_err(|_| Error::Unavailable)
        })
        .await
    }

    async fn async_subscribe(&self, channels: &[String]) -> Result<Self::Subscription, Error> {
        let connections = Arc::clone(&self.connections);
        let connection = tokio::task::spawn_blocking(move || {
            let connection = connections.subscription_connection()?;
            connection
                .set_read_timeout(Some(POLL_INTERVAL))
                .map_err(|_| Error::Unavailable)?;
            Ok::<_, Error>(connection)
        })
        .await
        .map_err(|_| Error::Unavailable)??;
        let (subscribed, ready) = oneshot::channel();
        let (sender, messages) = mpsc::channel(QUEUE_CAPACITY);
        let stop = Arc::new(AtomicBool::new(false));
        let reader = thread::Builder::new()
            .name("litellm-redis-subscription".into())
            .spawn({
                let stop = Arc::clone(&stop);
                let channels = channels.to_vec();
                move || read_messages(connection, &channels, subscribed, &sender, &stop)
            })
            .map_err(|_| Error::Unavailable)?;
        ready.await.map_err(|_| Error::Unavailable)??;
        Ok(RedisSubscription {
            messages,
            stop,
            reader: Some(reader),
        })
    }
}

fn read_messages(
    mut connection: redis::Connection,
    channels: &[String],
    subscribed: oneshot::Sender<Result<(), Error>>,
    sender: &mpsc::Sender<Result<Message, Error>>,
    stop: &AtomicBool,
) {
    let mut pubsub = connection.as_pubsub();
    let outcome = channels
        .iter()
        .try_for_each(|channel| pubsub.subscribe(channel))
        .map_err(|_| Error::Unavailable);
    let failed = outcome.is_err();
    let _ = subscribed.send(outcome);
    if failed {
        return;
    }
    while !stop.load(Ordering::Acquire) {
        match pubsub.get_message() {
            Ok(received) => {
                let message = Message {
                    channel: received.get_channel_name().to_owned(),
                    payload: received.get_payload_bytes().to_vec(),
                };
                if sender.blocking_send(Ok(message)).is_err() {
                    return;
                }
            }
            Err(error) if error.is_timeout() => continue,
            Err(_) => {
                let _ = sender.blocking_send(Err(Error::Unavailable));
                return;
            }
        }
    }
}
