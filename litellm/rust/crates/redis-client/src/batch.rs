use std::{
    io,
    sync::{Arc, Mutex, MutexGuard},
};

use futures_util::future::join_all;
use redis::{RedisError, Value};
use tokio::sync::oneshot::{self, error::TryRecvError};

use crate::{Command, Error, Pipeline, RedisClient, commands::Op};

/// Ops declared by many callers and sent together, like litellm's `RedisBatch`: one pipeline on a
/// standalone or Sentinel Redis, and each op on its own on a cluster or in an alone batch.
#[derive(Clone)]
pub struct RedisBatch(Arc<Shared>);

#[derive(Clone, Copy)]
enum Mode {
    /// litellm's `RedisBatch`: one pipeline, or each op on its own on a cluster.
    Batch,
    /// Each op on its own, the way the `RedisCache` method behind it sends it.
    Alone,
    /// One redis-py pipeline on every topology, split by slot on a cluster.
    Pipeline,
}

struct Shared {
    client: RedisClient,
    mode: Mode,
    pending: Mutex<Pending>,
    flushing: tokio::sync::Mutex<()>,
}

#[derive(Default)]
struct Pending {
    ops: Vec<Op>,
    replies: Vec<oneshot::Sender<Result<Value, Error>>>,
}

type Decode<T> = Box<dyn FnOnce(Value) -> Result<T, Error> + Send>;

#[must_use]
pub struct Declared<T> {
    batch: RedisBatch,
    reply: oneshot::Receiver<Result<Value, Error>>,
    decode: Decode<T>,
}

impl RedisBatch {
    pub(crate) fn new(client: RedisClient) -> Self {
        Self::build(client, Mode::Batch)
    }

    /// A batch whose every op goes out on its own, the way the `RedisCache` method behind it sends
    /// it, so a fresh one is "send it now": the first `resolve` sends it.
    pub fn alone(client: RedisClient) -> Self {
        Self::build(client, Mode::Alone)
    }

    /// A batch sent as one redis-py pipeline, like `async_increment_pipeline`, on every topology.
    pub fn pipelined(client: RedisClient) -> Self {
        Self::build(client, Mode::Pipeline)
    }

    fn build(client: RedisClient, mode: Mode) -> Self {
        Self(Arc::new(Shared {
            client,
            mode,
            pending: Mutex::new(Pending::default()),
            flushing: tokio::sync::Mutex::new(()),
        }))
    }

    pub fn add<T>(&self, command: Command<T>) -> Declared<T> {
        let (op, decode) = command.prepare(self.0.client.namespace());
        let (sender, reply) = oneshot::channel();
        let mut pending = self.pending();
        pending.ops.push(op);
        pending.replies.push(sender);
        Declared {
            batch: self.clone(),
            reply,
            decode,
        }
    }

    pub async fn flush(&self) {
        let _flushing = self.0.flushing.lock().await;
        let Pending { ops, replies } = std::mem::take(&mut *self.pending());
        if replies.is_empty() {
            return;
        }
        let client = &self.0.client;
        let alone = match self.0.mode {
            Mode::Alone => true,
            Mode::Batch => client.is_cluster(),
            Mode::Pipeline => false,
        };
        let results = match alone {
            true => join_all(ops.into_iter().map(|op| client.run_alone(op))).await,
            false => {
                client
                    .run(&Pipeline::of(client.namespace().clone(), ops))
                    .await
            }
        };
        for (reply, result) in replies.into_iter().zip(results) {
            let _ = reply.send(result);
        }
    }

    fn pending(&self) -> MutexGuard<'_, Pending> {
        self.0
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl<T> Declared<T> {
    pub async fn resolve(mut self) -> Result<T, Error> {
        match self.reply.try_recv() {
            Ok(reply) => return reply.and_then(self.decode),
            Err(TryRecvError::Closed) => return Err(abandoned()),
            Err(TryRecvError::Empty) => {}
        }
        self.batch.flush().await;
        self.settled().await
    }

    pub async fn settled(self) -> Result<T, Error> {
        let reply = self.reply.await.unwrap_or_else(|_| Err(abandoned()));
        reply.and_then(self.decode)
    }

    /// Every reply in order, flushing their batch once if nobody has yet.
    pub async fn all(declared: impl IntoIterator<Item = Self>) -> Vec<Result<T, Error>> {
        join_all(declared.into_iter().map(Self::resolve)).await
    }
}

fn abandoned() -> Error {
    Error::Transport(RedisError::from(io::Error::new(
        io::ErrorKind::Interrupted,
        "the batch flush carrying this op was cancelled",
    )))
}
