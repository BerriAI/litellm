use std::{
    io,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use redis::{FromRedisValue, RedisError, ToRedisArgs, ToSingleRedisArg, Value};
use tokio::sync::oneshot::{self, error::TryRecvError};

use crate::{Error, Expiry, Pipeline, RedisClient, Script};

#[derive(Clone)]
pub struct RedisBatch(Arc<Shared>);

struct Shared {
    client: RedisClient,
    pending: Mutex<Pending>,
    flushing: tokio::sync::Mutex<()>,
}

struct Pending {
    pipeline: Pipeline,
    replies: Vec<oneshot::Sender<Result<Value, Error>>>,
}

#[must_use]
pub struct Declared<T> {
    batch: RedisBatch,
    reply: oneshot::Receiver<Result<Value, Error>>,
    decode: fn(Value) -> Result<T, Error>,
}

impl RedisBatch {
    pub(crate) fn new(client: RedisClient) -> Self {
        let pending = Pending::new(&client);
        Self(Arc::new(Shared {
            client,
            pending: Mutex::new(pending),
            flushing: tokio::sync::Mutex::new(()),
        }))
    }

    pub fn get<T: FromRedisValue>(&self, key: &str) -> Declared<Option<T>> {
        self.declare(|pipeline| pipeline.get(key), decode)
    }

    pub fn set(&self, key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Declared<()> {
        self.declare(|pipeline| pipeline.set(key, value, ttl), decode)
    }

    pub fn del(&self, key: &str) -> Declared<usize> {
        self.declare(|pipeline| pipeline.del(key), decode)
    }

    pub fn expire(&self, key: &str, ttl: Duration) -> Declared<bool> {
        self.declare(|pipeline| pipeline.expire(key, ttl), decode)
    }

    pub fn incr_by(&self, key: &str, delta: i64, expiry: Expiry) -> Declared<i64> {
        self.declare(|pipeline| pipeline.incr_by(key, delta, expiry), decode)
    }

    pub fn incr_by_float(&self, key: &str, delta: f64, expiry: Expiry) -> Declared<f64> {
        self.declare(
            |pipeline| pipeline.incr_by_float(key, delta, expiry),
            decode,
        )
    }

    pub fn rpush(&self, key: &str, values: impl ToRedisArgs, expiry: Expiry) -> Declared<usize> {
        self.declare(|pipeline| pipeline.rpush(key, values, expiry), decode)
    }

    pub fn eval<T: FromRedisValue>(
        &self,
        script: &Script,
        keys: &[&str],
        args: impl ToRedisArgs,
    ) -> Declared<T> {
        self.declare(|pipeline| pipeline.eval(script, keys, args), decode)
    }

    pub async fn flush(&self) {
        let _flushing = self.0.flushing.lock().await;
        let Pending { pipeline, replies } =
            std::mem::replace(&mut *self.pending(), Pending::new(&self.0.client));
        if replies.is_empty() {
            return;
        }
        let results = self.0.client.run(&pipeline).await;
        for (reply, result) in replies.into_iter().zip(results) {
            let _ = reply.send(result);
        }
    }

    fn declare<T>(
        &self,
        add: impl FnOnce(Pipeline) -> Pipeline,
        decode: fn(Value) -> Result<T, Error>,
    ) -> Declared<T> {
        let (sender, reply) = oneshot::channel();
        let mut pending = self.pending();
        let pipeline = std::mem::replace(&mut pending.pipeline, self.0.client.pipeline());
        pending.pipeline = add(pipeline);
        pending.replies.push(sender);
        Declared {
            batch: self.clone(),
            reply,
            decode,
        }
    }

    fn pending(&self) -> MutexGuard<'_, Pending> {
        self.0
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Pending {
    fn new(client: &RedisClient) -> Self {
        Self {
            pipeline: client.pipeline(),
            replies: Vec::new(),
        }
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
}

fn decode<T: FromRedisValue>(value: Value) -> Result<T, Error> {
    T::from_redis_value(value).map_err(|error| Error::Decode(error.into()))
}

fn abandoned() -> Error {
    Error::Transport(RedisError::from(io::Error::new(
        io::ErrorKind::Interrupted,
        "the batch flush carrying this op was cancelled",
    )))
}
