use std::{
    io,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use futures_util::future::join_all;
use redis::{FromRedisValue, RedisError, ToRedisArgs, ToSingleRedisArg, Value};
use tokio::sync::oneshot::{self, error::TryRecvError};

use crate::{Amount, Error, Expiry, Pipeline, RedisClient, Script};

/// Ops declared by many callers and sent together, like litellm's `RedisBatch`: one pipeline on a
/// standalone or Sentinel Redis, and each op on its own on a cluster.
#[derive(Clone)]
pub struct RedisBatch(Arc<Shared>);

struct Shared {
    client: RedisClient,
    pending: Mutex<Pending>,
    flushing: tokio::sync::Mutex<()>,
}

#[derive(Default)]
struct Pending {
    ops: Vec<BatchOp>,
    replies: Vec<oneshot::Sender<Result<Value, Error>>>,
}

/// One declared op, kept as data so a flush can pipeline it or run it alone.
enum BatchOp {
    Get(String),
    MGet(Vec<String>),
    Set {
        key: String,
        value: Vec<u8>,
        ttl: Duration,
    },
    Del(String),
    Expire {
        key: String,
        ttl: Duration,
    },
    IncrByFloat {
        key: String,
        amount: Amount,
        ttl: Option<Duration>,
    },
    Eval {
        script: Script,
        keys: Vec<String>,
        args: Vec<Vec<u8>>,
    },
}

#[must_use]
pub struct Declared<T> {
    batch: RedisBatch,
    reply: oneshot::Receiver<Result<Value, Error>>,
    decode: fn(Value) -> Result<T, Error>,
}

impl RedisBatch {
    pub(crate) fn new(client: RedisClient) -> Self {
        Self(Arc::new(Shared {
            client,
            pending: Mutex::new(Pending::default()),
            flushing: tokio::sync::Mutex::new(()),
        }))
    }

    pub fn get<T: FromRedisValue>(&self, key: &str) -> Declared<Option<T>> {
        self.declare(BatchOp::Get(key.to_owned()), decode)
    }

    /// One `MGET` of `keys` in order, repeated keys read once, like `RedisBatch.mget`.
    pub fn mget<T: FromRedisValue>(&self, keys: &[&str]) -> Declared<Vec<Option<T>>> {
        let mut unique: Vec<String> = Vec::with_capacity(keys.len());
        for key in keys {
            if !unique.iter().any(|seen| seen == key) {
                unique.push((*key).to_owned());
            }
        }
        self.declare(BatchOp::MGet(unique), decode)
    }

    pub fn set(&self, key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Declared<()> {
        let value = value.to_redis_args().concat();
        self.declare(
            BatchOp::Set {
                key: key.to_owned(),
                value,
                ttl,
            },
            decode,
        )
    }

    pub fn del(&self, key: &str) -> Declared<usize> {
        self.declare(BatchOp::Del(key.to_owned()), decode)
    }

    pub fn expire(&self, key: &str, ttl: Duration) -> Declared<bool> {
        self.declare(
            BatchOp::Expire {
                key: key.to_owned(),
                ttl,
            },
            decode,
        )
    }

    /// `RedisBatch.increment`: `INCRBYFLOAT`, then `EXPIRE` when a TTL is given.
    pub fn incr_by_float(&self, key: &str, amount: Amount, ttl: Option<Duration>) -> Declared<f64> {
        self.declare(
            BatchOp::IncrByFloat {
                key: key.to_owned(),
                amount,
                ttl,
            },
            decode,
        )
    }

    pub fn eval<T: FromRedisValue>(
        &self,
        script: &Script,
        keys: &[&str],
        args: impl ToRedisArgs,
    ) -> Declared<T> {
        self.declare(
            BatchOp::Eval {
                script: script.clone(),
                keys: keys.iter().map(|key| (*key).to_owned()).collect(),
                args: args.to_redis_args(),
            },
            decode,
        )
    }

    pub async fn flush(&self) {
        let _flushing = self.0.flushing.lock().await;
        let Pending { ops, replies } = std::mem::take(&mut *self.pending());
        if replies.is_empty() {
            return;
        }
        let client = &self.0.client;
        let results = match client.is_cluster() {
            true => join_all(ops.iter().map(|op| op.run_alone(client))).await,
            false => client.run(&pipeline(client, &ops)).await,
        };
        for (reply, result) in replies.into_iter().zip(results) {
            let _ = reply.send(result);
        }
    }

    fn declare<T>(&self, op: BatchOp, decode: fn(Value) -> Result<T, Error>) -> Declared<T> {
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

    fn pending(&self) -> MutexGuard<'_, Pending> {
        self.0
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn pipeline(client: &RedisClient, ops: &[BatchOp]) -> Pipeline {
    ops.iter().fold(client.pipeline(), |pipeline, op| match op {
        BatchOp::Get(key) => pipeline.get(key),
        BatchOp::MGet(keys) => {
            let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
            pipeline.mget(&keys)
        }
        BatchOp::Set { key, value, ttl } => pipeline.set(key, value.as_slice(), *ttl),
        BatchOp::Del(key) => pipeline.del(key),
        BatchOp::Expire { key, ttl } => pipeline.expire(key, *ttl),
        BatchOp::IncrByFloat { key, amount, ttl } => pipeline.incr_by_float(key, *amount, *ttl),
        BatchOp::Eval { script, keys, args } => {
            let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
            pipeline.eval(script, &keys, args.as_slice())
        }
    })
}

impl BatchOp {
    /// What `RedisBatch` runs for an op on a cluster, where it never pipelines: the `RedisCache`
    /// method behind it, so an increment there keeps a TTL it already has like `async_increment`.
    async fn run_alone(&self, client: &RedisClient) -> Result<Value, Error> {
        match self {
            Self::Get(key) => client
                .get::<Value>(key)
                .await
                .map(|value| value.unwrap_or(Value::Nil)),
            Self::MGet(keys) => {
                let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
                let values = client.mget::<Value>(&keys).await?;
                Ok(Value::Array(
                    values
                        .into_iter()
                        .map(|value| value.unwrap_or(Value::Nil))
                        .collect(),
                ))
            }
            Self::Set { key, value, ttl } => client
                .set(key, value.as_slice(), *ttl)
                .await
                .map(|()| Value::Okay),
            Self::Del(key) => client
                .del(&[key])
                .await
                .map(|deleted| Value::Int(i64::try_from(deleted).unwrap_or(i64::MAX))),
            Self::Expire { key, ttl } => client
                .expire(key, *ttl)
                .await
                .map(|set| Value::Int(i64::from(set))),
            Self::IncrByFloat { key, amount, ttl } => {
                let value = match ttl {
                    Some(ttl) => {
                        client
                            .incr_by_float(key, *amount, Expiry::KeepExisting(*ttl))
                            .await?
                    }
                    None => client.incr_by_float_without_ttl(key, *amount).await?,
                };
                Ok(Value::Double(value))
            }
            Self::Eval { script, keys, args } => {
                let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
                client.eval::<Value>(script, &keys, args.as_slice()).await
            }
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
