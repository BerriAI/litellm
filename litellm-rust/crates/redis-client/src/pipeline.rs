use std::{collections::HashMap, time::Duration};

use futures_util::future::join_all;
use redis::{
    Cmd, ErrorKind, RedisError, RedisResult, ServerErrorKind, ToRedisArgs, ToSingleRedisArg, Value,
    cluster_routing::Slot,
};

use crate::{
    Amount, Error, Namespace, Script, commands, connection::Connection, script::ScriptCall,
};

pub struct Pipeline {
    namespace: Namespace,
    ops: Vec<Op>,
}

struct Op {
    entry: Entry,
    key: String,
    idempotent: bool,
}

/// What a pipeline sends for one op. Only a [`ScriptCall`] can be sent again, so a retry can never
/// repeat a command that already ran.
enum Entry {
    /// The op's reply is the first command's, like `RedisBatch`'s `INCRBYFLOAT` plus `EXPIRE`.
    Commands(Vec<Cmd>),
    Script(ScriptCall),
}

impl Entry {
    fn commands(&self) -> Vec<Cmd> {
        match self {
            Self::Commands(commands) => commands.clone(),
            Self::Script(call) => vec![call.script.evalsha(&call.keys, &call.args)],
        }
    }
}

type Settled = (usize, Result<Value, Error>);

impl Pipeline {
    pub(crate) fn new(namespace: Namespace) -> Self {
        Self {
            namespace,
            ops: Vec::new(),
        }
    }

    pub fn get(self, key: &str) -> Self {
        let key = self.namespace.apply(key);
        self.commands(vec![commands::get(key.clone())], key, true)
    }

    /// One `MGET`, so every key must share a hash slot on a cluster. Only `RedisBatch` sends it,
    /// and a batch never pipelines on a cluster.
    pub(crate) fn mget(self, keys: &[&str]) -> Self {
        let keys: Vec<String> = keys.iter().map(|key| self.namespace.apply(key)).collect();
        let key = keys.first().cloned().unwrap_or_default();
        self.commands(vec![commands::mget(keys)], key, true)
    }

    pub fn set(self, key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Self {
        let key = self.namespace.apply(key);
        self.commands(vec![commands::set(key.clone(), value, ttl)], key, true)
    }

    pub fn del(self, key: &str) -> Self {
        let key = self.namespace.apply(key);
        self.commands(vec![commands::del(vec![key.clone()])], key, true)
    }

    pub fn expire(self, key: &str, ttl: Duration) -> Self {
        let key = self.namespace.apply(key);
        self.commands(vec![commands::expire(key.clone(), ttl)], key, true)
    }

    /// `INCRBYFLOAT`, then `EXPIRE` when a TTL is given, like `RedisBatch` and
    /// `async_increment_pipeline` send it.
    pub fn incr_by_float(self, key: &str, amount: Amount, ttl: Option<Duration>) -> Self {
        let key = self.namespace.apply(key);
        let increment = commands::incr_by_float(key.clone(), amount);
        let commands = match ttl {
            Some(ttl) => vec![increment, commands::expire(key.clone(), ttl)],
            None => vec![increment],
        };
        self.commands(commands, key, false)
    }

    pub fn eval(self, script: &Script, keys: &[&str], args: impl ToRedisArgs) -> Self {
        let keys: Vec<String> = keys.iter().map(|key| self.namespace.apply(key)).collect();
        let key = keys.first().cloned().unwrap_or_default();
        let call = ScriptCall {
            script: script.clone(),
            keys,
            args: args.to_redis_args(),
        };
        self.push(Entry::Script(call), key, false)
    }

    fn commands(self, commands: Vec<Cmd>, key: String, idempotent: bool) -> Self {
        self.push(Entry::Commands(commands), key, idempotent)
    }

    fn push(mut self, entry: Entry, key: String, idempotent: bool) -> Self {
        self.ops.push(Op {
            entry,
            key,
            idempotent,
        });
        self
    }

    pub(crate) fn len(&self) -> usize {
        self.ops.len()
    }

    pub(crate) async fn run(&self, connection: &Connection) -> Vec<Result<Value, Error>> {
        let groups = self.groups(connection);
        let settled = join_all(
            groups
                .iter()
                .map(|batches| self.run_in_order(connection, batches)),
        )
        .await;
        let mut settled: Vec<Settled> = settled.into_iter().flatten().collect();
        settled.sort_by_key(|(index, _)| *index);
        settled.into_iter().map(|(_, result)| result).collect()
    }

    async fn run_in_order(&self, connection: &Connection, batches: &[Vec<usize>]) -> Vec<Settled> {
        let mut settled = Vec::with_capacity(batches.iter().map(Vec::len).sum());
        for batch in batches {
            settled.extend(self.run_group(connection.clone(), batch).await);
        }
        settled
    }

    fn groups(&self, connection: &Connection) -> Vec<Vec<Vec<usize>>> {
        match connection {
            _ if self.ops.is_empty() => Vec::new(),
            Connection::Standalone(_) | Connection::Sentinel(_) => {
                vec![vec![(0..self.ops.len()).collect()]]
            }
            Connection::Cluster(_) => {
                let mut by_slot: HashMap<Slot, Vec<usize>> = HashMap::new();
                for index in 0..self.ops.len() {
                    by_slot
                        .entry(Slot::for_key(&self.ops[index].key))
                        .or_default()
                        .push(index);
                }
                by_slot
                    .into_values()
                    .map(|indices| self.retry_safe_batches(indices))
                    .collect()
            }
        }
    }

    fn retry_safe_batches(&self, indices: Vec<usize>) -> Vec<Vec<usize>> {
        indices
            .chunk_by(|&left, &right| self.ops[left].idempotent && self.ops[right].idempotent)
            .map(<[usize]>::to_vec)
            .collect()
    }

    /// A script the server had not cached ran nothing, so it alone runs again, the way litellm's
    /// `RedisBatch` reruns it through redis-py's `Script`: `EVALSHA`, then `SCRIPT LOAD` and
    /// `EVALSHA` on `NOSCRIPT`.
    async fn settle_script(
        &self,
        connection: Connection,
        index: usize,
        reply: RedisResult<Value>,
    ) -> (usize, RedisResult<Value>) {
        match (&self.ops[index].entry, reply) {
            (Entry::Script(call), Err(error)) if is_missing_script(&error) => {
                (index, rerun_missing_script(connection, call).await)
            }
            (_, reply) => (index, reply),
        }
    }

    async fn run_group(&self, mut connection: Connection, indices: &[usize]) -> Vec<Settled> {
        let deployment = connection.deployment();
        let mut pipeline = redis::pipe();
        pipeline.ignore_errors();
        let widths: Vec<usize> = indices
            .iter()
            .map(|&index| {
                let commands = self.ops[index].entry.commands();
                let width = commands.len();
                for command in commands {
                    pipeline.add_command(command);
                }
                width
            })
            .collect();
        let replies: RedisResult<Vec<RedisResult<Value>>> =
            pipeline.query_async(&mut connection).await;
        let replies = match replies {
            Ok(replies) => replies,
            Err(error) => {
                return indices
                    .iter()
                    .map(|&index| (index, Err(Error::classify(error.clone(), deployment))))
                    .collect();
            }
        };
        let mut replies = replies.into_iter();
        let per_op: Vec<(usize, RedisResult<Value>)> = indices
            .iter()
            .zip(widths)
            .map(|(&index, width)| (index, first_reply(replies.by_ref().take(width))))
            .collect();
        join_all(
            per_op
                .into_iter()
                .map(|(index, reply)| self.settle_script(connection.clone(), index, reply)),
        )
        .await
        .into_iter()
        .map(|(index, reply)| {
            (
                index,
                reply.map_err(|error| Error::classify(error, deployment)),
            )
        })
        .collect()
    }
}

async fn rerun_missing_script(mut connection: Connection, call: &ScriptCall) -> RedisResult<Value> {
    call.script
        .invocation(&call.keys, &call.args)
        .invoke_async(&mut connection)
        .await
}

fn is_missing_script(error: &RedisError) -> bool {
    error.kind() == ErrorKind::Server(ServerErrorKind::NoScript)
}

// RedisBatch settles an op from its first reply and fails it on any reply that failed
fn first_reply(replies: impl Iterator<Item = RedisResult<Value>>) -> RedisResult<Value> {
    replies
        .reduce(|first, next| match (first, next) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (first, Ok(_)) => first,
        })
        .unwrap_or(Ok(Value::Nil))
}
