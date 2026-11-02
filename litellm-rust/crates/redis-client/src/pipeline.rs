use std::{collections::HashMap, time::Duration};

use futures_util::future::join_all;
use redis::{
    Cmd, FromRedisValue, RedisResult, ToRedisArgs, ToSingleRedisArg, Value, cluster_routing::Slot,
};

use crate::{
    Error, Expiry, Namespace, Script,
    commands::{self, ScriptCall},
    connection::Connection,
};

pub struct Pipeline {
    namespace: Namespace,
    ops: Vec<Op>,
}

struct Op {
    command: Cmd,
    key: String,
    idempotent: bool,
    integer_reply: bool,
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
        self.command(commands::get(key.clone()), key, true, false)
    }

    pub fn set(self, key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::set(key.clone(), value, ttl), key, true, false)
    }

    pub fn del(self, key: &str) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::del(vec![key.clone()]), key, true, false)
    }

    pub fn expire(self, key: &str, ttl: Duration) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::expire(key.clone(), ttl), key, true, false)
    }

    pub fn incr_by(self, key: &str, delta: i64, expiry: Expiry) -> Self {
        let key = self.namespace.apply(key);
        self.script_call(commands::incr_by(key, delta, expiry), true)
    }

    pub fn incr_by_float(self, key: &str, delta: f64, expiry: Expiry) -> Self {
        let key = self.namespace.apply(key);
        self.script_call(commands::incr_by_float(key, delta, expiry), false)
    }

    pub fn rpush(self, key: &str, values: impl ToRedisArgs, expiry: Expiry) -> Self {
        let key = self.namespace.apply(key);
        self.script_call(commands::rpush(key, values, expiry), false)
    }

    pub fn eval(self, script: &Script, keys: &[&str], args: impl ToRedisArgs) -> Self {
        let keys: Vec<String> = keys.iter().map(|key| self.namespace.apply(key)).collect();
        let key = keys.first().cloned().unwrap_or_default();
        self.command(script.eval(&keys, &args.to_redis_args()), key, false, false)
    }

    fn script_call(self, call: ScriptCall, integer_reply: bool) -> Self {
        let key = call.keys.first().cloned().unwrap_or_default();
        self.command(
            call.script.eval(&call.keys, &call.args),
            key,
            false,
            integer_reply,
        )
    }

    fn command(mut self, command: Cmd, key: String, idempotent: bool, integer_reply: bool) -> Self {
        self.ops.push(Op {
            command,
            key,
            idempotent,
            integer_reply,
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

    async fn run_group(&self, mut connection: Connection, indices: &[usize]) -> Vec<Settled> {
        let deployment = connection.deployment();
        let mut pipeline = redis::pipe();
        pipeline.ignore_errors();
        for &index in indices {
            pipeline.add_command(self.ops[index].command.clone());
        }
        let replies: RedisResult<Vec<RedisResult<Value>>> =
            pipeline.query_async(&mut connection).await;
        match replies {
            Ok(replies) => indices
                .iter()
                .zip(replies)
                .map(|(&index, reply)| {
                    (
                        index,
                        reply
                            .map_err(|error| Error::classify(error, deployment))
                            .and_then(|value| {
                                if self.ops[index].integer_reply {
                                    i64::from_redis_value(value)
                                        .map(Value::Int)
                                        .map_err(|error| Error::Decode(error.into()))
                                } else {
                                    Ok(value)
                                }
                            }),
                    )
                })
                .collect(),
            Err(error) => indices
                .iter()
                .map(|&index| (index, Err(Error::classify(error.clone(), deployment))))
                .collect(),
        }
    }
}
