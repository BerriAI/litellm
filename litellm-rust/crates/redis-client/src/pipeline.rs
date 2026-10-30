use std::{collections::HashMap, time::Duration};

use futures_util::future::join_all;
use redis::{Cmd, RedisResult, ToRedisArgs, ToSingleRedisArg, Value, cluster_routing::Slot};

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
        self.command(commands::get(key.clone()), key)
    }

    pub fn set(self, key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::set(key.clone(), value, ttl), key)
    }

    pub fn del(self, key: &str) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::del(vec![key.clone()]), key)
    }

    pub fn expire(self, key: &str, ttl: Duration) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::expire(key.clone(), ttl), key)
    }

    pub fn incr_by(self, key: &str, delta: i64, expiry: Expiry) -> Self {
        let key = self.namespace.apply(key);
        self.script_call(commands::incr_by(key, delta, expiry))
    }

    pub fn incr_by_float(self, key: &str, delta: f64, expiry: Expiry) -> Self {
        let key = self.namespace.apply(key);
        self.script_call(commands::incr_by_float(key, delta, expiry))
    }

    pub fn rpush(self, key: &str, values: impl ToRedisArgs, expiry: Expiry) -> Self {
        let key = self.namespace.apply(key);
        self.script_call(commands::rpush(key, values, expiry))
    }

    pub fn eval(self, script: &Script, keys: &[&str], args: impl ToRedisArgs) -> Self {
        let keys: Vec<String> = keys.iter().map(|key| self.namespace.apply(key)).collect();
        let key = keys.first().cloned().unwrap_or_default();
        self.command(script.eval(&keys, &args.to_redis_args()), key)
    }

    fn script_call(self, call: ScriptCall) -> Self {
        let key = call.keys.first().cloned().unwrap_or_default();
        self.command(call.script.eval(&call.keys, &call.args), key)
    }

    fn command(mut self, command: Cmd, key: String) -> Self {
        self.ops.push(Op { command, key });
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
                .map(|group| self.run_group(connection.clone(), group)),
        )
        .await;
        let mut settled: Vec<Settled> = settled.into_iter().flatten().collect();
        settled.sort_by_key(|(index, _)| *index);
        settled.into_iter().map(|(_, result)| result).collect()
    }

    fn groups(&self, connection: &Connection) -> Vec<Vec<usize>> {
        match connection {
            _ if self.ops.is_empty() => Vec::new(),
            Connection::Standalone(_) | Connection::Sentinel(_) => {
                vec![(0..self.ops.len()).collect()]
            }
            Connection::Cluster(_) => {
                let mut by_slot: HashMap<Slot, Vec<usize>> = HashMap::new();
                for (index, op) in self.ops.iter().enumerate() {
                    by_slot
                        .entry(Slot::for_key(&op.key))
                        .or_default()
                        .push(index);
                }
                by_slot.into_values().collect()
            }
        }
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
                        reply.map_err(|error| Error::classify(error, deployment)),
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
