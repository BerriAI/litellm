use std::{collections::HashMap, time::Duration};

use futures_util::future::join_all;
use redis::{
    Cmd, ErrorKind, RedisResult, ServerErrorKind, ToRedisArgs, ToSingleRedisArg, Value,
    cluster_routing::Slot,
};

use crate::{Error, Namespace, Script, commands, connection::Connection};

pub struct Pipeline {
    namespace: Namespace,
    ops: Vec<Op>,
}

struct Op {
    command: Cmd,
    key: String,
    script: Option<ScriptCall>,
}

struct ScriptCall {
    script: Script,
    keys: Vec<String>,
    args: Vec<Vec<u8>>,
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

    pub fn incr_by(self, key: &str, delta: i64) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::incr_by(key.clone(), delta), key)
    }

    pub fn incr_by_float(self, key: &str, delta: f64) -> Self {
        let key = self.namespace.apply(key);
        self.command(commands::incr_by_float(key.clone(), delta), key)
    }

    pub fn eval(self, script: &Script, keys: &[&str], args: impl ToRedisArgs) -> Self {
        let keys: Vec<String> = keys.iter().map(|key| self.namespace.apply(key)).collect();
        let args = args.to_redis_args();
        let op = Op {
            command: script.evalsha(&keys, &args),
            key: keys.first().cloned().unwrap_or_default(),
            script: Some(ScriptCall {
                script: script.clone(),
                keys,
                args,
            }),
        };
        self.push(op)
    }

    fn command(self, command: Cmd, key: String) -> Self {
        self.push(Op {
            command,
            key,
            script: None,
        })
    }

    fn push(mut self, op: Op) -> Self {
        self.ops.push(op);
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
            Connection::Standalone(_) => vec![(0..self.ops.len()).collect()],
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
        let mut pipeline = redis::pipe();
        pipeline.ignore_errors();
        for &index in indices {
            pipeline.add_command(self.ops[index].command.clone());
        }
        let replies: RedisResult<Vec<RedisResult<Value>>> =
            pipeline.query_async(&mut connection).await;
        let replies = match replies {
            Ok(replies) => replies,
            Err(error) => {
                return indices
                    .iter()
                    .map(|&index| (index, Err(Error::from(error.clone()))))
                    .collect();
            }
        };
        let mut settled = Vec::with_capacity(indices.len());
        for (&index, reply) in indices.iter().zip(replies) {
            settled.push((index, self.settle(&mut connection, index, reply).await));
        }
        settled
    }

    async fn settle(
        &self,
        connection: &mut Connection,
        index: usize,
        reply: RedisResult<Value>,
    ) -> Result<Value, Error> {
        match (reply, &self.ops[index].script) {
            (Err(error), Some(call))
                if error.kind() == ErrorKind::Server(ServerErrorKind::NoScript) =>
            {
                Ok(call
                    .script
                    .invocation(&call.keys, &call.args)
                    .invoke_async(connection)
                    .await?)
            }
            (reply, _) => Ok(reply?),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use redis::Value;
    use rstest::rstest;

    use crate::{
        DEFAULT_TIMEOUT, Error, Script,
        test_support::{Server, Topology, flush_scripts},
    };

    #[rstest]
    #[tokio::test]
    async fn each_op_settles_on_its_own_in_input_order(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;
        client
            .set("text", "not a number", Duration::from_secs(60))
            .await
            .unwrap();
        let pipeline = client
            .pipeline()
            .incr_by("a", 1)
            .incr_by("text", 1)
            .incr_by("c", 5)
            .set("b", "2", Duration::from_secs(60))
            .get("b")
            .incr_by("a", 1)
            .get("missing");

        let results = client.run(&pipeline).await;

        assert_eq!(results.len(), 7);
        assert_eq!(results[0].as_ref().unwrap(), &Value::Int(1));
        assert!(
            matches!(results[1], Err(Error::Server(_))),
            "{:?}",
            results[1]
        );
        assert_eq!(results[2].as_ref().unwrap(), &Value::Int(5));
        assert_eq!(results[3].as_ref().unwrap(), &Value::Okay);
        assert_eq!(
            results[4].as_ref().unwrap(),
            &Value::BulkString(b"2".to_vec())
        );
        assert_eq!(results[5].as_ref().unwrap(), &Value::Int(2));
        assert_eq!(results[6].as_ref().unwrap(), &Value::Nil);
    }

    #[rstest]
    #[tokio::test]
    async fn a_flushed_script_in_a_pipeline_reruns_alone(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("ns", DEFAULT_TIMEOUT).await;
        let incr = Script::new("return redis.call('INCRBY', KEYS[1], ARGV[1])");
        client.eval::<i64>(&incr, &["counter"], 1).await.unwrap();
        flush_scripts(&client).await;
        let pipeline = client
            .pipeline()
            .eval(&incr, &["counter"], 2)
            .incr_by("other", 7);

        let results = client.run(&pipeline).await;

        assert_eq!(results[0].as_ref().unwrap(), &Value::Int(3));
        assert_eq!(results[1].as_ref().unwrap(), &Value::Int(7));
        assert_eq!(client.get::<i64>("counter").await.unwrap(), Some(3));
    }

    #[rstest]
    #[tokio::test]
    async fn an_unreachable_server_fails_every_op_as_transport(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", Duration::from_millis(500)).await;
        let pipeline = client.pipeline().incr_by("a", 1).incr_by("c", 1);

        server.freeze().await;
        let results = client.run(&pipeline).await;
        server.thaw().await;

        assert_eq!(results.len(), 2);
        assert!(
            matches!(results[0], Err(Error::Transport(_))),
            "{:?}",
            results[0]
        );
        assert!(
            matches!(results[1], Err(Error::Transport(_))),
            "{:?}",
            results[1]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_empty_pipeline_returns_without_asking_redis(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", Duration::from_secs(2)).await;

        server.freeze().await;
        let started = Instant::now();
        let results = client.run(&client.pipeline()).await;
        let elapsed = started.elapsed();
        server.thaw().await;

        assert!(results.is_empty());
        assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
    }
}
