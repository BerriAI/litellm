use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_circuit_breaker::{Admission, Breaker, Failure};
use redis::{Cmd, FromRedisValue, RedisError, ToRedisArgs, ToSingleRedisArg, Value};

use crate::{
    CallEvent, ConnectError, Error, Expiry, FailureKind, Namespace, Operation, Outcome, Pipeline,
    RedisConfig, RedisEvents, Script,
    commands::{self, ScriptCall},
    connection::Connection,
};

#[derive(Clone)]
pub struct RedisClient {
    pub(crate) connection: Connection,
    namespace: Namespace,
    breaker: Option<Breaker>,
    events: Option<Arc<dyn RedisEvents>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Ttl {
    Missing,
    Persistent,
    Expires(Duration),
}

impl RedisClient {
    pub fn new(config: RedisConfig) -> Result<Self, ConnectError> {
        Ok(Self {
            connection: Connection::new(&config.endpoint, &config.credentials, config.timeouts)?,
            namespace: config.namespace,
            breaker: config.breaker,
            events: config.events,
        })
    }

    pub async fn ping(&self) -> Result<(), Error> {
        self.query("PING", &commands::ping()).await
    }

    pub async fn get<T: FromRedisValue>(&self, key: &str) -> Result<Option<T>, Error> {
        self.query("GET", &commands::get(self.key(key))).await
    }

    pub async fn mget<T: FromRedisValue>(&self, keys: &[&str]) -> Result<Vec<Option<T>>, Error> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        self.query("MGET", &commands::mget(self.keys(keys))).await
    }

    pub async fn set(
        &self,
        key: &str,
        value: impl ToSingleRedisArg,
        ttl: Duration,
    ) -> Result<(), Error> {
        self.query("SET", &commands::set(self.key(key), value, ttl))
            .await
    }

    pub async fn set_nx(
        &self,
        key: &str,
        value: impl ToSingleRedisArg,
        ttl: Duration,
    ) -> Result<bool, Error> {
        let reply: Option<String> = self
            .query("SET", &commands::set_nx(self.key(key), value, ttl))
            .await?;
        Ok(reply.is_some())
    }

    pub async fn del(&self, keys: &[&str]) -> Result<usize, Error> {
        if keys.is_empty() {
            return Ok(0);
        }
        self.query("DEL", &commands::del(self.keys(keys))).await
    }

    pub async fn expire(&self, key: &str, ttl: Duration) -> Result<bool, Error> {
        self.query("PEXPIRE", &commands::expire(self.key(key), ttl))
            .await
    }

    pub async fn ttl(&self, key: &str) -> Result<Ttl, Error> {
        let reply: i64 = self.query("PTTL", &commands::ttl(self.key(key))).await?;
        Ok(match reply {
            -2 => Ttl::Missing,
            -1 => Ttl::Persistent,
            millis => Ttl::Expires(Duration::from_millis(millis.unsigned_abs())),
        })
    }

    pub async fn incr_by(&self, key: &str, delta: i64, expiry: Expiry) -> Result<i64, Error> {
        self.call("INCRBY", commands::incr_by(self.key(key), delta, expiry))
            .await
    }

    pub async fn incr_by_float(&self, key: &str, delta: f64, expiry: Expiry) -> Result<f64, Error> {
        self.call(
            "INCRBYFLOAT",
            commands::incr_by_float(self.key(key), delta, expiry),
        )
        .await
    }

    pub async fn rpush(
        &self,
        key: &str,
        values: impl ToRedisArgs,
        expiry: Expiry,
    ) -> Result<usize, Error> {
        self.call("RPUSH", commands::rpush(self.key(key), values, expiry))
            .await
    }

    pub async fn eval<T: FromRedisValue>(
        &self,
        script: &Script,
        keys: &[&str],
        args: impl ToRedisArgs,
    ) -> Result<T, Error> {
        let keys = self.keys(keys);
        let args = args.to_redis_args();
        self.guarded(Operation::Script, async {
            script
                .invocation(&keys, &args)
                .invoke_async(&mut self.connection.clone())
                .await
        })
        .await
    }

    pub fn pipeline(&self) -> Pipeline {
        Pipeline::new(self.namespace.clone())
    }

    pub async fn run(&self, pipeline: &Pipeline) -> Vec<Result<Value, Error>> {
        let operation = Operation::Pipeline {
            ops: pipeline.len(),
        };
        let started = Instant::now();
        let admission = match self.admit() {
            Ok(admission) => admission,
            Err(_) => {
                self.observe(
                    operation,
                    started,
                    Outcome::Failed(FailureKind::BreakerOpen),
                );
                return (0..pipeline.len())
                    .map(|_| Err(Error::BreakerOpen))
                    .collect();
            }
        };
        let results = pipeline.run(&self.connection).await;
        let transport = results.iter().find_map(|result| {
            result
                .as_ref()
                .err()
                .filter(|error| matches!(error, Error::Transport(_)))
        });
        settle(admission, transport);
        let failure = transport.or_else(|| results.iter().find_map(|result| result.as_ref().err()));
        let outcome = failure.map_or(Outcome::Succeeded, |error| {
            Outcome::Failed(FailureKind::of(error))
        });
        self.observe(operation, started, outcome);
        results
    }

    fn key(&self, key: &str) -> String {
        self.namespace.apply(key)
    }

    fn keys(&self, keys: &[&str]) -> Vec<String> {
        keys.iter().map(|key| self.key(key)).collect()
    }

    async fn query<T: FromRedisValue>(
        &self,
        name: &'static str,
        command: &Cmd,
    ) -> Result<T, Error> {
        self.guarded(Operation::Command(name), async {
            command.query_async(&mut self.connection.clone()).await
        })
        .await
    }

    async fn call<T: FromRedisValue>(
        &self,
        name: &'static str,
        call: ScriptCall,
    ) -> Result<T, Error> {
        self.guarded(Operation::Command(name), async {
            call.script
                .invocation(&call.keys, &call.args)
                .invoke_async(&mut self.connection.clone())
                .await
        })
        .await
    }

    async fn guarded<T>(
        &self,
        operation: Operation,
        call: impl Future<Output = Result<T, RedisError>>,
    ) -> Result<T, Error> {
        let started = Instant::now();
        let result = match self.admit() {
            Ok(admission) => {
                let result = call
                    .await
                    .map_err(|error| Error::classify(error, self.connection.deployment()));
                settle(admission, result.as_ref().err());
                result
            }
            Err(error) => Err(error),
        };
        self.observe(operation, started, Outcome::of(&result));
        result
    }

    fn observe(&self, operation: Operation, started: Instant, outcome: Outcome) {
        if let Some(events) = &self.events {
            events.call(&CallEvent {
                operation,
                duration: started.elapsed(),
                outcome,
            });
        }
    }

    fn admit(&self) -> Result<Option<Admission>, Error> {
        self.breaker
            .as_ref()
            .map(|breaker| {
                breaker
                    .admit(Instant::now())
                    .map_err(|_| Error::BreakerOpen)
            })
            .transpose()
    }
}

fn settle(admission: Option<Admission>, error: Option<&Error>) {
    let Some(admission) = admission else {
        return;
    };
    match error {
        None => admission.succeeded(),
        Some(Error::Transport(error)) if error.is_timeout() => {
            admission.failed(Failure::Timeout, Instant::now())
        }
        Some(Error::Transport(_)) => admission.failed(Failure::Hard, Instant::now()),
        Some(Error::Server(_) | Error::Decode(_) | Error::BreakerOpen) => drop(admission),
    }
}
