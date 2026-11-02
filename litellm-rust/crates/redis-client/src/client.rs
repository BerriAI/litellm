use std::{
    future::Future,
    sync::{Arc, Mutex, PoisonError, Weak},
    time::{Duration, Instant},
};

use futures_util::stream::BoxStream;
use litellm_circuit_breaker::{Admission, Breaker, Failure, State};
use redis::{Cmd, FromRedisValue, RedisError, ToRedisArgs, ToSingleRedisArg, Value};

use crate::{
    Amount, CallEvent, ConnectError, Error, Expiry, FailureKind, Namespace, Operation, Outcome,
    Pipeline, RedisBatch, RedisConfig, RedisEvents, Script, commands, connection::Connection,
    pubsub::Subscriber,
};

/// One Redis client. Clones share the connection, the breaker and the subscriber.
#[derive(Clone)]
pub struct RedisClient(Arc<Shared>);

/// A handle that does not keep the client alive, for a registry that must not outlive its users.
#[derive(Clone)]
pub struct WeakRedisClient(Weak<Shared>);

impl WeakRedisClient {
    pub fn upgrade(&self) -> Option<RedisClient> {
        self.0.upgrade().map(RedisClient)
    }
}

struct Shared {
    connection: Connection,
    subscriber: Subscriber,
    namespace: Namespace,
    breaker: Option<Breaker>,
    events: Option<Arc<dyn RedisEvents>>,
    health_check: Option<Duration>,
    last_used: Mutex<Instant>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Ttl {
    Missing,
    Persistent,
    Expires(Duration),
}

impl RedisClient {
    pub fn new(config: RedisConfig) -> Result<Self, ConnectError> {
        let connection = Connection::new(&config.target, &config.credentials, config.timeouts)?;
        Ok(Self(Arc::new(Shared {
            subscriber: Subscriber::new(
                &config.target,
                &config.credentials,
                config.timeouts,
                &connection,
            )?,
            connection,
            namespace: config.namespace,
            breaker: config.breaker,
            events: config.events,
            health_check: config.health_check,
            last_used: Mutex::new(Instant::now()),
        })))
    }

    pub fn downgrade(&self) -> WeakRedisClient {
        WeakRedisClient(Arc::downgrade(&self.0))
    }

    /// The breaker's current state, or none when the breaker is disabled.
    pub fn breaker_state(&self) -> Option<State> {
        self.0.breaker.as_ref().map(Breaker::state)
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
        self.query("EXPIRE", &commands::expire(self.key(key), ttl))
            .await
    }

    pub async fn ttl(&self, key: &str) -> Result<Ttl, Error> {
        let reply: i64 = self.query("TTL", &commands::ttl(self.key(key))).await?;
        Ok(match reply {
            -2 => Ttl::Missing,
            -1 => Ttl::Persistent,
            seconds => Ttl::Expires(Duration::from_secs(seconds.unsigned_abs())),
        })
    }

    /// `RedisCache.increment_cache`: `INCRBY`, then `TTL`, then `EXPIRE` only when the key has no
    /// TTL, each its own round trip.
    pub async fn incr_by(&self, key: &str, delta: i64, ttl: Duration) -> Result<i64, Error> {
        let key = self.key(key);
        self.guarded(Operation::Command("INCRBY"), async {
            let mut connection = self.0.connection.clone();
            let value: i64 = commands::incr_by(key.clone(), delta)
                .query_async(&mut connection)
                .await?;
            let current: i64 = commands::ttl(key.clone())
                .query_async(&mut connection)
                .await?;
            if current == -1 {
                commands::expire(key, ttl)
                    .exec_async(&mut connection)
                    .await?;
            }
            Ok(value)
        })
        .await
    }

    /// `INCRBYFLOAT` alone, `_incrbyfloat_with_ttl` without a TTL.
    pub(crate) async fn incr_by_float_without_ttl(
        &self,
        key: &str,
        amount: Amount,
    ) -> Result<f64, Error> {
        self.query(
            "INCRBYFLOAT",
            &commands::incr_by_float(self.key(key), amount),
        )
        .await
    }

    /// `RedisCache.async_increment`: `INCRBYFLOAT` and its TTL command in one round trip, and for
    /// [`Expiry::KeepExisting`] a third, `EXPIRE`, only when the key has no TTL.
    pub async fn incr_by_float(
        &self,
        key: &str,
        amount: Amount,
        expiry: Expiry,
    ) -> Result<f64, Error> {
        let key = self.key(key);
        self.guarded(Operation::Command("INCRBYFLOAT"), async {
            let mut connection = self.0.connection.clone();
            let mut pipeline = redis::pipe();
            pipeline.add_command(commands::incr_by_float(key.clone(), amount));
            match expiry {
                Expiry::Refresh(ttl) => {
                    let (value, _): (f64, bool) = pipeline
                        .add_command(commands::expire(key, ttl))
                        .query_async(&mut connection)
                        .await?;
                    Ok(value)
                }
                Expiry::KeepExisting(ttl) => {
                    let (value, current): (f64, i64) = pipeline
                        .add_command(commands::ttl(key.clone()))
                        .query_async(&mut connection)
                        .await?;
                    if current == -1 {
                        commands::expire(key, ttl)
                            .exec_async(&mut connection)
                            .await?;
                    }
                    Ok(value)
                }
            }
        })
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
                .invoke_async(&mut self.0.connection.clone())
                .await
        })
        .await
    }

    /// The channel is sent as given, without the namespace, like redis-py's `publish`.
    pub async fn publish(&self, channel: &str, payload: &[u8]) -> Result<(), Error> {
        self.query("PUBLISH", &commands::publish(channel, payload))
            .await
    }

    /// Messages published on `channel`, on a connection of its own. After the connection is lost
    /// it reconnects and resubscribes, and messages published meanwhile are not delivered.
    pub fn subscribe(&self, channel: &str) -> BoxStream<'static, Vec<u8>> {
        self.0.subscriber.clone().messages(channel.to_owned())
    }

    pub fn pipeline(&self) -> Pipeline {
        Pipeline::new(self.0.namespace.clone())
    }

    pub fn same_client(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn is_cluster(&self) -> bool {
        matches!(self.0.connection, Connection::Cluster(_))
    }

    pub fn batch(&self) -> RedisBatch {
        RedisBatch::new(self.clone())
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
        self.check_health().await;
        let results = pipeline.run(&self.0.connection).await;
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
        self.0.namespace.apply(key)
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
            command.query_async(&mut self.0.connection.clone()).await
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
                self.check_health().await;
                let result = call
                    .await
                    .map_err(|error| Error::classify(error, self.0.connection.deployment()));
                settle(admission, result.as_ref().err());
                result
            }
            Err(error) => Err(error),
        };
        self.observe(operation, started, Outcome::of(&result));
        result
    }

    fn observe(&self, operation: Operation, started: Instant, outcome: Outcome) {
        *self
            .0
            .last_used
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Instant::now();
        if let Some(events) = &self.0.events {
            events.call(&CallEvent {
                operation,
                duration: started.elapsed(),
                outcome,
            });
        }
    }

    /// redis-py pings a connection that sat idle longer than `health_check_interval` before
    /// sending the next command on it, and reconnects when the ping fails. The ping is not a
    /// call of the caller's and never reaches the breaker or the events.
    async fn check_health(&self) {
        let Some(interval) = self.0.health_check else {
            return;
        };
        let idle = self
            .0
            .last_used
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .elapsed();
        if idle < interval {
            return;
        }
        if let Err(error) = self.0.connection.ping_every_node().await {
            tracing::debug!(%error, "idle Redis connection failed its health check, the next command reconnects");
        }
    }

    fn admit(&self) -> Result<Option<Admission>, Error> {
        self.0
            .breaker
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
