use std::{
    future::Future,
    sync::{Arc, Mutex, PoisonError, Weak},
    time::{Duration, Instant},
};

use futures_util::stream::BoxStream;
use litellm_circuit_breaker::{Admission, Breaker, Failure};
use redis::{Cmd, ErrorKind, RedisError, ServerErrorKind, Value};

use crate::{
    CallEvent, Command, ConnectError, Error, FailureKind, Namespace, Operation, Outcome, Pipeline,
    RedisBatch, RedisConfig, RedisEvents,
    commands::{self, Op, Plan, Trips},
    connection::Connection,
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
        let connection = Connection::new(
            &config.target,
            &config.credentials,
            config.timeouts,
            config.keepalive,
        )?;
        Ok(Self(Arc::new(Shared {
            subscriber: Subscriber::new(
                &config.target,
                &config.credentials,
                config.timeouts,
                config.keepalive,
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

    pub async fn ping(&self) -> Result<(), Error> {
        self.guarded(Operation::Command("PING"), async {
            commands::ping()
                .query_async(&mut self.0.connection.clone())
                .await
        })
        .await
    }

    /// Sends one command on its own, the way the `RedisCache` method behind it sends it.
    pub async fn send<T>(&self, command: Command<T>) -> Result<T, Error> {
        let (op, decode) = command.prepare(&self.0.namespace);
        self.run_alone(op).await.and_then(decode)
    }

    pub(crate) async fn run_alone(&self, op: Op) -> Result<Value, Error> {
        let Op {
            operation, plan, ..
        } = op;
        self.guarded(operation, async {
            let mut connection = self.0.connection.clone();
            match plan {
                Plan::Reply(value) => Ok(value),
                Plan::Commands(commands) => match <[Cmd; 1]>::try_from(commands) {
                    Ok([command]) => command.query_async(&mut connection).await,
                    Err(commands) => {
                        let mut pipeline = redis::pipe();
                        for command in commands {
                            pipeline.add_command(command);
                        }
                        let replies: Vec<Value> = pipeline.query_async(&mut connection).await?;
                        Ok(replies.into_iter().next().unwrap_or(Value::Nil))
                    }
                },
                Plan::Script(call) => {
                    call.script
                        .invocation(&call.keys, &call.args)
                        .invoke_async(&mut connection)
                        .await
                }
                Plan::KeepingTtl {
                    increment,
                    key,
                    ttl,
                    trips,
                } => {
                    let (value, current): (Value, i64) = match trips {
                        Trips::Shared => {
                            redis::pipe()
                                .add_command(increment)
                                .add_command(commands::ttl_cmd(&key))
                                .query_async(&mut connection)
                                .await?
                        }
                        Trips::Separate => (
                            increment.query_async(&mut connection).await?,
                            commands::ttl_cmd(&key).query_async(&mut connection).await?,
                        ),
                    };
                    if current == -1 {
                        redis::cmd("EXPIRE")
                            .arg(&key)
                            .arg(ttl.as_secs())
                            .exec_async(&mut connection)
                            .await?;
                    }
                    Ok(value)
                }
            }
        })
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

    pub(crate) fn namespace(&self) -> &Namespace {
        &self.0.namespace
    }

    pub fn is_cluster(&self) -> bool {
        matches!(self.0.connection, Connection::Cluster(_))
    }

    pub fn same_client(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
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
    /// sending the next command on it. A ping that times out or finds the socket dropped leaves
    /// that connection behind, as a command would, so the command then goes out on a fresh one,
    /// where redis-py 5.3.1 disconnects and fails the command with the ping's error. The ping is
    /// not a call of the caller's and never reaches the breaker or the events.
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
            tracing::debug!(%error, "idle Redis connection failed its health check, replacing it");
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
        Some(Error::Server(error)) if proves_redis_answers(error) => admission.succeeded(),
        Some(Error::Server(_) | Error::Decode(_) | Error::BreakerOpen) => drop(admission),
    }
}

// A replica refusing a write is the old primary mid-failover, which must not reset a run of failures
fn proves_redis_answers(error: &RedisError) -> bool {
    !matches!(
        error.kind(),
        ErrorKind::InvalidClientConfig | ErrorKind::Server(ServerErrorKind::ReadOnly)
    )
}
