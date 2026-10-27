use std::{
    future::Future,
    time::{Duration, Instant},
};

use litellm_circuit_breaker::{Admission, Breaker, Failure};
use redis::{Cmd, FromRedisValue, IntoConnectionInfo, ToRedisArgs, ToSingleRedisArg, Value};

use crate::{
    ConnectError, Endpoint, Error, Namespace, Pipeline, RedisConfig, Script, commands,
    connection::Connection,
};

#[derive(Clone)]
pub struct RedisClient {
    pub(crate) connection: Connection,
    namespace: Namespace,
    breaker: Option<Breaker>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Ttl {
    Missing,
    Persistent,
    Expires(Duration),
}

impl RedisClient {
    pub fn new(config: RedisConfig) -> Result<Self, ConnectError> {
        let connection = match &config.endpoint {
            Endpoint::Url(url) => {
                Connection::standalone(url.as_str().into_connection_info()?, config.timeout)?
            }
            Endpoint::Host(host) => {
                Connection::standalone(host.connection_info()?, config.timeout)?
            }
            Endpoint::Cluster(cluster) => Connection::cluster(cluster, config.timeout)?,
        };
        Ok(Self {
            connection,
            namespace: config.namespace,
            breaker: config.breaker,
        })
    }

    pub async fn ping(&self) -> Result<(), Error> {
        self.query(&commands::ping()).await
    }

    pub async fn get<T: FromRedisValue>(&self, key: &str) -> Result<Option<T>, Error> {
        self.query(&commands::get(self.key(key))).await
    }

    pub async fn mget<T: FromRedisValue>(&self, keys: &[&str]) -> Result<Vec<Option<T>>, Error> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        self.query(&commands::mget(self.keys(keys))).await
    }

    pub async fn set(
        &self,
        key: &str,
        value: impl ToSingleRedisArg,
        ttl: Duration,
    ) -> Result<(), Error> {
        self.query(&commands::set(self.key(key), value, ttl)).await
    }

    pub async fn set_nx(
        &self,
        key: &str,
        value: impl ToSingleRedisArg,
        ttl: Duration,
    ) -> Result<bool, Error> {
        let reply: Option<String> = self
            .query(&commands::set_nx(self.key(key), value, ttl))
            .await?;
        Ok(reply.is_some())
    }

    pub async fn del(&self, keys: &[&str]) -> Result<usize, Error> {
        if keys.is_empty() {
            return Ok(0);
        }
        self.query(&commands::del(self.keys(keys))).await
    }

    pub async fn expire(&self, key: &str, ttl: Duration) -> Result<bool, Error> {
        self.query(&commands::expire(self.key(key), ttl)).await
    }

    pub async fn ttl(&self, key: &str) -> Result<Ttl, Error> {
        let reply: i64 = self.query(&commands::ttl(self.key(key))).await?;
        Ok(match reply {
            -2 => Ttl::Missing,
            -1 => Ttl::Persistent,
            millis => Ttl::Expires(Duration::from_millis(millis.unsigned_abs())),
        })
    }

    pub async fn incr_by(&self, key: &str, delta: i64) -> Result<i64, Error> {
        self.query(&commands::incr_by(self.key(key), delta)).await
    }

    pub async fn incr_by_float(&self, key: &str, delta: f64) -> Result<f64, Error> {
        self.query(&commands::incr_by_float(self.key(key), delta))
            .await
    }

    pub async fn rpush(&self, key: &str, values: impl ToRedisArgs) -> Result<usize, Error> {
        self.query(&commands::rpush(self.key(key), values)).await
    }

    pub async fn eval<T: FromRedisValue>(
        &self,
        script: &Script,
        keys: &[&str],
        args: impl ToRedisArgs,
    ) -> Result<T, Error> {
        let keys = self.keys(keys);
        let args = args.to_redis_args();
        self.guarded(async {
            Ok(script
                .invocation(&keys, &args)
                .invoke_async(&mut self.connection.clone())
                .await?)
        })
        .await
    }

    pub fn pipeline(&self) -> Pipeline {
        Pipeline::new(self.namespace.clone())
    }

    pub async fn run(&self, pipeline: &Pipeline) -> Vec<Result<Value, Error>> {
        let admission = match self.admit() {
            Ok(admission) => admission,
            Err(_) => {
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
        results
    }

    fn key(&self, key: &str) -> String {
        self.namespace.apply(key)
    }

    fn keys(&self, keys: &[&str]) -> Vec<String> {
        keys.iter().map(|key| self.key(key)).collect()
    }

    async fn query<T: FromRedisValue>(&self, command: &Cmd) -> Result<T, Error> {
        self.guarded(async { Ok(command.query_async(&mut self.connection.clone()).await?) })
            .await
    }

    async fn guarded<T>(&self, call: impl Future<Output = Result<T, Error>>) -> Result<T, Error> {
        let admission = self.admit()?;
        let result = call.await;
        settle(admission, result.as_ref().err());
        result
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

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use litellm_redis_testing::RedisServer;
    use rstest::rstest;

    use litellm_circuit_breaker::{Breaker, Config, State};

    use super::{RedisClient, Ttl};
    use crate::{
        DEFAULT_TIMEOUT, Endpoint, Error, HostEndpoint, Namespace, RedisConfig,
        test_support::{Server, Topology, client_id},
    };

    #[rstest]
    #[tokio::test]
    async fn namespaces_isolate_keys_on_the_server(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let a = server.client("a", DEFAULT_TIMEOUT).await;
        let b = server.client("b", DEFAULT_TIMEOUT).await;
        let bare = server.client("", DEFAULT_TIMEOUT).await;

        a.set("x", "1", Duration::from_secs(60)).await.unwrap();

        assert_eq!(a.get::<String>("x").await.unwrap().as_deref(), Some("1"));
        assert_eq!(a.get::<String>("a:x").await.unwrap().as_deref(), Some("1"));
        assert_eq!(
            bare.get::<String>("a:x").await.unwrap().as_deref(),
            Some("1")
        );
        assert_eq!(b.get::<String>("x").await.unwrap(), None);
    }

    #[rstest]
    #[tokio::test]
    async fn a_rejected_command_keeps_the_connection(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;
        client.rpush("list", "item").await.unwrap();
        let before = client_id(&client, "list").await;

        let rejected = client.incr_by("list", 1).await;

        assert!(matches!(rejected, Err(Error::Server(_))), "{rejected:?}");
        assert_eq!(client.rpush("list", "next").await.unwrap(), 2);
        assert_eq!(client_id(&client, "list").await, before);
    }

    #[rstest]
    #[tokio::test]
    async fn a_reply_of_another_type_is_a_decode_error_and_a_missing_key_is_none(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;
        client
            .set("float", "15.0", Duration::from_secs(60))
            .await
            .unwrap();

        let decoded = client.get::<i64>("float").await;

        assert!(matches!(decoded, Err(Error::Decode(_))), "{decoded:?}");
        assert_eq!(client.get::<i64>("missing").await.unwrap(), None);
    }

    #[rstest]
    #[tokio::test]
    async fn a_timed_out_command_does_not_leak_its_reply_into_the_next_one(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", Duration::from_millis(500)).await;
        client
            .set("first", "1", Duration::from_secs(60))
            .await
            .unwrap();
        client
            .set("second", "2", Duration::from_secs(60))
            .await
            .unwrap();

        server.freeze().await;
        let timed_out = client.get::<String>("first").await;
        server.thaw().await;

        assert!(
            matches!(timed_out, Err(Error::Transport(_))),
            "{timed_out:?}"
        );
        assert_eq!(
            client.get::<String>("second").await.unwrap().as_deref(),
            Some("2")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn the_same_client_recovers_after_redis_restarts(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", Duration::from_millis(500)).await;

        server.restart().await;

        let deadline = Instant::now() + Duration::from_secs(30);
        let recovered = loop {
            match client.incr_by("after_restart", 1).await {
                Ok(value) => break Some(value),
                Err(_) if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(200)).await
                }
                Err(_) => break None,
            }
        };
        assert_eq!(recovered, Some(1));
    }

    #[rstest]
    #[tokio::test]
    async fn a_client_created_while_redis_is_down_works_once_redis_is_up(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        server.stop().await;

        let client = server.client("", Duration::from_millis(500)).await;
        let while_down = client.ping().await;
        server.resume().await;

        assert!(
            matches!(while_down, Err(Error::Transport(_))),
            "{while_down:?}"
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        let recovered = loop {
            match client.ping().await {
                Ok(()) => break true,
                Err(_) if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(200)).await
                }
                Err(_) => break false,
            }
        };
        assert!(recovered);
    }

    #[rstest]
    #[tokio::test]
    async fn empty_batches_succeed_without_asking_redis(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;

        assert_eq!(
            client.mget::<String>(&[]).await.unwrap(),
            Vec::<Option<String>>::new()
        );
        assert_eq!(client.del(&[]).await.unwrap(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn mget_and_del_span_slots_in_input_order(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;
        client.set("a", "1", Duration::from_secs(60)).await.unwrap();
        client.set("c", "3", Duration::from_secs(60)).await.unwrap();

        let values = client.mget::<String>(&["c", "b", "a"]).await.unwrap();
        let deleted = client.del(&["a", "b", "c"]).await.unwrap();

        assert_eq!(
            values,
            vec![Some("3".to_owned()), None, Some("1".to_owned())]
        );
        assert_eq!(deleted, 2);
        assert_eq!(
            client.mget::<String>(&["a", "c"]).await.unwrap(),
            vec![None, None]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn set_nx_only_writes_a_missing_key(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;

        let first = client
            .set_nx("lock", "owner-1", Duration::from_secs(60))
            .await
            .unwrap();
        let second = client
            .set_nx("lock", "owner-2", Duration::from_secs(60))
            .await
            .unwrap();

        assert!(first);
        assert!(!second);
        assert_eq!(
            client.get::<String>("lock").await.unwrap().as_deref(),
            Some("owner-1")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn ttl_reports_missing_persistent_and_remaining_time(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;
        client.incr_by("persistent", 1).await.unwrap();
        client
            .set("expiring", "1", Duration::from_secs(60))
            .await
            .unwrap();
        client.incr_by_float("expired_later", 0.5).await.unwrap();
        assert!(
            client
                .expire("expired_later", Duration::from_secs(30))
                .await
                .unwrap()
        );

        assert_eq!(client.ttl("missing").await.unwrap(), Ttl::Missing);
        assert_eq!(client.ttl("persistent").await.unwrap(), Ttl::Persistent);
        assert!(matches!(
            client.ttl("expiring").await.unwrap(),
            Ttl::Expires(left) if left > Duration::from_secs(50) && left <= Duration::from_secs(60)
        ));
        assert!(matches!(
            client.ttl("expired_later").await.unwrap(),
            Ttl::Expires(left) if left > Duration::from_secs(20) && left <= Duration::from_secs(30)
        ));
        assert!(
            !client
                .expire("missing", Duration::from_secs(30))
                .await
                .unwrap()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn incr_by_float_accumulates_fractions(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;

        client.incr_by_float("spend", 0.25).await.unwrap();

        assert_eq!(client.incr_by_float("spend", 0.5).await.unwrap(), 0.75);
    }

    #[rstest]
    #[tokio::test]
    async fn concurrent_commands_on_one_cloned_client_all_apply(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server.client("", DEFAULT_TIMEOUT).await;

        let tasks: Vec<_> = (0..100)
            .map(|_| {
                let client = client.clone();
                tokio::spawn(async move {
                    for _ in 0..10 {
                        client.incr_by("shared", 1).await.unwrap();
                    }
                })
            })
            .collect();
        for task in tasks {
            task.await.unwrap();
        }

        assert_eq!(client.get::<i64>("shared").await.unwrap(), Some(1000));
    }

    fn breaker(failure_threshold: u32) -> Breaker {
        Breaker::new(
            Config {
                failure_threshold,
                recovery_timeout: Duration::from_millis(200),
                timeout_min_duration: Duration::ZERO,
            },
            Box::new(|_| {}),
        )
    }

    #[rstest]
    #[tokio::test]
    async fn an_open_breaker_rejects_without_waiting_on_redis_and_a_probe_closes_it(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let breaker = breaker(2);
        let client = server
            .guarded_client("", Duration::from_millis(300), Some(breaker.clone()))
            .await;

        server.freeze().await;
        let first = client.get::<String>("k").await;
        let second = client.get::<String>("k").await;
        let started = Instant::now();
        let third = client.get::<String>("k").await;
        let rejected_in = started.elapsed();
        server.thaw().await;

        assert!(matches!(first, Err(Error::Transport(_))), "{first:?}");
        assert!(matches!(second, Err(Error::Transport(_))), "{second:?}");
        assert!(matches!(third, Err(Error::BreakerOpen)), "{third:?}");
        assert!(rejected_in < Duration::from_millis(50), "{rejected_in:?}");

        let deadline = Instant::now() + Duration::from_secs(30);
        let recovered = loop {
            match client.get::<String>("k").await {
                Ok(value) => break Some(value),
                Err(_) if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(50)).await
                }
                Err(_) => break None,
            }
        };
        assert_eq!(recovered, Some(None));
        assert!(
            matches!(breaker.state(), State::Closed { .. }),
            "{:?}",
            breaker.state()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn rejected_commands_never_open_the_breaker(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let breaker = breaker(2);
        let client = server
            .guarded_client("", DEFAULT_TIMEOUT, Some(breaker.clone()))
            .await;
        client.rpush("list", "item").await.unwrap();

        let rejected = [
            client.incr_by("list", 1).await,
            client.incr_by("list", 1).await,
            client.incr_by("list", 1).await,
        ];

        assert!(
            matches!(rejected[2], Err(Error::Server(_))),
            "{:?}",
            rejected[2]
        );
        assert!(
            matches!(breaker.state(), State::Closed { hard: 0, .. }),
            "{:?}",
            breaker.state()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_open_breaker_fails_every_pipeline_op_without_sending(
        #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    ) {
        let server = Server::start(topology).await;
        let client = server
            .guarded_client("", Duration::from_millis(300), Some(breaker(1)))
            .await;
        let pipeline = client.pipeline().incr_by("a", 1).incr_by("c", 1);

        server.freeze().await;
        let failed = client.run(&pipeline).await;
        let rejected = client.run(&pipeline).await;
        server.thaw().await;

        assert!(
            matches!(failed[0], Err(Error::Transport(_))),
            "{:?}",
            failed[0]
        );
        assert!(
            matches!(rejected[0], Err(Error::BreakerOpen)),
            "{:?}",
            rejected[0]
        );
        assert!(
            matches!(rejected[1], Err(Error::BreakerOpen)),
            "{:?}",
            rejected[1]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_host_endpoint_connects_to_its_database() {
        let server = RedisServer::start().await.unwrap();
        let port = server.url().rsplit(':').next().unwrap().parse().unwrap();
        let connect = |endpoint| {
            RedisClient::new(RedisConfig {
                endpoint,
                namespace: Namespace::default(),
                timeout: DEFAULT_TIMEOUT,
                breaker: None,
            })
        };
        let in_db_3 = connect(Endpoint::Host(HostEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            username: None,
            password: None,
            tls: false,
            db: 3,
        }))
        .unwrap();
        let in_db_0 = connect(Endpoint::Url(server.url().to_owned())).unwrap();
        let via_url = connect(Endpoint::Url(format!("{}/3", server.url()))).unwrap();

        in_db_3
            .set("k", "3", Duration::from_secs(60))
            .await
            .unwrap();

        assert_eq!(in_db_0.get::<String>("k").await.unwrap(), None);
        assert_eq!(
            via_url.get::<String>("k").await.unwrap().as_deref(),
            Some("3")
        );
    }
}
