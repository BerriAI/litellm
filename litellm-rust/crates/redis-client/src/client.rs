use std::time::Duration;

use redis::{
    Cmd, FromRedisValue, ToRedisArgs, ToSingleRedisArg,
    aio::{ConnectionManager, ConnectionManagerConfig},
};

use crate::{ConnectError, Error, Namespace, RedisConfig};

const RECONNECT_RETRIES: usize = 3;

#[derive(Clone)]
pub struct RedisClient {
    connection: ConnectionManager,
    namespace: Namespace,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Ttl {
    Missing,
    Persistent,
    Expires(Duration),
}

impl RedisClient {
    pub async fn connect(config: RedisConfig) -> Result<Self, ConnectError> {
        let client = redis::Client::open(config.endpoint.connection_info()?)?;
        let manager_config = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(config.timeout))
            .set_response_timeout(Some(config.timeout))
            .set_number_of_retries(RECONNECT_RETRIES);
        Ok(Self {
            connection: client
                .get_connection_manager_with_config(manager_config)
                .await?,
            namespace: config.namespace,
        })
    }

    pub async fn get<T: FromRedisValue>(&self, key: &str) -> Result<Option<T>, Error> {
        self.query(redis::cmd("GET").arg(self.key(key))).await
    }

    pub async fn mget<T: FromRedisValue>(&self, keys: &[&str]) -> Result<Vec<Option<T>>, Error> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        self.query(redis::cmd("MGET").arg(self.keys(keys))).await
    }

    pub async fn set(
        &self,
        key: &str,
        value: impl ToSingleRedisArg,
        ttl: Duration,
    ) -> Result<(), Error> {
        self.query(
            redis::cmd("SET")
                .arg(self.key(key))
                .arg(value)
                .arg("PX")
                .arg(millis(ttl)),
        )
        .await
    }

    pub async fn set_nx(
        &self,
        key: &str,
        value: impl ToSingleRedisArg,
        ttl: Duration,
    ) -> Result<bool, Error> {
        let reply: Option<String> = self
            .query(
                redis::cmd("SET")
                    .arg(self.key(key))
                    .arg(value)
                    .arg("NX")
                    .arg("PX")
                    .arg(millis(ttl)),
            )
            .await?;
        Ok(reply.is_some())
    }

    pub async fn del(&self, keys: &[&str]) -> Result<usize, Error> {
        if keys.is_empty() {
            return Ok(0);
        }
        self.query(redis::cmd("DEL").arg(self.keys(keys))).await
    }

    pub async fn expire(&self, key: &str, ttl: Duration) -> Result<bool, Error> {
        self.query(redis::cmd("PEXPIRE").arg(self.key(key)).arg(millis(ttl)))
            .await
    }

    pub async fn ttl(&self, key: &str) -> Result<Ttl, Error> {
        let reply: i64 = self.query(redis::cmd("PTTL").arg(self.key(key))).await?;
        Ok(match reply {
            -2 => Ttl::Missing,
            -1 => Ttl::Persistent,
            millis => Ttl::Expires(Duration::from_millis(millis.unsigned_abs())),
        })
    }

    pub async fn incr_by(&self, key: &str, delta: i64) -> Result<i64, Error> {
        self.query(redis::cmd("INCRBY").arg(self.key(key)).arg(delta))
            .await
    }

    pub async fn incr_by_float(&self, key: &str, delta: f64) -> Result<f64, Error> {
        self.query(redis::cmd("INCRBYFLOAT").arg(self.key(key)).arg(delta))
            .await
    }

    pub async fn rpush(&self, key: &str, values: impl ToRedisArgs) -> Result<usize, Error> {
        self.query(redis::cmd("RPUSH").arg(self.key(key)).arg(values))
            .await
    }

    fn key(&self, key: &str) -> String {
        self.namespace.apply(key)
    }

    fn keys(&self, keys: &[&str]) -> Vec<String> {
        keys.iter().map(|key| self.key(key)).collect()
    }

    async fn query<T: FromRedisValue>(&self, command: &Cmd) -> Result<T, Error> {
        Ok(command.query_async(&mut self.connection.clone()).await?)
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use litellm_redis_testing::RedisServer;
    use rstest::rstest;

    use super::{RedisClient, Ttl};
    use crate::{DEFAULT_TIMEOUT, Endpoint, Error, HostEndpoint, Namespace, RedisConfig};

    async fn client(server: &RedisServer, namespace: &str, timeout: Duration) -> RedisClient {
        RedisClient::connect(RedisConfig {
            endpoint: Endpoint::Url(server.url().to_owned()),
            namespace: Namespace::new(namespace),
            timeout,
        })
        .await
        .unwrap()
    }

    async fn client_id(client: &RedisClient) -> i64 {
        redis::cmd("CLIENT")
            .arg("ID")
            .query_async(&mut client.connection.clone())
            .await
            .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn namespaces_isolate_keys_on_the_server() {
        let server = RedisServer::start().await.unwrap();
        let a = client(&server, "a", DEFAULT_TIMEOUT).await;
        let b = client(&server, "b", DEFAULT_TIMEOUT).await;
        let bare = client(&server, "", DEFAULT_TIMEOUT).await;

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
    async fn a_rejected_command_keeps_the_connection() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", DEFAULT_TIMEOUT).await;
        client.rpush("list", "item").await.unwrap();
        let before = client_id(&client).await;

        let rejected = client.incr_by("list", 1).await;

        assert!(matches!(rejected, Err(Error::Server(_))), "{rejected:?}");
        assert_eq!(client.incr_by("counter", 2).await.unwrap(), 2);
        assert_eq!(client_id(&client).await, before);
    }

    #[rstest]
    #[tokio::test]
    async fn a_reply_of_another_type_is_a_decode_error_and_a_missing_key_is_none() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", DEFAULT_TIMEOUT).await;
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
    async fn a_timed_out_command_does_not_leak_its_reply_into_the_next_one() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", Duration::from_millis(500)).await;
        client
            .set("first", "1", Duration::from_secs(60))
            .await
            .unwrap();
        client
            .set("second", "2", Duration::from_secs(60))
            .await
            .unwrap();

        server.freeze().await.unwrap();
        let timed_out = client.get::<String>("first").await;
        server.thaw().await.unwrap();

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
    async fn the_same_client_recovers_after_redis_restarts() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", Duration::from_millis(500)).await;

        server.restart().await.unwrap();

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
    async fn empty_batches_succeed_without_asking_redis() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", DEFAULT_TIMEOUT).await;

        assert_eq!(
            client.mget::<String>(&[]).await.unwrap(),
            Vec::<Option<String>>::new()
        );
        assert_eq!(client.del(&[]).await.unwrap(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn mget_returns_values_in_input_order_with_gaps() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "ns", DEFAULT_TIMEOUT).await;
        client.set("a", "1", Duration::from_secs(60)).await.unwrap();
        client.set("c", "3", Duration::from_secs(60)).await.unwrap();

        let values = client.mget::<String>(&["c", "b", "a"]).await.unwrap();

        assert_eq!(
            values,
            vec![Some("3".to_owned()), None, Some("1".to_owned())]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn set_nx_only_writes_a_missing_key() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", DEFAULT_TIMEOUT).await;

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
    async fn ttl_reports_missing_persistent_and_remaining_time() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", DEFAULT_TIMEOUT).await;
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
    async fn incr_by_float_accumulates_fractions() {
        let server = RedisServer::start().await.unwrap();
        let client = client(&server, "", DEFAULT_TIMEOUT).await;

        client.incr_by_float("spend", 0.25).await.unwrap();

        assert_eq!(client.incr_by_float("spend", 0.5).await.unwrap(), 0.75);
    }

    #[rstest]
    #[tokio::test]
    async fn a_host_endpoint_connects_to_its_database() {
        let server = RedisServer::start().await.unwrap();
        let port = server.url().rsplit(':').next().unwrap().parse().unwrap();
        let in_db_3 = RedisClient::connect(RedisConfig {
            endpoint: Endpoint::Host(HostEndpoint {
                host: "127.0.0.1".to_owned(),
                port,
                username: None,
                password: None,
                tls: false,
                db: 3,
            }),
            namespace: Namespace::default(),
            timeout: DEFAULT_TIMEOUT,
        })
        .await
        .unwrap();
        let in_db_0 = client(&server, "", DEFAULT_TIMEOUT).await;

        in_db_3
            .set("k", "3", Duration::from_secs(60))
            .await
            .unwrap();

        assert_eq!(in_db_0.get::<String>("k").await.unwrap(), None);
        let via_url = RedisClient::connect(RedisConfig {
            endpoint: Endpoint::Url(format!("{}/3", server.url())),
            namespace: Namespace::default(),
            timeout: DEFAULT_TIMEOUT,
        })
        .await
        .unwrap();
        assert_eq!(
            via_url.get::<String>("k").await.unwrap().as_deref(),
            Some("3")
        );
    }
}
