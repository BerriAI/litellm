mod support;

use std::time::Duration;

use litellm_redis_testing::{
    RedisDeployment, RedisSentinel, RedisServer, SERVICE_NAME, Topology, eventually,
};
use rstest::rstest;

use litellm_redis_client::{
    ClusterNode, Credentials, Endpoint, Error, HostEndpoint, Namespace, RedisClient, RedisConfig,
    SentinelEndpoint, Timeouts,
};
use litellm_secret_value::SecretValue;
use support::connect_as;

#[rstest]
#[tokio::test]
async fn an_out_of_range_database_is_a_server_rejection() {
    let server = RedisServer::start().await.unwrap();
    let port = server.url().rsplit(':').next().unwrap().parse().unwrap();
    let client = RedisClient::new(RedisConfig {
        endpoint: Endpoint::Host(HostEndpoint {
            host: "127.0.0.1".into(),
            port,
            tls: None,
            db: 999,
        }),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_secs(2), None),
        breaker: None,
        events: None,
    })
    .unwrap();
    let result = client.ping().await;
    assert!(matches!(result, Err(Error::Server(_))), "{result:?}");
}

#[rstest]
#[tokio::test]
async fn a_rejected_credential_is_a_transport_failure() {
    let server = RedisDeployment::start(Topology::Standalone).await.unwrap();
    server
        .cli_on_every_node(&["ACL", "SETUSER", "default", "on", ">right"])
        .await
        .unwrap();
    let with_password = |password: &str| {
        connect_as(
            &server,
            Credentials {
                username: None,
                password: Some(SecretValue::new(password)),
            },
        )
    };
    with_password("right").ping().await.unwrap();

    let rejected = with_password("wrong").ping().await;

    assert!(matches!(rejected, Err(Error::Transport(_))), "{rejected:?}");
}

fn sentinel_client(sentinel: &RedisSentinel) -> RedisClient {
    let (host, port) = sentinel.sentinel();
    RedisClient::new(RedisConfig {
        endpoint: Endpoint::Sentinel(SentinelEndpoint {
            sentinels: vec![ClusterNode {
                host: host.to_owned(),
                port,
            }],
            service_name: SERVICE_NAME.to_owned(),
            sentinel_password: None,
            tls: None,
            db: 0,
        }),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_secs(1), None),
        breaker: None,
        events: None,
    })
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn sentinel_writes_follow_the_master_through_a_failover() {
    let sentinel = RedisSentinel::start().await.unwrap();
    let client = sentinel_client(&sentinel);
    client
        .set("before", "1", Duration::from_secs(600))
        .await
        .unwrap();
    eventually(async || {
        let replica =
            redis::Client::open(format!("redis://127.0.0.1:{}", sentinel.replica_port())).ok()?;
        let mut replica = replica.get_multiplexed_async_connection().await.ok()?;
        let value: Option<String> = redis::cmd("GET")
            .arg("before")
            .query_async(&mut replica)
            .await
            .ok()?;
        value
    })
    .await;

    let promoted = sentinel.failover().await.unwrap();
    eventually(async || {
        client
            .set("after", "2", Duration::from_secs(600))
            .await
            .ok()
    })
    .await;

    assert_eq!(promoted, sentinel.replica_port());
    assert_eq!(
        client.get::<String>("before").await.unwrap().as_deref(),
        Some("1")
    );
    let promoted = redis::Client::open(format!("redis://127.0.0.1:{promoted}")).unwrap();
    let mut promoted = promoted.get_multiplexed_async_connection().await.unwrap();
    let after: Option<String> = redis::cmd("GET")
        .arg("after")
        .query_async(&mut promoted)
        .await
        .unwrap();
    assert_eq!(after.as_deref(), Some("2"));
}
