mod support;

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use litellm_circuit_breaker::{Breaker, Config, Failure, State};
use litellm_redis_client::{
    Credentials, DEFAULT_TIMEOUT, Error, Namespace, Node, RedisClient, RedisConfig, Sentinel,
    Target, Timeouts,
};
use litellm_redis_testing::{RedisDeployment, RedisSentinel, SERVICE_NAME, Topology, eventually};
use rstest::rstest;
use support::{connect, connect_guarded};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const CHANNEL: &str = "litellm_proxy.auth_cache_invalidation";

fn listen(client: &RedisClient, channel: &str) -> UnboundedReceiver<Vec<u8>> {
    let (sender, receiver) = unbounded_channel();
    let mut messages = client.subscribe(channel);
    tokio::spawn(async move {
        while let Some(message) = messages.next().await {
            if sender.send(message).is_err() {
                return;
            }
        }
    });
    receiver
}

async fn delivered(
    publisher: &RedisClient,
    channel: &str,
    received: &mut UnboundedReceiver<Vec<u8>>,
    payload: &[u8],
) {
    eventually(async || {
        publisher.publish(channel, &[payload.to_vec()]).await.ok()?;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(200);
        while let Ok(Some(message)) = tokio::time::timeout_at(deadline, received.recv()).await {
            if message == payload {
                return Some(());
            }
        }
        None
    })
    .await;
}

#[rstest]
#[tokio::test]
async fn a_subscriber_receives_raw_bytes_on_the_channel_name_without_the_namespace(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let publisher = connect(&server, "ns", DEFAULT_TIMEOUT);
    let plain = connect(&server, "", DEFAULT_TIMEOUT);
    let mut received = listen(&plain, CHANNEL);

    delivered(
        &publisher,
        CHANNEL,
        &mut received,
        b"{\"cache_key\":\"k\"}\xff",
    )
    .await;
}

#[rstest]
#[tokio::test]
async fn a_subscriber_resubscribes_after_redis_restarts(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", Duration::from_millis(500));
    let mut received = listen(&client, CHANNEL);
    delivered(&client, CHANNEL, &mut received, b"before").await;

    server.restart().await.unwrap();

    delivered(&client, CHANNEL, &mut received, b"after").await;
}

#[rstest]
#[tokio::test]
async fn a_sentinel_subscriber_receives_from_the_master() {
    let server = RedisSentinel::start().await.unwrap();
    let (host, port) = server.sentinel();
    let client = RedisClient::new(RedisConfig {
        target: Target::Sentinel(Sentinel {
            monitors: vec![Node {
                host: host.to_owned(),
                port,
            }],
            service_name: SERVICE_NAME.to_owned(),
            monitor_auth: None,
            tls: None,
            db: 0,
        }),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::default(),
        breaker: None,
        events: None,
        health_check: None,
        keepalive: true,
    })
    .unwrap();
    let mut received = listen(&client, CHANNEL);

    delivered(&client, CHANNEL, &mut received, b"payload").await;
}

async fn publishes_received(server: &RedisDeployment) -> u64 {
    let replies = match server {
        RedisDeployment::Standalone(server) => {
            vec![server.cli(&["INFO", "commandstats"]).await.unwrap()]
        }
        RedisDeployment::Cluster(cluster) => cluster
            .cli_outputs(&["INFO", "commandstats"])
            .await
            .unwrap(),
    };
    replies
        .iter()
        .flat_map(|reply| reply.lines())
        .filter_map(|line| line.trim().strip_prefix("cmdstat_publish:calls="))
        .filter_map(|rest| rest.split(',').next()?.parse::<u64>().ok())
        .sum()
}

// main published through redis-py's pub/sub client, which the breaker does not wrap, so an
// invalidation went out the moment Redis was back instead of waiting out the recovery timeout
#[rstest]
#[tokio::test]
async fn an_open_breaker_neither_refuses_nor_counts_a_publish(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let breaker = Breaker::new(
        Config {
            failure_threshold: 2,
            recovery_timeout: Duration::from_secs(60),
            timeout_min_duration: Duration::ZERO,
        },
        Box::new(|_| {}),
    );
    let client = connect_guarded(
        &server,
        "",
        Duration::from_millis(300),
        Some(breaker.clone()),
    );
    breaker.failed(Failure::Hard, Instant::now());
    breaker.failed(Failure::Hard, Instant::now());

    let alone = client.publish(CHANNEL, &[b"payload".to_vec()]).await;
    let pipelined = client
        .publish(CHANNEL, &[b"first".to_vec(), b"second".to_vec()])
        .await;
    let gated = client.ping().await;

    assert!(alone.is_ok(), "{alone:?}");
    assert!(pipelined.is_ok(), "{pipelined:?}");
    assert!(matches!(gated, Err(Error::BreakerOpen)), "{gated:?}");
    assert!(
        matches!(breaker.state(), State::Open { .. }),
        "{:?}",
        breaker.state()
    );
    assert_eq!(publishes_received(&server).await, 3);
}

#[rstest]
#[tokio::test]
async fn the_first_publish_after_redis_is_back_is_sent() {
    let server = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let client = connect(&server, "", Duration::from_millis(500));
    client
        .publish(CHANNEL, &[b"before".to_vec()])
        .await
        .unwrap();
    server.stop().await.unwrap();
    let while_down = client.publish(CHANNEL, &[b"lost".to_vec()]).await;
    server.resume().await.unwrap();
    eventually(async || server.cli_on_every_node(&["PING"]).await.ok()).await;

    let once_back = client.publish(CHANNEL, &[b"after".to_vec()]).await;

    assert!(
        matches!(while_down, Err(Error::Transport(_))),
        "{while_down:?}"
    );
    assert!(once_back.is_ok(), "{once_back:?}");
    assert_eq!(publishes_received(&server).await, 1);
}

// redis-py's PubSub subscribes on a RESP2 connection, so it never sends HELLO
#[rstest]
#[tokio::test]
async fn a_subscription_speaks_resp2_like_redis_py(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    server
        .cli_on_every_node(&["CONFIG", "RESETSTAT"])
        .await
        .unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    let mut received = listen(&client, CHANNEL);

    delivered(&client, CHANNEL, &mut received, b"evict").await;

    let stats = support::command_stats(&server).await;
    assert!(support::calls(&stats, "subscribe") > 0, "{stats}");
    assert_eq!(support::calls(&stats, "hello"), 0, "{stats}");
}
