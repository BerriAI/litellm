mod support;

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use litellm_circuit_breaker::{Breaker, Config};
use litellm_redis_client::commands;
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
        publisher
            .send(commands::publish(channel, payload))
            .await
            .ok()?;
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

async fn publish_calls(server: &RedisDeployment) -> Vec<String> {
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
        .filter(|line| line.starts_with("cmdstat_publish:"))
        .map(str::to_owned)
        .collect()
}

#[rstest]
#[tokio::test]
async fn an_open_breaker_rejects_a_publish_without_sending_it(
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
    let client = connect_guarded(&server, "", Duration::from_millis(300), Some(breaker));

    server.freeze().await.unwrap();
    let opened = [client.ping().await, client.ping().await];
    let started = Instant::now();
    let rejected = client.send(commands::publish(CHANNEL, b"payload")).await;
    let rejected_in = started.elapsed();
    server.thaw().await.unwrap();

    assert!(
        opened
            .iter()
            .all(|ping| matches!(ping, Err(Error::Transport(_)))),
        "{opened:?}"
    );
    assert!(matches!(rejected, Err(Error::BreakerOpen)), "{rejected:?}");
    assert!(rejected_in < Duration::from_millis(50), "{rejected_in:?}");
    assert_eq!(publish_calls(&server).await, Vec::<String>::new());
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
