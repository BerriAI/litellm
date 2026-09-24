mod support;

use std::time::Duration;

use litellm_cache::{Error, JsonCodec, Message, MessageStream, PubSubCache};
use litellm_cache_redis::RedisCache;
use redis_test::{MockCmd, MockRedisConnection};
use support::server::PubSubServer;

type Json = JsonCodec<serde_json::Value>;

fn live(server: &PubSubServer) -> RedisCache<Json> {
    RedisCache::new(&server.url(), None, JsonCodec::new()).unwrap()
}

fn channels(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[tokio::test]
async fn publish_reports_receivers_through_the_pool() {
    let connection = MockRedisConnection::new(vec![MockCmd::new(
        redis::cmd("PUBLISH")
            .arg("events")
            .arg(b"payload".as_slice()),
        Ok(redis::Value::Int(2)),
    )])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, Json::new());
    assert_eq!(cache.async_publish("events", b"payload").await, Ok(2));
}

#[tokio::test]
async fn subscriptions_need_a_pooled_connection() {
    let cache = RedisCache::with_connection(MockRedisConnection::new(vec![]), None, Json::new());
    assert_eq!(
        cache
            .async_subscribe(&channels(&["events"]))
            .await
            .map(|_| ())
            .unwrap_err(),
        Error::UnsupportedOperation
    );
}

#[tokio::test]
async fn subscribers_receive_published_messages_until_closed() {
    let server = PubSubServer::start();
    let cache = live(&server);
    let mut subscription = cache.async_subscribe(&channels(&["events"])).await.unwrap();
    assert_eq!(
        subscription.next_message(Some(Duration::ZERO)).await,
        Ok(None)
    );

    assert_eq!(cache.async_publish("events", b"hello").await, Ok(1));
    assert_eq!(
        subscription
            .next_message(Some(Duration::from_secs(5)))
            .await,
        Ok(Some(Message {
            channel: "events".into(),
            payload: b"hello".to_vec(),
        }))
    );
    assert_eq!(cache.async_publish("other", b"ignored").await, Ok(0));
    assert_eq!(
        subscription
            .next_message(Some(Duration::from_millis(50)))
            .await,
        Ok(None)
    );

    subscription.close().await.unwrap();
    assert_eq!(server.unsubscribed(), vec!["events".to_owned()]);
    assert_eq!(cache.async_publish("events", b"nobody").await, Ok(0));
}

#[tokio::test]
async fn a_dropped_connection_surfaces_as_unavailable() {
    let server = PubSubServer::start();
    let cache = live(&server);
    let mut subscription = cache.async_subscribe(&channels(&["events"])).await.unwrap();
    server.drop_subscribers();
    assert_eq!(
        subscription
            .next_message(Some(Duration::from_secs(5)))
            .await,
        Err(Error::Unavailable)
    );
}

#[tokio::test]
async fn subscribing_to_an_unreachable_server_fails_before_returning_a_stream() {
    let url = PubSubServer::start().url();
    let cache: RedisCache<Json> = RedisCache::new(&url, None, JsonCodec::new()).unwrap();
    assert_eq!(
        cache
            .async_subscribe(&channels(&["events"]))
            .await
            .map(|_| ())
            .unwrap_err(),
        Error::Unavailable
    );
}
