mod support;

use std::time::Duration;

use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;

use litellm_redis_client::Script;
use litellm_redis_client::{DEFAULT_TIMEOUT, Error};
use support::{KEEP, connect};

#[rstest]
#[tokio::test]
async fn script_keys_are_namespaced(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "a", DEFAULT_TIMEOUT);
    let bare = connect(&server, "", DEFAULT_TIMEOUT);
    let read = Script::new("return redis.call('GET', KEYS[1])");
    bare.set("a:x", "1", Duration::from_secs(60)).await.unwrap();

    let value: Option<String> = client
        .eval(&read, &["x"], Vec::<String>::new())
        .await
        .unwrap();

    assert_eq!(value.as_deref(), Some("1"));
}

#[rstest]
#[tokio::test]
async fn a_failing_script_is_a_server_error_and_keeps_the_connection(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    let incr = Script::new("return redis.call('INCRBY', KEYS[1], ARGV[1])");
    client.rpush("list", "item", KEEP).await.unwrap();

    let failed = client.eval::<i64>(&incr, &["list"], 1).await;

    assert!(matches!(failed, Err(Error::Server(_))), "{failed:?}");
    assert_eq!(client.eval::<i64>(&incr, &["counter"], 3).await.unwrap(), 3);
}

#[rstest]
#[tokio::test]
async fn a_flushed_script_is_loaded_again(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    let incr = Script::new("return redis.call('INCRBY', KEYS[1], ARGV[1])");
    client.eval::<i64>(&incr, &["counter"], 1).await.unwrap();

    server
        .cli_on_every_node(&["SCRIPT", "FLUSH"])
        .await
        .unwrap();

    assert_eq!(client.eval::<i64>(&incr, &["counter"], 1).await.unwrap(), 2);
}
