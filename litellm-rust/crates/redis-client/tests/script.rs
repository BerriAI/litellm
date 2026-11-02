mod support;

use std::time::Duration;

use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;

use litellm_redis_client::Script;
use litellm_redis_client::{DEFAULT_TIMEOUT, Error, Ttl};
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

const INCREMENT_WITH_FLOOR: &str = "local count = redis.call('INCRBY', KEYS[1], ARGV[1]) \
if count < 0 then count = redis.call('INCRBY', KEYS[1], -count) end \
if redis.call('TTL', KEYS[1]) < 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end \
return count";

#[rstest]
#[tokio::test]
async fn pythons_increment_with_floor_script_floors_at_zero_and_expires_in_one_call(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(false, true)] pipelined: bool,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "ns", DEFAULT_TIMEOUT);
    let floor = Script::new(INCREMENT_WITH_FLOOR);
    server
        .cli_on_every_node(&["SCRIPT", "FLUSH"])
        .await
        .unwrap();

    let count = match pipelined {
        false => client
            .eval::<i64>(&floor, &["slots"], (-5, 60))
            .await
            .unwrap(),
        true => {
            let results = client
                .run(
                    &client
                        .pipeline()
                        .eval(&floor, &["slots"], (-5, 60))
                        .get("slots"),
                )
                .await;
            assert_eq!(
                results[1].as_ref().unwrap(),
                &redis::Value::BulkString(b"0".to_vec())
            );
            redis::from_redis_value(results[0].as_ref().unwrap().clone()).unwrap()
        }
    };

    assert_eq!(count, 0);
    assert_eq!(client.get::<i64>("slots").await.unwrap(), Some(0));
    assert!(matches!(
        client.ttl("slots").await.unwrap(),
        Ttl::Expires(left) if left > Duration::from_secs(50) && left <= Duration::from_secs(60)
    ));
}
