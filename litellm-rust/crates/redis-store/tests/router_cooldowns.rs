mod support;

use std::time::Duration;

use litellm_redis_client::Ttl;
use litellm_redis_store::router_cooldowns::{self, Cooldown};
use rstest::rstest;
use support::{Server, Topology};

const KEY: &str = "deployment:gpt-a:cooldown";
// `str(entry)` from CPython, the bytes main's sync `RedisCache.set_cache` writes
const PYTHON_REPR: &str = r#"{'exception_received': 'RateLimitError: "quota" exceeded café it\'s', 'status_code': '429', 'timestamp': 1761868800.25, 'cooldown_time': 5.0}"#;
// `json.dumps(entry)`, which a JSON-writing Python pod would leave behind
const PYTHON_JSON: &str = r#"{"exception_received": "RateLimitError: \"quota\" exceeded café it's", "status_code": "429", "timestamp": 1761868800.25, "cooldown_time": 5.0}"#;

fn cooldown() -> Cooldown {
    Cooldown {
        exception_received: "RateLimitError: \"quota\" exceeded café it's".to_owned(),
        status_code: "429".to_owned(),
        timestamp: 1_761_868_800.25,
        cooldown_time: 5.0,
    }
}

#[rstest]
#[tokio::test]
async fn put_writes_the_bytes_python_writes_and_expires(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = Server::start(topology).await;
    let client = server.client();

    router_cooldowns::put(&client, "gpt-a", &cooldown(), Duration::from_secs(5))
        .await
        .unwrap();

    let raw: Option<Vec<u8>> = client.get(KEY).await.unwrap();
    assert_eq!(raw.as_deref(), Some(PYTHON_REPR.as_bytes()));
    let Ttl::Expires(remaining) = client.ttl(KEY).await.unwrap() else {
        panic!("the cooldown has no expiry");
    };
    assert!(remaining <= Duration::from_secs(5));
}

#[rstest]
#[case::python_repr(PYTHON_REPR)]
#[case::json(PYTHON_JSON)]
#[tokio::test]
async fn get_many_reads_both_formats_python_writes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] stored: &str,
) {
    let server = Server::start(topology).await;
    let client = server.client();
    client
        .set(KEY, stored, Duration::from_secs(60))
        .await
        .unwrap();

    let read = router_cooldowns::get_many(&client, &["gpt-a", "gpt-b"])
        .await
        .unwrap();

    assert!(matches!(read.as_slice(), [Some(Ok(found)), None] if *found == cooldown()), "{read:?}");
}

#[tokio::test]
async fn get_many_reports_an_unreadable_entry_without_hiding_the_others() {
    let server = Server::start(Topology::Standalone).await;
    let client = server.client();
    client
        .set(KEY, "not a cooldown", Duration::from_secs(60))
        .await
        .unwrap();
    router_cooldowns::put(&client, "gpt-b", &cooldown(), Duration::from_secs(60))
        .await
        .unwrap();

    let read = router_cooldowns::get_many(&client, &["gpt-a", "gpt-b"])
        .await
        .unwrap();

    assert!(matches!(read.as_slice(), [Some(Err(_)), Some(Ok(found))] if *found == cooldown()), "{read:?}");
}
