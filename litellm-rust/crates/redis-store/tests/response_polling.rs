mod support;

use std::time::Duration;

use litellm_redis_client::Ttl;
use litellm_redis_store::{
    Error,
    response_polling::{self, State},
};
use rstest::rstest;
use support::{Server, Topology};

const POLLING_ID: &str = "litellm_poll_x";
const KEY: &str = "litellm:polling:response:litellm_poll_x";
const DOCUMENT: &str = "{\"id\":\"litellm_poll_x\",\"model\":\"caf\u{e9} \u{1f600}\",\"note\":\"tab\\there \u{7f}\"}";
// `json.dumps(DOCUMENT)` from CPython, the bytes main's `RedisCache.async_set_cache` writes
const PYTHON_BYTES: &str = r#""{\"id\":\"litellm_poll_x\",\"model\":\"café 😀\",\"note\":\"tab\\there \u007f\"}""#;

fn state() -> State {
    State::parse(DOCUMENT.to_owned()).unwrap()
}

#[rstest]
#[tokio::test]
async fn put_writes_the_bytes_python_writes_and_expires(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = Server::start(topology).await;
    let client = server.client();

    response_polling::put(&client, POLLING_ID, &state(), Duration::from_secs(7200))
        .await
        .unwrap();

    let raw: Option<Vec<u8>> = client.get(KEY).await.unwrap();
    assert_eq!(raw.as_deref(), Some(PYTHON_BYTES.as_bytes()));
    let Ttl::Expires(remaining) = client.ttl(KEY).await.unwrap() else {
        panic!("the state has no expiry");
    };
    assert!(remaining > Duration::from_secs(7190) && remaining <= Duration::from_secs(7200));
}

#[rstest]
#[tokio::test]
async fn get_reads_what_python_wrote_until_deleted(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = Server::start(topology).await;
    let client = server.client();
    client
        .set(KEY, PYTHON_BYTES, Duration::from_secs(60))
        .await
        .unwrap();

    assert_eq!(
        response_polling::get(&client, POLLING_ID).await.unwrap(),
        Some(state())
    );
    response_polling::delete(&client, POLLING_ID).await.unwrap();
    assert_eq!(response_polling::get(&client, POLLING_ID).await.unwrap(), None);
}

#[rstest]
#[case::not_json("not json")]
#[case::not_a_document_string(r#"{"status": "queued"}"#)]
#[case::document_not_an_object(r#""[1, 2]""#)]
#[tokio::test]
async fn get_rejects_a_value_in_another_format(#[case] stored: &str) {
    let server = Server::start(Topology::Standalone).await;
    let client = server.client();
    client
        .set(KEY, stored, Duration::from_secs(60))
        .await
        .unwrap();

    let read = response_polling::get(&client, POLLING_ID).await;

    assert!(matches!(read, Err(Error::Format(_))), "{read:?}");
}
