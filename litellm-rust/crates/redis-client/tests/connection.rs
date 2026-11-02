mod support;

use std::time::Duration;

use litellm_redis_client::{Endpoint, Error, HostEndpoint};
use litellm_redis_testing::RedisServer;
use litellm_secret_value::SecretValue;
use rstest::rstest;

use support::connect_to;

#[rstest]
#[tokio::test]
async fn an_out_of_range_database_is_a_server_rejection() {
    let server = RedisServer::start().await.unwrap();
    let port = server.url().rsplit(':').next().unwrap().parse().unwrap();
    let client = connect_to(
        Endpoint::Host(HostEndpoint {
            host: "127.0.0.1".into(),
            port,
            username: None,
            password: None,
            tls: false,
            db: 999,
        }),
        "",
        Duration::from_secs(2),
        None,
    );
    let result = client.ping().await;
    assert!(matches!(result, Err(Error::Server(_))), "{result:?}");
}

#[rstest]
#[tokio::test]
async fn a_rejected_password_is_a_transport_failure() {
    let server = RedisServer::start().await.unwrap();
    server
        .cli(&["ACL", "SETUSER", "default", "on", ">right"])
        .await
        .unwrap();
    let port = server.url().rsplit(':').next().unwrap().parse().unwrap();
    let with_password = |password: &str| {
        connect_to(
            Endpoint::Host(HostEndpoint {
                host: "127.0.0.1".to_owned(),
                port,
                username: None,
                password: Some(SecretValue::new(password)),
                tls: false,
                db: 0,
            }),
            "",
            Duration::from_secs(2),
            None,
        )
    };
    with_password("right").ping().await.unwrap();

    let rejected = with_password("wrong").ping().await;

    assert!(matches!(rejected, Err(Error::Transport(_))), "{rejected:?}");
}
