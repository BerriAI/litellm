use std::{collections::HashMap, time::Duration};

use litellm_config::{RedisSettings, ResolvedRedis, SecretSnapshot};
use litellm_redis_connect::{Error, connect};
use litellm_redis_testing::{RedisSentinel, RedisServer, RedisTlsServer, SERVICE_NAME, eventually};
use rstest::rstest;

fn resolved(kwargs: serde_json::Value, environment: &[(&str, &str)]) -> ResolvedRedis {
    let settings: RedisSettings = serde_json::from_value(kwargs).unwrap();
    let environment: HashMap<&str, &str> = environment.iter().copied().collect();
    let lookup = |name: &str| environment.get(name).map(|value| (*value).to_owned());
    settings
        .resolve(
            &SecretSnapshot::read(settings.secret_names(), &lookup),
            &lookup,
        )
        .unwrap()
}

fn port(url: &str) -> u16 {
    url.rsplit(':').next().unwrap().parse().unwrap()
}

#[rstest]
#[case::trusted_ca(true, true)]
#[case::system_roots_only(false, false)]
#[tokio::test]
async fn python_tls_kwargs_reach_the_server(#[case] with_ca: bool, #[case] connects: bool) {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let server = RedisTlsServer::start(false).await.unwrap();
    let kwargs = serde_json::json!({
        "host": server.host(),
        "port": server.port(),
        "ssl": true,
        "ssl_ca_certs": with_ca.then(|| server.files().ca.display().to_string()),
        "socket_timeout": 2,
    });

    let pinged = connect(resolved(kwargs, &[]), None).unwrap().ping().await;

    assert_eq!(pinged.is_ok(), connects, "{pinged:?}");
}

#[rstest]
#[case::configured_password(Some("right"), &[], true)]
#[case::environment_password(None, &[("REDIS_PASSWORD", "right")], true)]
#[case::wrong_password(Some("wrong"), &[], false)]
#[tokio::test]
async fn static_credentials_from_kwargs_or_environment_authenticate(
    #[case] password: Option<&str>,
    #[case] environment: &[(&str, &str)],
    #[case] connects: bool,
) {
    let server = RedisServer::start().await.unwrap();
    server
        .cli(&["CONFIG", "SET", "requirepass", "right"])
        .await
        .unwrap();
    let kwargs = serde_json::json!({
        "host": "127.0.0.1",
        "port": port(server.url()),
        "password": password,
        "socket_timeout": 2,
    });

    let pinged = connect(resolved(kwargs, environment), None)
        .unwrap()
        .ping()
        .await;

    assert_eq!(pinged.is_ok(), connects, "{pinged:?}");
}

#[rstest]
#[tokio::test]
async fn python_sentinel_kwargs_follow_the_master_through_a_failover() {
    let sentinel = RedisSentinel::start().await.unwrap();
    let (host, sentinel_port) = sentinel.sentinel();
    let kwargs = serde_json::json!({
        "sentinel_nodes": [[host, sentinel_port]],
        "service_name": SERVICE_NAME,
        "socket_timeout": 1,
    });
    let client = connect(resolved(kwargs, &[]), None).unwrap();
    client
        .set("before", "1", Duration::from_secs(600))
        .await
        .unwrap();

    let promoted = sentinel.failover().await.unwrap();
    eventually(async || {
        client
            .set("after", "2", Duration::from_secs(600))
            .await
            .ok()
    })
    .await;

    assert_eq!(promoted, sentinel.replica_port());
    assert_eq!(sentinel.master_port().await.unwrap(), promoted);
    assert_eq!(
        client.get::<String>("after").await.unwrap().as_deref(),
        Some("2")
    );
}

#[rstest]
#[case::credential_provider(serde_json::json!({"host": "redis", "credential_provider": true}), &[], None)]
#[case::gcp(serde_json::json!({"host": "redis", "gcp_service_account": "account"}), &[], Some("gcp_service_account"))]
#[case::azure(serde_json::json!({"host": "redis"}), &[("REDIS_AZURE_AD_TOKEN", "true")], Some("azure_redis_ad_token"))]
#[case::aws(
    serde_json::json!({"host": "cache", "ssl": true, "aws_iam_auth": true, "aws_iam_user_name": "user", "aws_iam_cache_name": "cache", "aws_iam_region": "us-east-1"}),
    &[],
    Some("aws_iam_auth")
)]
fn auth_this_host_cannot_build_fails_at_boot_naming_the_setting(
    #[case] kwargs: serde_json::Value,
    #[case] environment: &[(&str, &str)],
    #[case] cloud_setting: Option<&str>,
) {
    let error = connect(resolved(kwargs, environment), None).err().unwrap();

    match (error, cloud_setting) {
        (Error::NoHostCredentialProvider, None) => {}
        (Error::CloudAuth { setting }, Some(expected)) => assert_eq!(setting, expected),
        (error, _) => panic!("unexpected {error:?}"),
    }
}
