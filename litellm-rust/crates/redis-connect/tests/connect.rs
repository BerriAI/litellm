use std::{collections::BTreeMap, sync::Arc, time::Duration};

use litellm_redis_connect::{Error, HostServices, connect};
use litellm_redis_testing::{
    RedisDeployment, RedisSentinel, RedisTlsServer, SERVICE_NAME, Topology, eventually,
};
use litellm_redis_types::{ProviderHandle, Resolution};
use rstest::rstest;

// a host hands its credential_provider object over as a handle, so these tests do the same
fn resolved(kwargs: serde_json::Value) -> Resolution {
    let provider = kwargs
        .get("credential_provider")
        .map(|_| ProviderHandle::new(1));
    let kwargs = serde_json::from_value::<BTreeMap<String, serde_json::Value>>(kwargs).unwrap();
    litellm_redis_resolve::interpret(&litellm_redis_resolve::kwargs(kwargs), provider).unwrap()
}

fn services() -> HostServices {
    HostServices {
        events: None,
        http: litellm_http::Client::for_test(),
        environment: Arc::new(|_: &str| None),
    }
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

    let pinged = connect(resolved(kwargs), services(), None)
        .unwrap()
        .ping()
        .await;

    assert_eq!(pinged.is_ok(), connects, "{pinged:?}");
}

#[rstest]
#[case::right_password("right", true)]
#[case::wrong_password("wrong", false)]
#[tokio::test]
async fn static_credentials_authenticate(
    #[case] password: &str,
    #[case] connects: bool,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    server
        .cli_on_every_node(&["ACL", "SETUSER", "default", "on", ">right"])
        .await
        .unwrap();
    let kwargs = match &server {
        RedisDeployment::Standalone(server) => serde_json::json!({
            "host": "127.0.0.1",
            "port": port(server.url()),
            "password": password,
            "socket_timeout": 2,
        }),
        RedisDeployment::Cluster(cluster) => serde_json::json!({
            "startup_nodes": cluster.nodes().map(|(host, port)| serde_json::json!({
                "host": host,
                "port": port,
            })).collect::<Vec<_>>(),
            "password": password,
            "socket_timeout": 2,
        }),
    };

    let pinged = connect(resolved(kwargs), services(), None)
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
    let client = connect(resolved(kwargs), services(), None).unwrap();
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
fn a_python_credential_provider_fails_at_boot() {
    let error = connect(
        resolved(
            serde_json::json!({"host": "redis", "credential_provider": true})),
        services(),
        None,
    )
    .err()
    .unwrap();

    assert!(
        matches!(error, Error::NoHostCredentialProvider),
        "{error:?}"
    );
}

#[rstest]
#[case::profile("AWS_PROFILE")]
#[case::role("AWS_ROLE_ARN")]
#[case::web_identity("AWS_WEB_IDENTITY_TOKEN_FILE")]
fn config_only_aws_identities_fail_before_connecting(#[case] setting: &'static str) {
    let configured = format!("{}-override", std::env::var(setting).unwrap_or_default());
    let host = HostServices {
        environment: Arc::new(move |name: &str| (name == setting).then(|| configured.clone())),
        ..services()
    };
    let error = connect(
        resolved(
            serde_json::json!({
                "host": "redis", "ssl": true, "aws_iam_auth": true,
                "aws_iam_user_name": "user", "aws_iam_cache_name": "cache",
                "aws_iam_region": "us-east-1",
            })),
        host,
        None,
    )
    .err()
    .expect("must not select ambient credentials");

    assert!(matches!(
        error,
        Error::ProcessEnvironmentRequired { setting: rejected } if rejected == setting
    ));
}
