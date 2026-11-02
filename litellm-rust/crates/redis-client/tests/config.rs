use rstest::rstest;

use litellm_redis_client::{Credentials, Endpoint, SentinelEndpoint, TlsOptions};
use litellm_secret_value::SecretValue;

#[rstest]
#[case::url(format!("{:?}", Endpoint::Url { url: SecretValue::new("redis://user:credential-value@cache:6379/0"), tls: TlsOptions::verified() }))]
#[case::static_password(format!("{:?}", Credentials { username: None, password: Some(SecretValue::new("credential-value")) }))]
#[case::sentinel_password(format!("{:?}", Endpoint::Sentinel(SentinelEndpoint {
    sentinels: Vec::new(),
    service_name: "primary".to_owned(),
    sentinel_password: Some(SecretValue::new("credential-value")),
    tls: None,
    db: 0,
})))]
fn debug_output_never_shows_the_secret(#[case] debug: String) {
    assert!(!debug.contains("credential-value"), "{debug}");
    assert!(debug.contains("[REDACTED]"), "{debug}");
}

#[derive(Clone, Copy, Debug)]
enum Ca {
    Trusted,
    SystemRoots,
}

async fn ping_over_tls(
    require_client_certificate: bool,
    host: &str,
    ca: Ca,
    verify_certificate: bool,
    check_hostname: bool,
    client_certificate: bool,
) -> Result<(), litellm_redis_client::Error> {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let server = litellm_redis_testing::RedisTlsServer::start(require_client_certificate)
        .await
        .unwrap();
    let files = server.files();
    let tls = TlsOptions {
        ca_certs: matches!(ca, Ca::Trusted).then(|| files.ca.clone()),
        certfile: client_certificate.then(|| files.client_cert.clone()),
        keyfile: client_certificate.then(|| files.client_key.clone()),
        verify_certificate,
        check_hostname,
    };
    let client = litellm_redis_client::RedisClient::new(litellm_redis_client::RedisConfig {
        endpoint: Endpoint::Host(litellm_redis_client::HostEndpoint {
            host: host.to_owned(),
            port: server.port(),
            tls: Some(tls),
            db: 0,
        }),
        credentials: Credentials::NONE,
        namespace: litellm_redis_client::Namespace::default(),
        timeouts: litellm_redis_client::Timeouts::default(),
        breaker: None,
        events: None,
    })
    .unwrap();
    client.ping().await
}

#[rstest]
#[case::trusted_ca_without_hostname_check_like_redis_py(
    "127.0.0.1",
    Ca::Trusted,
    true,
    false,
    true
)]
#[case::hostname_check_with_a_matching_name("localhost", Ca::Trusted, true, true, true)]
#[case::hostname_check_against_an_unlisted_address("127.0.0.1", Ca::Trusted, true, true, false)]
#[case::an_untrusted_ca_is_refused("127.0.0.1", Ca::SystemRoots, true, false, false)]
#[case::cert_reqs_none_skips_verification("127.0.0.1", Ca::SystemRoots, false, false, true)]
#[tokio::test]
async fn tls_options_decide_which_servers_are_trusted(
    #[case] host: &str,
    #[case] ca: Ca,
    #[case] verify_certificate: bool,
    #[case] check_hostname: bool,
    #[case] connects: bool,
) {
    let result = ping_over_tls(false, host, ca, verify_certificate, check_hostname, false).await;

    assert_eq!(result.is_ok(), connects, "{result:?}");
    if let Err(error) = result {
        assert!(
            matches!(error, litellm_redis_client::Error::Transport(_)),
            "{error:?}"
        );
    }
}

#[rstest]
#[case::with_the_client_certificate(true, true)]
#[case::without_a_client_certificate(false, false)]
#[tokio::test]
async fn mutual_tls_sends_the_configured_client_certificate(
    #[case] client_certificate: bool,
    #[case] connects: bool,
) {
    let result = ping_over_tls(
        true,
        "127.0.0.1",
        Ca::Trusted,
        true,
        false,
        client_certificate,
    )
    .await;

    assert_eq!(result.is_ok(), connects, "{result:?}");
}

#[rstest]
fn a_missing_tls_file_names_the_path() {
    let tls = TlsOptions {
        ca_certs: Some("/nonexistent/ca.pem".into()),
        ..TlsOptions::verified()
    };

    let error = litellm_redis_client::RedisClient::new(litellm_redis_client::RedisConfig {
        endpoint: Endpoint::Host(litellm_redis_client::HostEndpoint {
            host: "127.0.0.1".to_owned(),
            port: 6379,
            tls: Some(tls),
            db: 0,
        }),
        credentials: Credentials::NONE,
        namespace: litellm_redis_client::Namespace::default(),
        timeouts: litellm_redis_client::Timeouts::default(),
        breaker: None,
        events: None,
    })
    .err()
    .unwrap();

    assert!(error.to_string().contains("/nonexistent/ca.pem"), "{error}");
}
