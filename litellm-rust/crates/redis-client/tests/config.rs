use rstest::rstest;

use litellm_redis_client::{Credentials, RedisUrl, Sentinel, StaticAuth, Target, TlsOptions};
use litellm_secret_value::SecretValue;

#[rstest]
#[case::url(format!("{:?}", Target::Url(RedisUrl { url: SecretValue::new("redis://user:credential-value@cache:6379/0"), tls: None, db: None, username: None, password: None })))]
#[case::static_password(format!("{:?}", Credentials::Static { username: None, password: Some(SecretValue::new("credential-value")) }))]
#[case::sentinel_password(format!("{:?}", Target::Sentinel(Sentinel {
    monitors: Vec::new(),
    service_name: "primary".to_owned(),
    monitor_auth: Some(StaticAuth { username: None, password: Some(SecretValue::new("credential-value")) }),
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
    TrustedData,
    TrustedDirectory,
    SystemRoots,
}

fn directory_holding(ca: &std::path::Path) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!("litellm-ca-dir-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::copy(ca, directory.join("ca.pem")).unwrap();
    directory
}

#[derive(Clone, Copy, Debug)]
enum ClientKey {
    Absent,
    Plain,
    Encrypted { password: Option<&'static str> },
}

async fn tls_server(require_client_certificate: bool) -> litellm_redis_testing::RedisTlsServer {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    litellm_redis_testing::RedisTlsServer::start(require_client_certificate)
        .await
        .unwrap()
}

fn tls_client(
    server: &litellm_redis_testing::RedisTlsServer,
    host: &str,
    tls: TlsOptions,
) -> Result<litellm_redis_client::RedisClient, litellm_redis_client::ConnectError> {
    litellm_redis_client::RedisClient::new(litellm_redis_client::RedisConfig {
        target: Target::Standalone(litellm_redis_client::Standalone {
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
        health_check: None,
        keepalive: true,
    })
}

fn tls_options(
    files: &litellm_redis_testing::TlsFiles,
    ca: Ca,
    verify_certificate: bool,
    check_hostname: bool,
    client_key: ClientKey,
) -> TlsOptions {
    let (keyfile, key_password) = match client_key {
        ClientKey::Absent => (None, None),
        ClientKey::Plain => (Some(files.client_key.clone()), None),
        ClientKey::Encrypted { password } => (
            Some(files.encrypted_client_key.clone()),
            password.map(SecretValue::new),
        ),
    };
    TlsOptions {
        ca_certs: matches!(ca, Ca::Trusted).then(|| files.ca.clone()),
        ca_data: matches!(ca, Ca::TrustedData).then(|| std::fs::read_to_string(&files.ca).unwrap()),
        ca_path: matches!(ca, Ca::TrustedDirectory).then(|| directory_holding(&files.ca)),
        certfile: keyfile.is_some().then(|| files.client_cert.clone()),
        keyfile,
        key_password,
        verify_certificate,
        check_hostname,
    }
}

async fn ping_over_tls(
    require_client_certificate: bool,
    host: &str,
    ca: Ca,
    verify_certificate: bool,
    check_hostname: bool,
    client_key: ClientKey,
) -> Result<(), litellm_redis_client::Error> {
    let server = tls_server(require_client_certificate).await;
    let tls = tls_options(
        server.files(),
        ca,
        verify_certificate,
        check_hostname,
        client_key,
    );
    tls_client(&server, host, tls).unwrap().ping().await
}

#[rstest]
#[case::trusted_ca_without_hostname_check_like_redis_py(
    "127.0.0.1",
    Ca::Trusted,
    true,
    false,
    true
)]
#[case::trusted_ca_as_pem_text_like_ssl_ca_data("127.0.0.1", Ca::TrustedData, true, false, true)]
#[case::trusted_ca_in_a_directory_like_ssl_ca_path(
    "127.0.0.1",
    Ca::TrustedDirectory,
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
    let result = ping_over_tls(
        false,
        host,
        ca,
        verify_certificate,
        check_hostname,
        ClientKey::Absent,
    )
    .await;

    assert_eq!(result.is_ok(), connects, "{result:?}");
    if let Err(error) = result {
        assert!(
            matches!(error, litellm_redis_client::Error::Transport(_)),
            "{error:?}"
        );
    }
}

#[rstest]
#[case::with_the_client_certificate(ClientKey::Plain, true)]
#[case::with_an_encrypted_key_and_its_ssl_password(
    ClientKey::Encrypted { password: Some(litellm_redis_testing::CLIENT_KEY_PASSWORD) },
    true
)]
#[case::without_a_client_certificate(ClientKey::Absent, false)]
#[tokio::test]
async fn mutual_tls_sends_the_configured_client_certificate(
    #[case] client_key: ClientKey,
    #[case] connects: bool,
) {
    let result = ping_over_tls(true, "127.0.0.1", Ca::Trusted, true, false, client_key).await;

    assert_eq!(result.is_ok(), connects, "{result:?}");
}

#[rstest]
#[case::without_ssl_password(None, "ssl_password is required")]
#[case::with_the_wrong_ssl_password(Some("wrong-password"), "does not decrypt the key")]
#[tokio::test]
async fn an_encrypted_key_that_cannot_be_decrypted_fails_before_connecting(
    #[case] password: Option<&'static str>,
    #[case] reason: &str,
) {
    let server = tls_server(true).await;
    let tls = tls_options(
        server.files(),
        Ca::Trusted,
        true,
        false,
        ClientKey::Encrypted { password },
    );

    let error = tls_client(&server, "127.0.0.1", tls).err().unwrap();

    let litellm_redis_client::ConnectError::TlsKey {
        path,
        reason: cause,
    } = &error
    else {
        panic!("expected a TLS key error, got {error:?}");
    };
    assert_eq!(
        *path,
        server.files().encrypted_client_key.display().to_string()
    );
    assert!(cause.to_string().contains(reason), "{cause}");
}

#[rstest]
fn a_missing_tls_file_names_the_path() {
    let tls = TlsOptions {
        ca_certs: Some("/nonexistent/ca.pem".into()),
        ..TlsOptions::verified()
    };

    let error = litellm_redis_client::RedisClient::new(litellm_redis_client::RedisConfig {
        target: Target::Standalone(litellm_redis_client::Standalone {
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
        health_check: None,
        keepalive: true,
    })
    .err()
    .unwrap();

    assert!(error.to_string().contains("/nonexistent/ca.pem"), "{error}");
}

#[rstest]
fn a_legacy_encrypted_pem_key_is_refused_with_a_conversion_hint() {
    let directory = std::env::temp_dir().join(format!("litellm-legacy-key-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let keyfile = directory.join("client.key");
    std::fs::write(
        &keyfile,
        "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-256-CBC,00112233445566778899AABBCCDDEEFF\n\nAAAA\n-----END RSA PRIVATE KEY-----\n",
    )
    .unwrap();
    let tls = TlsOptions {
        certfile: Some(keyfile.clone()),
        keyfile: Some(keyfile),
        key_password: Some(SecretValue::new("legacy-password")),
        ..TlsOptions::verified()
    };

    let error = litellm_redis_client::RedisClient::new(litellm_redis_client::RedisConfig {
        target: Target::Standalone(litellm_redis_client::Standalone {
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
        health_check: None,
        keepalive: true,
    })
    .err()
    .unwrap();

    assert!(
        matches!(
            &error,
            litellm_redis_client::ConnectError::TlsKey {
                reason: litellm_redis_client::KeyError::LegacyEncryption,
                ..
            }
        ),
        "{error:?}"
    );
}
