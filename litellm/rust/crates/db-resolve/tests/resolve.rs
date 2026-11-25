use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use litellm_config::{Config, GeneralSettings};
use litellm_db::{ClientIdentity, Endpoint, PoolLimits, SslMode};
use litellm_db_resolve::{Database, Error, Expected, Limits, Overlay, Rule, WRITER_URL, interpret};
use litellm_secret_value::SecretValue;
use litellm_secrets::{OidcResolver, SecretManagerState, SecretResolver};
use rstest::rstest;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

const WRITER: &str = "postgresql://app:secret@db.internal/proxy";

fn general(yaml: &str) -> GeneralSettings {
    Config::from_yaml(yaml).unwrap().general_settings
}

fn secrets(variables: &[(&str, &str)]) -> SecretResolver {
    let variables: BTreeMap<String, String> = variables
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::default()),
        Arc::new(move |name: &str| variables.get(name).cloned()),
        OidcResolver::new(litellm_http::Client::for_test()),
    )
}

async fn try_resolve(yaml: &str, variables: &[(&str, &str)]) -> Result<Option<Database>, Error> {
    litellm_db_resolve::resolve(&general(yaml), &secrets(variables)).await
}

async fn resolve(yaml: &str, variables: &[(&str, &str)]) -> Database {
    try_resolve(yaml, variables).await.unwrap().unwrap()
}

async fn resolve_url(url: &str) -> Database {
    resolve("model_list: []", &[("DATABASE_URL", url)]).await
}

fn endpoint() -> Endpoint {
    Endpoint {
        host: "db.internal".to_owned(),
        port: 5432,
        socket: None,
        username: "app".to_owned(),
        password: Some(SecretValue::new("secret")),
        database: "proxy".to_owned(),
        ssl_mode: SslMode::Prefer,
        ssl_root_cert: None,
        ssl_client: None,
        application_name: Some("litellm-native".to_owned()),
        options: Vec::new(),
        statement_cache_capacity: 100,
        pool: PoolLimits {
            max_connections: 10,
            acquire_timeout_seconds: Some(60.0),
            idle_timeout_seconds: Some(60.0),
            socket_timeout_seconds: None,
        },
    }
}

#[tokio::test]
async fn the_configured_url_wins_and_its_settings_override_what_the_url_pins() {
    let yaml = r#"
general_settings:
  database_url: "postgresql://app:p%40ss@DB.internal:6543/proxy?connection_limit=50&pool_timeout=5&schema=tenant&options=-c%20statement_timeout%3D5000"
  database_connection_pool_limit: 20
  database_socket_timeout: 7
  database_statement_timeout: 30
  database_lock_timeout: 2.5
"#;

    let database = resolve(
        yaml,
        &[("DATABASE_URL", "postgresql://other@elsewhere/other")],
    )
    .await;

    assert_eq!(
        database,
        Database {
            writer: Endpoint {
                port: 6543,
                password: Some(SecretValue::new("p@ss")),
                options: vec![
                    ("search_path".to_owned(), "\"tenant\"".to_owned()),
                    ("statement_timeout".to_owned(), "5000".to_owned()),
                    ("lock_timeout".to_owned(), "2500".to_owned()),
                ],
                pool: PoolLimits {
                    max_connections: 20,
                    socket_timeout_seconds: Some(7.0),
                    ..endpoint().pool
                },
                ..endpoint()
            },
            reader: None,
            ignored: Vec::new(),
        }
    );
}

#[tokio::test]
async fn discrete_variables_assemble_the_writer_and_a_root_cert_alone_verifies() {
    let database = resolve(
        "model_list: []",
        &[
            ("DATABASE_HOST", "db.internal"),
            ("DATABASE_PORT", "6543"),
            ("DATABASE_USERNAME", "app"),
            ("DATABASE_PASSWORD", "p@ss/word"),
            ("DATABASE_NAME", "proxy"),
            ("DATABASE_SCHEMA", "tenant"),
            ("DATABASE_SSLROOTCERT", "/etc/ssl/rds.pem"),
        ],
    )
    .await;

    assert_eq!(
        database.writer,
        Endpoint {
            port: 6543,
            password: Some(SecretValue::new("p@ss/word")),
            ssl_mode: SslMode::VerifyFull,
            ssl_root_cert: Some(PathBuf::from("/etc/ssl/rds.pem")),
            options: vec![("search_path".to_owned(), "\"tenant\"".to_owned())],
            ..endpoint()
        }
    );
}

#[tokio::test]
async fn what_a_replica_url_pins_wins_over_the_configured_pool() {
    let yaml = "general_settings:\n  database_connection_pool_limit: 20";

    let database = resolve(
        yaml,
        &[
            ("DATABASE_URL", WRITER),
            (
                "DATABASE_URL_READ_REPLICA",
                "postgresql://app:secret@replica.internal/proxy?connection_limit=3&pool_timeout=0",
            ),
        ],
    )
    .await;

    assert_eq!(database.writer.pool.max_connections, 20);
    assert_eq!(
        database.reader,
        Some(Endpoint {
            host: "replica.internal".to_owned(),
            pool: PoolLimits {
                max_connections: 3,
                acquire_timeout_seconds: None,
                ..endpoint().pool
            },
            ..endpoint()
        })
    );
}

#[tokio::test]
async fn a_replica_host_takes_the_writer_fields_it_does_not_set() {
    let database = resolve(
        "model_list: []",
        &[
            ("DATABASE_HOST", "db.internal"),
            ("DATABASE_USER", "app"),
            ("DATABASE_PASSWORD", "secret"),
            ("DATABASE_NAME", "proxy"),
            ("DATABASE_HOST_READ_REPLICA", "replica.internal"),
            ("DATABASE_PASSWORD_READ_REPLICA", "other"),
        ],
    )
    .await;

    assert_eq!(database.writer, endpoint());
    assert_eq!(
        database.reader,
        Some(Endpoint {
            host: "replica.internal".to_owned(),
            password: Some(SecretValue::new("other")),
            ..endpoint()
        })
    );
}

#[tokio::test]
async fn a_variable_is_read_only_under_its_exact_name() {
    let database = resolve(
        "model_list: []",
        &[
            ("DATABASE_HOST", "db.internal"),
            ("DATABASE_host", "elsewhere"),
            ("DATABASE_USER", "app"),
            ("DATABASE_PASSWORD", "secret"),
            ("DATABASE_NAME", "proxy"),
            ("DATABASE_url_READ_REPLICA", WRITER),
        ],
    )
    .await;

    assert_eq!(database.writer, endpoint());
    assert_eq!(database.reader, None);
}

#[tokio::test]
async fn a_configured_timeout_spelled_as_text_is_read_like_a_number() {
    let yaml = "general_settings:\n  database_statement_timeout: \"30\"";

    let database = resolve(yaml, &[("DATABASE_URL", WRITER)]).await;

    assert_eq!(
        database.writer.options,
        vec![("statement_timeout".to_owned(), "30000".to_owned())]
    );
}

#[tokio::test]
async fn a_malformed_configured_timeout_is_refused_by_name() {
    let refused = try_resolve(
        "general_settings:\n  database_lock_timeout: soon",
        &[("DATABASE_URL", WRITER)],
    )
    .await
    .unwrap_err();

    assert!(
        matches!(&refused, Error::Invalid { setting, expected: Expected::Seconds } if setting == "general_settings.database_lock_timeout"),
        "{refused:?}"
    );
}

#[rstest]
#[case::disabled("?sslmode=disable", SslMode::Disable, None)]
#[case::required("?sslmode=require", SslMode::Require, None)]
#[case::libpq_verify_ca("?sslmode=verify-ca", SslMode::VerifyFull, None)]
#[case::libpq_root_cert(
    "?sslmode=verify-full&sslrootcert=/ca.pem",
    SslMode::VerifyFull,
    Some("/ca.pem")
)]
#[case::prisma_strict(
    "?sslmode=require&sslaccept=strict&sslcert=/ca.pem",
    SslMode::VerifyFull,
    Some("/ca.pem")
)]
#[case::prisma_cert_without_strict("?sslmode=require&sslcert=/ca.pem", SslMode::Require, None)]
#[tokio::test]
async fn tls_settings_are_read_in_both_dialects(
    #[case] query: &str,
    #[case] ssl_mode: SslMode,
    #[case] root_cert: Option<&str>,
) {
    let database = resolve_url(&format!("{WRITER}{query}")).await;

    assert_eq!(database.writer.ssl_mode, ssl_mode);
    assert_eq!(database.writer.ssl_root_cert, root_cert.map(PathBuf::from));
}

#[rstest]
#[case::server_refuses(b'N', SslMode::Disable)]
#[case::server_offers(b'S', SslMode::VerifyFull)]
#[tokio::test]
async fn strict_tls_under_prefer_asks_the_server_once(
    #[case] answer: u8,
    #[case] ssl_mode: SslMode,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0u8; 8];
        stream.read_exact(&mut request).await.unwrap();
        stream.write_all(&[answer]).await.unwrap();
        request
    });

    let database = resolve_url(&format!(
        "postgresql://app:secret@127.0.0.1:{port}/proxy?sslaccept=strict"
    ))
    .await;

    assert_eq!(database.writer.ssl_mode, ssl_mode);
    assert_eq!(
        server.await.unwrap().as_slice(),
        [8_u32.to_be_bytes(), 80_877_103_u32.to_be_bytes()].concat()
    );
}

#[rstest]
#[case::url_names_pgbouncer("?pgbouncer=true", &[], 0)]
#[case::variable_disables_prepared_statements("", &[("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")], 0)]
#[case::url_sizes_the_cache("?statement_cache_size=25", &[], 25)]
#[case::default("", &[], 100)]
#[tokio::test]
async fn the_statement_cache_follows_the_url_and_the_environment(
    #[case] query: &str,
    #[case] variables: &[(&str, &str)],
    #[case] capacity: usize,
) {
    let url = format!("{WRITER}{query}");
    let variables: Vec<(&str, &str)> = variables
        .iter()
        .copied()
        .chain([("DATABASE_URL", url.as_str())])
        .collect();

    let database = resolve("model_list: []", &variables).await;

    assert_eq!(database.writer.statement_cache_capacity, capacity);
}

#[tokio::test]
async fn the_config_can_disable_prepared_statements() {
    let yaml = "general_settings:\n  database_disable_prepared_statements: true";

    let database = resolve(yaml, &[("DATABASE_URL", WRITER)]).await;

    assert_eq!(database.writer.statement_cache_capacity, 0);
}

#[rstest]
#[case::operator_named("?application_name=billing", "billing")]
#[case::prisma_default("?application_name=litellm-prisma", "litellm-native")]
#[tokio::test]
async fn only_an_operator_application_name_is_kept(#[case] query: &str, #[case] expected: &str) {
    let database = resolve_url(&format!("{WRITER}{query}")).await;

    assert_eq!(database.writer.application_name.as_deref(), Some(expected));
}

#[tokio::test]
async fn settings_the_pool_cannot_carry_are_reported() {
    let database = resolve_url(&format!(
        "{WRITER}?sslidentity=/client.p12&options=-c%20lock_timeout%3D10%20-x"
    ))
    .await;

    assert_eq!(
        database.writer.options,
        vec![("lock_timeout".to_owned(), "10".to_owned())]
    );
    assert_eq!(database.writer.ssl_client, None);
    assert_eq!(
        database
            .ignored
            .iter()
            .map(|ignored| ignored.setting.as_str())
            .collect::<Vec<_>>(),
        ["DATABASE_URL options -x", "DATABASE_URL ?sslidentity"]
    );
}

#[tokio::test]
async fn nothing_configured_is_no_database() {
    assert!(matches!(try_resolve("model_list: []", &[]).await, Ok(None)));
}

#[rstest]
#[case::iam("IAM_TOKEN_DB_AUTH")]
#[case::entra("AZURE_POSTGRESQL_AUTH")]
#[tokio::test]
async fn token_auth_is_refused_by_name(#[case] flag: &'static str) {
    let refused = try_resolve(
        "model_list: []",
        &[("DATABASE_URL", WRITER), (flag, "True")],
    )
    .await;

    assert!(matches!(refused, Err(Error::TokenAuth { variable }) if variable == flag));
}

#[tokio::test]
async fn an_unsupported_scheme_is_refused_without_echoing_the_url() {
    let refused = try_resolve(
        "model_list: []",
        &[("DATABASE_URL", "mysql://root:hunter2@db.internal/proxy")],
    )
    .await
    .unwrap_err();

    assert!(matches!(&refused, Error::Scheme { scheme, .. } if scheme == "mysql"));
    assert!(!refused.to_string().contains("hunter2"));
}

#[tokio::test]
async fn a_replica_limit_that_is_not_a_positive_integer_is_refused() {
    let refused = try_resolve(
        "model_list: []",
        &[
            ("DATABASE_URL", WRITER),
            (
                "DATABASE_URL_READ_REPLICA",
                "postgresql://app@replica.internal/proxy?connection_limit=0",
            ),
        ],
    )
    .await
    .unwrap_err();

    assert!(
        matches!(&refused, Error::Invalid { setting, expected: Expected::PositiveInteger } if setting == "DATABASE_URL_READ_REPLICA ?connection_limit"),
        "{refused:?}"
    );
}

fn budgeted(max_connections: u32) -> Overlay {
    Overlay {
        limits: Limits {
            max_connections: Rule::OverUrl(max_connections),
            acquire_timeout: Rule::UnderUrl(Some(10.0)),
            idle_timeout: Rule::UnderUrl(None),
            socket_timeout: Rule::UnderUrl(None),
        },
        unnamed_statements: false,
        session_timeouts: Vec::new(),
        client_identity: None,
    }
}

fn budgeted_endpoint() -> Endpoint {
    Endpoint {
        host: "db".to_owned(),
        port: 5432,
        socket: None,
        username: "u".to_owned(),
        password: None,
        database: "proxy".to_owned(),
        ssl_mode: SslMode::Prefer,
        ssl_root_cert: None,
        ssl_client: None,
        application_name: Some("litellm-native".to_owned()),
        options: Vec::new(),
        statement_cache_capacity: 100,
        pool: PoolLimits {
            max_connections: 2,
            acquire_timeout_seconds: Some(10.0),
            idle_timeout_seconds: None,
            socket_timeout_seconds: None,
        },
    }
}

#[rstest]
#[case::the_hosts_share_replaces_the_urls_limit_and_the_url_keeps_its_timeouts(
    "postgresql://litellm:p%40ss%3Aword@db.internal:6543/proxy?connection_limit=12&pool_timeout=20&max_idle_connection_lifetime=60&socket_timeout=7&application_name=litellm-prisma",
    Endpoint {
        host: "db.internal".to_owned(),
        port: 6543,
        username: "litellm".to_owned(),
        password: Some(SecretValue::new("p@ss:word")),
        pool: PoolLimits {
            max_connections: 2,
            acquire_timeout_seconds: Some(20.0),
            idle_timeout_seconds: Some(60.0),
            socket_timeout_seconds: Some(7.0),
        },
        ..budgeted_endpoint()
    }
)]
#[case::strict_tls_schema_timeouts_and_pgbouncer(
    "postgresql://svc%40corp@db/proxy?sslmode=require&sslcert=/certs/root.pem&sslaccept=strict&schema=Tenant&options=-c%20statement_timeout%3D30000%20-clock_timeout%3D5000&pgbouncer=true",
    Endpoint {
        username: "svc@corp".to_owned(),
        ssl_mode: SslMode::VerifyFull,
        ssl_root_cert: Some(PathBuf::from("/certs/root.pem")),
        options: vec![
            ("search_path".to_owned(), "\"Tenant\"".to_owned()),
            ("statement_timeout".to_owned(), "30000".to_owned()),
            ("lock_timeout".to_owned(), "5000".to_owned()),
        ],
        statement_cache_capacity: 0,
        ..budgeted_endpoint()
    }
)]
#[case::require_without_strict_never_verifies_the_operator_cert(
    "postgresql://u@db/proxy?sslmode=require&sslcert=/certs/root.pem&schema=public",
    Endpoint { ssl_mode: SslMode::Require, ..budgeted_endpoint() }
)]
#[case::disable_wins_over_strict(
    "postgresql://u@db/proxy?sslmode=disable&sslaccept=strict",
    Endpoint { ssl_mode: SslMode::Disable, ..budgeted_endpoint() }
)]
#[case::unix_socket_and_pool_timeout_zero_waits_forever(
    "postgresql://u@localhost/proxy?host=/cloudsql/project:region:db&pool_timeout=0",
    Endpoint {
        host: "localhost".to_owned(),
        socket: Some(PathBuf::from("/cloudsql/project:region:db")),
        pool: PoolLimits { acquire_timeout_seconds: None, ..budgeted_endpoint().pool },
        ..budgeted_endpoint()
    }
)]
#[case::a_socket_url_may_leave_the_host_out(
    "postgresql://u@/proxy?host=/run/postgresql",
    Endpoint {
        host: "localhost".to_owned(),
        socket: Some(PathBuf::from("/run/postgresql")),
        ..budgeted_endpoint()
    }
)]
#[tokio::test]
async fn a_final_url_is_read_under_what_the_host_sets(
    #[case] url: &str,
    #[case] expected: Endpoint,
) {
    let (endpoint, ignored) = interpret(WRITER_URL, url, &budgeted(2)).await.unwrap();

    assert_eq!(endpoint, expected);
    assert_eq!(ignored, Vec::new());
}

#[rstest]
#[case::flags(
    "-c statement_timeout=5 -B 64",
    vec![("statement_timeout", "5")],
    vec!["DATABASE_URL options -B", "DATABASE_URL options 64"]
)]
#[case::bare_setting("statement_timeout=5", vec![], vec!["DATABASE_URL options statement_timeout=5"])]
#[case::unbalanced_quote("-c 'unterminated", vec![], vec!["DATABASE_URL options -c 'unterminated"])]
#[tokio::test]
async fn options_the_pool_cannot_send_are_reported_instead_of_failing(
    #[case] options: &str,
    #[case] sent: Vec<(&str, &str)>,
    #[case] unsent: Vec<&str>,
) {
    let url = format!(
        "postgresql://u@db/proxy?options={}",
        options.replace(' ', "%20")
    );

    let (endpoint, ignored) = interpret(WRITER_URL, &url, &budgeted(2)).await.unwrap();

    assert_eq!(
        endpoint.options,
        sent.into_iter()
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        ignored
            .iter()
            .map(|ignored| ignored.setting.as_str())
            .collect::<Vec<_>>(),
        unsent
    );
}

#[rstest]
#[case::no_user("postgresql://db.internal/proxy")]
#[case::no_host_and_no_socket("postgresql://u@/proxy")]
#[tokio::test]
async fn a_url_without_a_host_or_a_user_is_refused_by_name(#[case] url: &str) {
    let refused = interpret(WRITER_URL, url, &budgeted(2)).await.unwrap_err();

    assert_eq!(
        refused.to_string(),
        "DATABASE_URL must name a host and a user"
    );
}

#[rstest]
#[case::sent_over_tls("?sslmode=require&sslidentity=/client.p12", true)]
#[case::dropped_without_tls("?sslmode=disable&sslidentity=/client.p12", false)]
#[tokio::test]
async fn a_hosts_client_identity_is_sent_only_over_tls(#[case] query: &str, #[case] sent: bool) {
    let identity = ClientIdentity {
        cert_pem: "cert".to_owned(),
        key_pem: SecretValue::new("key"),
    };
    let overlay = Overlay {
        client_identity: Some(identity.clone()),
        ..budgeted(2)
    };

    let (endpoint, ignored) = interpret(
        WRITER_URL,
        &format!("postgresql://u@db/proxy{query}"),
        &overlay,
    )
    .await
    .unwrap();

    assert_eq!(endpoint.ssl_client, sent.then_some(identity));
    assert_eq!(ignored, Vec::new());
}
