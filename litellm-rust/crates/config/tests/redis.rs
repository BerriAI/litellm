use std::{
    collections::HashMap,
    fmt::Write,
    sync::{Arc, Mutex},
};

use litellm_config::{
    AwsIamAuth, AzureAdAuth, CacheSettings, Config, Expected, RedisAuth, RedisError, RedisSettings,
    ResolvedRedis, SecretSnapshot,
};
use litellm_redis_types::{ClusterNode, Endpoint, TlsOptions};
use rstest::rstest;
use tracing::{Event, Subscriber, field::Visit};
use tracing_subscriber::{Layer, layer::Context, prelude::*};

fn resolve_settings(
    settings: &RedisSettings,
    environment: &HashMap<String, String>,
) -> Result<ResolvedRedis, RedisError> {
    let lookup = |name: &str| environment.get(name).cloned();
    settings.resolve(
        &SecretSnapshot::read(settings.secret_names(), &lookup),
        &lookup,
    )
}

fn resolve(
    yaml: &str,
    environment: &[(&str, &str)],
) -> Result<litellm_config::ResolvedRedis, RedisError> {
    let environment: HashMap<String, String> = environment
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    let config = Config::from_yaml(yaml).unwrap();
    let settings = config
        .general_settings
        .coordination_redis
        .unwrap_or_default();
    resolve_settings(&settings, &environment)
}

fn resolve_cache(yaml: &str) -> Result<ResolvedRedis, RedisError> {
    resolve_cache_in(yaml, &[])
}

fn resolve_cache_in(yaml: &str, environment: &[(&str, &str)]) -> Result<ResolvedRedis, RedisError> {
    let lookup = |name: &str| {
        environment
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_owned())
    };
    let config = Config::from_yaml(yaml).unwrap();
    match config.litellm_settings.cache_params.unwrap() {
        CacheSettings::Redis(settings) => settings.resolve_cache(
            &SecretSnapshot::read(settings.secret_names(), &lookup),
            &lookup,
        ),
        CacheSettings::Other(_) => panic!("expected a Redis cache"),
    }
}

fn cluster_ports(resolved: &ResolvedRedis) -> Vec<u16> {
    match &resolved.endpoint {
        Endpoint::Cluster(cluster) => cluster.nodes.iter().map(|node| node.port).collect(),
        endpoint => panic!("expected a cluster, got {endpoint:?}"),
    }
}

#[rstest]
#[case::host(
    "general_settings:\n  coordination_redis:\n    host: redis\n    port: 6380\n    db: 2\n    ssl: true\n    namespace: app\n    socket_timeout: 1.5",
    &[],
    "host:redis:6380:2:true:app:1500"
)]
#[case::environment_references_resolve_the_host_and_port(
    "general_settings:\n  coordination_redis:\n    host: os.environ/COORD_HOST\n    port: os.environ/COORD_PORT",
    &[("COORD_HOST", "referenced"), ("COORD_PORT", "6400")],
    "host:referenced:6400:0:false::none"
)]
#[case::a_null_host_falls_back_to_the_environment(
    "general_settings:\n  coordination_redis:\n    host: null",
    &[("REDIS_HOST", "environment")],
    "host:environment:6379:0:false::none"
)]
fn resolves_redis_through_the_parsed_config(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: &str,
) {
    let resolved = resolve(yaml, environment).unwrap();
    let description = match resolved.endpoint {
        Endpoint::Url { url, .. } => format!("url:{}", url.expose()),
        Endpoint::Sentinel(sentinel) => format!("sentinel:{}", sentinel.service_name),
        Endpoint::Host(host) => format!(
            "host:{}:{}:{}:{}:{}:{}",
            host.host,
            host.port,
            host.db,
            host.tls.is_some(),
            resolved.namespace.unwrap_or_default(),
            resolved
                .timeout
                .map(|timeout| timeout.as_millis().to_string())
                .unwrap_or_else(|| "none".to_owned())
        ),
        Endpoint::Cluster(cluster) => format!(
            "cluster:{}",
            cluster
                .nodes
                .iter()
                .map(|node| format!("{}:{}", node.host, node.port))
                .collect::<Vec<_>>()
                .join(",")
        ),
    };

    assert_eq!(description, expected);
}

#[rstest]
#[case::tls_ciphers(
    "general_settings:\n  coordination_redis:\n    host: redis\n    ssl_ciphers: HIGH",
    &[],
    "ssl_ciphers"
)]
#[case::ssl_ciphers_from_environment(
    "model_list: []",
    &[("REDIS_HOST", "redis"), ("REDIS_SSL_CIPHERS", "HIGH")],
    "REDIS_SSL_CIPHERS"
)]
#[case::connection_pool(
    "general_settings:\n  coordination_redis:\n    host: redis\n    connection_pool: shared",
    &[],
    "connection_pool"
)]
fn rejected_settings_name_their_source(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: &str,
) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(matches!(error, RedisError::Unsupported { setting } if setting == expected));
}

fn static_auth(username: Option<&str>, password: Option<&str>) -> RedisAuth {
    RedisAuth::Static {
        username: username.map(litellm_secret_value::SecretValue::new),
        password: password.map(litellm_secret_value::SecretValue::new),
    }
}

#[rstest]
fn an_explicit_host_keeps_environment_credentials() {
    let resolved = resolve(
        "general_settings:\n  coordination_redis:\n    host: configured",
        &[
            ("REDIS_PASSWORD", "environment-secret"),
            ("REDIS_SSL", "True"),
        ],
    )
    .unwrap();

    assert_eq!(resolved.auth, static_auth(None, Some("environment-secret")));
    assert!(matches!(resolved.endpoint, Endpoint::Host(host) if host.tls.is_some()));
}

fn azure(username: Option<&str>) -> RedisAuth {
    RedisAuth::AzureAd(AzureAdAuth {
        client_id: Some("client".to_owned()),
        tenant_id: None,
        client_secret: None,
        username: username.map(str::to_owned),
    })
}

fn aws(serverless: bool) -> RedisAuth {
    RedisAuth::AwsIam(AwsIamAuth {
        user_name: "iam-user".to_owned(),
        cache_name: "cache-1".to_owned(),
        region: "us-east-1".to_owned(),
        serverless,
    })
}

#[rstest]
#[case::static_credentials(
    "general_settings:\n  coordination_redis:\n    host: redis\n    username: user\n    password: pass",
    &[],
    static_auth(Some("user"), Some("pass"))
)]
#[case::an_explicit_provider_beats_every_cloud_auth(
    "general_settings:\n  coordination_redis:\n    host: redis\n    password: pass\n    credential_provider: host-object\n    gcp_service_account: account\n    aws_iam_auth: true",
    &[("REDIS_AZURE_AD_TOKEN", "true")],
    RedisAuth::Host
)]
#[case::gcp_beats_azure_and_aws(
    "general_settings:\n  coordination_redis:\n    host: redis\n    password: pass\n    azure_redis_ad_token: \"True\"\n    aws_iam_auth: true",
    &[("REDIS_GCP_SERVICE_ACCOUNT", "projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com")],
    RedisAuth::GcpIam { service_account: "projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com".to_owned() }
)]
#[case::azure_beats_aws_and_takes_the_environment_username(
    "general_settings:\n  coordination_redis:\n    host: redis\n    username: configured-user\n    azure_redis_ad_token: true\n    azure_client_id: client\n    aws_iam_auth: true",
    &[("REDIS_USERNAME", "environment-user")],
    azure(Some("environment-user"))
)]
#[case::azure_without_an_environment_username_sends_only_the_token(
    "general_settings:\n  coordination_redis:\n    host: redis\n    username: configured-user\n    azure_redis_ad_token: true\n    azure_client_id: client",
    &[],
    azure(None)
)]
#[case::azure_needs_the_literal_true(
    "general_settings:\n  coordination_redis:\n    host: redis\n    azure_redis_ad_token: \"1\"",
    &[],
    static_auth(None, None)
)]
#[case::aws_with_a_region_from_the_environment(
    "general_settings:\n  coordination_redis:\n    host: redis\n    ssl: true\n    password: dropped\n    aws_iam_auth: \"yes\"\n    aws_iam_user_name: iam-user\n    aws_iam_cache_name: cache-1",
    &[("AWS_DEFAULT_REGION", "us-east-1")],
    aws(false)
)]
#[case::aws_serverless_over_a_rediss_url(
    "general_settings:\n  coordination_redis:\n    url: rediss://cache-1:6379\n    aws_iam_auth: true\n    aws_iam_user_name: iam-user\n    aws_iam_cache_name: cache-1\n    aws_iam_region: us-east-1\n    aws_iam_serverless: true",
    &[],
    aws(true)
)]
fn auth_follows_python_precedence(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: RedisAuth,
) {
    assert_eq!(resolve(yaml, environment).unwrap().auth, expected);
}

#[rstest]
#[case::without_tls(
    "general_settings:\n  coordination_redis:\n    host: redis\n    aws_iam_auth: true\n    aws_iam_user_name: u\n    aws_iam_cache_name: c\n    aws_iam_region: r",
    "requires TLS"
)]
#[case::a_plain_url_ignores_ssl(
    "general_settings:\n  coordination_redis:\n    url: redis://cache:6379\n    ssl: true\n    aws_iam_auth: true\n    aws_iam_user_name: u\n    aws_iam_cache_name: c\n    aws_iam_region: r",
    "requires TLS"
)]
#[case::missing_settings(
    "general_settings:\n  coordination_redis:\n    host: redis\n    ssl: true\n    aws_iam_auth: true\n    aws_iam_cache_name: c",
    "requires: aws_iam_user_name, aws_iam_region"
)]
fn incomplete_aws_iam_auth_fails(#[case] yaml: &str, #[case] message: &str) {
    let error = resolve(yaml, &[]).unwrap_err();

    assert!(error.to_string().contains(message), "{error}");
}

#[rstest]
#[case::defaults_verify_the_certificate_but_not_the_hostname(
    "    ssl: true",
    &[],
    Some(TlsOptions { ca_certs: None, certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::every_option(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    ssl_certfile: /client.pem\n    ssl_keyfile: /client.key\n    ssl_cert_reqs: none\n    ssl_check_hostname: true",
    &[],
    Some(TlsOptions { ca_certs: Some("/ca.pem".into()), certfile: Some("/client.pem".into()), keyfile: Some("/client.key".into()), verify_certificate: false, check_hostname: true })
)]
#[case::options_from_the_environment(
    "    ssl: true",
    &[("REDIS_SSL_CA_CERTS", "/env-ca.pem"), ("REDIS_SSL_CERT_REQS", "optional")],
    Some(TlsOptions { ca_certs: Some("/env-ca.pem".into()), certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::gcp_ca_certs_replace_the_configured_ones(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    gcp_service_account: account\n    gcp_ssl_ca_certs: /gcp-ca.pem",
    &[],
    Some(TlsOptions { ca_certs: Some("/gcp-ca.pem".into()), certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::options_without_ssl_stay_plaintext("    ssl_ca_certs: /ca.pem", &[], None)]
fn tls_options_resolve_like_redis_py(
    #[case] block: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: Option<TlsOptions>,
) {
    let yaml = format!("general_settings:\n  coordination_redis:\n    host: redis\n{block}");

    let resolved = resolve(&yaml, environment).unwrap();

    assert!(matches!(resolved.endpoint, Endpoint::Host(host) if host.tls == expected));
}

#[rstest]
fn an_unknown_cert_requirement_fails() {
    let error = resolve(
        "general_settings:\n  coordination_redis:\n    host: redis\n    ssl: true\n    ssl_cert_reqs: strict",
        &[],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        RedisError::Invalid { setting, expected: Expected::CertRequirement } if setting == "ssl_cert_reqs"
    ));
}

#[rstest]
#[case::from_config(
    "general_settings:\n  coordination_redis:\n    sentinel_nodes: [[sentinel-a, 26379], [sentinel-b, \"26380\"]]\n    service_name: primary\n    sentinel_password: sentinel-secret\n    db: 2",
    &[]
)]
#[case::from_the_environment(
    "model_list: []",
    &[
        ("REDIS_SENTINEL_NODES", r#"[["sentinel-a", 26379], ["sentinel-b", "26380"]]"#),
        ("REDIS_SERVICE_NAME", "primary"),
        ("REDIS_SENTINEL_PASSWORD", "sentinel-secret"),
        ("REDIS_DB", "2"),
    ]
)]
fn sentinel_nodes_resolve_to_a_sentinel_endpoint(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
) {
    let Endpoint::Sentinel(sentinel) = resolve(yaml, environment).unwrap().endpoint else {
        panic!("expected a sentinel endpoint");
    };

    assert_eq!(
        sentinel.sentinels,
        vec![
            ClusterNode {
                host: "sentinel-a".to_owned(),
                port: 26379
            },
            ClusterNode {
                host: "sentinel-b".to_owned(),
                port: 26380
            },
        ]
    );
    assert_eq!(sentinel.service_name, "primary");
    assert_eq!(
        sentinel
            .sentinel_password
            .as_ref()
            .map(|password| password.expose()),
        Some("sentinel-secret")
    );
    assert_eq!(sentinel.db, 2);
}

#[rstest]
#[case::a_url_beats_sentinels(
    "general_settings:\n  coordination_redis:\n    url: redis://single:6379\n    sentinel_nodes: [[s, 26379]]\n    service_name: primary",
    "url:redis://single:6379"
)]
fn sentinel_precedence(#[case] yaml: &str, #[case] expected: &str) {
    let resolved = resolve(yaml, &[]).unwrap();
    let description = match resolved.endpoint {
        Endpoint::Url { url, .. } => format!("url:{}", url.expose()),
        Endpoint::Sentinel(sentinel) => format!("sentinel:{}", sentinel.service_name),
        other => panic!("unexpected {other:?}"),
    };

    assert_eq!(description, expected);
}

#[rstest]
fn sentinels_without_a_service_name_fail() {
    let error = resolve(
        "general_settings:\n  coordination_redis:\n    sentinel_nodes: [[s, 26379]]",
        &[],
    )
    .unwrap_err();

    assert!(matches!(error, RedisError::NoSentinelService), "{error:?}");
}

#[rstest]
#[case::configured_host(
    "general_settings:\n  coordination_redis:\n    host: ignored\n    sentinel_nodes: [[s, 26379]]\n    service_name: primary",
    &[],
    "host"
)]
#[case::environment_port(
    "general_settings:\n  coordination_redis:\n    sentinel_nodes: [[s, 26379]]\n    service_name: primary",
    &[("REDIS_PORT", "6380")],
    "REDIS_PORT"
)]
fn a_host_or_port_cannot_reach_sentinel(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] source: &str,
) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(
        matches!(&error, RedisError::SentinelWithHost { setting } if setting == source),
        "{error:?}"
    );
}

#[rstest]
#[case::empty("general_settings:\n  coordination_redis:\n    sentinel_nodes: []\n    service_name: primary", &[], "sentinel_nodes")]
#[case::not_pairs(
    "general_settings:\n  coordination_redis:\n    sentinel_nodes: [{host: s, port: 26379}]\n    service_name: primary",
    &[],
    "sentinel_nodes"
)]
#[case::from_the_environment(
    "model_list: []",
    &[("REDIS_SENTINEL_NODES", "not-json"), ("REDIS_SERVICE_NAME", "primary")],
    "REDIS_SENTINEL_NODES"
)]
fn invalid_sentinel_nodes_name_their_source(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] source: &str,
) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(
        matches!(&error, RedisError::Invalid { setting, expected: Expected::SentinelNodes } if setting == source),
        "{error:?}"
    );
}

#[rstest]
#[case::from_config("general_settings:\n  coordination_redis:\n    startup_nodes: [{host: a}]", &[], "startup_nodes")]
#[case::from_the_environment("model_list: []", &[("REDIS_CLUSTER_NODES", "not-json")], "REDIS_CLUSTER_NODES")]
#[case::out_of_range_port_from_the_environment("model_list: []", &[("REDIS_HOST", "redis"), ("REDIS_PORT", "70000")], "REDIS_PORT")]
fn invalid_values_name_their_source(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] source: &str,
) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(
        matches!(&error, RedisError::Invalid { setting, .. } if setting == source),
        "{error:?}"
    );
}

#[rstest]
#[case::falls_back_to_the_socket_timeout("    socket_timeout: 1.5", Some(1500), None)]
#[case::explicit_connect_timeout(
    "    socket_timeout: 1.5\n    socket_connect_timeout: 0.25",
    Some(1500),
    Some(250)
)]
fn connect_timeout_is_resolved_separately(
    #[case] block: &str,
    #[case] timeout_ms: Option<u128>,
    #[case] connect_ms: Option<u128>,
) {
    let resolved = resolve(
        &format!("general_settings:\n  coordination_redis:\n    host: redis\n{block}"),
        &[],
    )
    .unwrap();

    assert_eq!(
        resolved.timeout.map(|timeout| timeout.as_millis()),
        timeout_ms
    );
    assert_eq!(
        resolved.connect_timeout.map(|timeout| timeout.as_millis()),
        connect_ms
    );
}

#[rstest]
#[case::only_a_port("general_settings:\n  coordination_redis:\n    port: 6379", &[])]
#[case::a_structured_host_with_an_environment_target(
    "general_settings:\n  coordination_redis:\n    host: []",
    &[("REDIS_HOST", "environment"), ("REDIS_URL", "redis://environment:6379")]
)]
fn a_block_without_a_usable_target_fails(#[case] yaml: &str, #[case] environment: &[(&str, &str)]) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(matches!(error, RedisError::NoTarget), "{error:?}");
}

#[rstest]
fn the_cache_alias_names_cluster_nodes() {
    let resolved = resolve_cache(
        "litellm_settings:\n  cache_params:\n    type: redis\n    redis_startup_nodes:\n      - host: node-a\n        port: 7000",
    )
    .unwrap();

    assert_eq!(cluster_ports(&resolved), vec![7000]);
}

#[rstest]
fn canonical_startup_nodes_beat_the_cache_alias() {
    let resolved = resolve_cache(
        "litellm_settings:\n  cache_params:\n    type: redis\n    startup_nodes:\n      - host: canonical\n        port: 7000\n    redis_startup_nodes:\n      - host: alias\n        port: 7100\n      - host: alias\n        port: 7101",
    )
    .unwrap();

    assert_eq!(cluster_ports(&resolved), vec![7000]);
}

#[rstest]
fn a_null_canonical_startup_nodes_setting_shadows_the_cache_alias() {
    let resolved = resolve_cache(
        "litellm_settings:\n  cache_params:\n    type: redis\n    host: cache\n    startup_nodes: null\n    redis_startup_nodes:\n      - host: alias\n        port: 7100",
    )
    .unwrap();

    assert!(matches!(resolved.endpoint, Endpoint::Host(_)));
}

#[rstest]
fn a_missing_canonical_startup_nodes_reference_shadows_the_cache_alias() {
    let resolved = resolve_cache(
        "litellm_settings:\n  cache_params:\n    type: redis\n    host: cache\n    startup_nodes: os.environ/MISSING_NODES\n    redis_startup_nodes:\n      - host: alias\n        port: 7100",
    )
    .unwrap();

    assert!(matches!(resolved.endpoint, Endpoint::Host(_)));
}

#[rstest]
fn rejected_environment_settings_never_print_the_value() {
    let error = resolve(
        "model_list: []",
        &[
            ("REDIS_HOST", "redis"),
            ("REDIS_SSL_CIPHERS", "private-option-value"),
        ],
    )
    .unwrap_err();
    let message = error.to_string();

    assert!(message.contains("REDIS_SSL_CIPHERS"));
    assert!(!message.contains("private-option-value"));
}

#[rstest]
#[case::negative("-1")]
#[case::infinite("inf")]
fn rejects_bad_socket_timeout(#[case] timeout: &str) {
    let yaml = format!(
        "general_settings:\n  coordination_redis:\n    host: redis\n    socket_timeout: {timeout}"
    );
    let error = resolve(&yaml, &[]).unwrap_err();

    assert!(matches!(
        error,
        RedisError::Invalid { setting, expected: Expected::Seconds } if setting == "socket_timeout"
    ));
}

#[rstest]
fn debug_redacts_redis_secrets_and_url() {
    let config = Config::from_yaml(
        "general_settings:\n  coordination_redis:\n    url: redis://user:password@redis:6379\n    username: credential-user\n    password: credential-password",
    )
    .unwrap();
    let settings = config.general_settings.coordination_redis.unwrap();
    let resolved = resolve_settings(&settings, &HashMap::new()).unwrap();
    let debug = format!("{settings:?} {resolved:?}");

    assert!(!debug.contains("redis://user:password"));
    assert!(!debug.contains("credential-user"));
    assert!(!debug.contains("credential-password"));
    assert!(debug.contains("redacted") || debug.contains("REDACTED"));
}

#[rstest]
fn inactive_redis_cache_keeps_malformed_known_inputs_deferred() {
    let config = Config::from_yaml(
        "litellm_settings:\n  cache: false\n  cache_params:\n    type: redis\n    host: []\n    port: invalid\n    socket_timeout: invalid",
    )
    .unwrap();

    assert!(matches!(
        config.litellm_settings.cache_params,
        Some(litellm_config::CacheSettings::Redis(_))
    ));
}

#[rstest]
fn non_redis_cache_settings_remain_opaque() {
    let config = Config::from_yaml(
        "litellm_settings:\n  cache_params:\n    type: local\n    custom_option:\n      nested: true",
    )
    .unwrap();

    assert!(matches!(
        config.litellm_settings.cache_params,
        Some(litellm_config::CacheSettings::Other(_))
    ));
}

#[rstest]
fn legacy_unknown_settings_do_not_change_resolution() {
    let resolved = resolve(
        "general_settings:\n  coordination_redis:\n    host: redis\n    redis_flush_size: 42\n    service_name: primary",
        &[],
    )
    .unwrap();

    assert!(matches!(resolved.endpoint, Endpoint::Host(_)));
}

struct WarningCapture(Arc<Mutex<Vec<String>>>);

impl<S: Subscriber> Layer<S> for WarningCapture {
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

#[derive(Default)]
struct Fields(String);

impl Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        write!(self.0, "{}={value:?} ", field.name()).unwrap();
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        write!(self.0, "{}={value} ", field.name()).unwrap();
    }
}

#[rstest]
fn ignored_setting_warning_names_the_setting_without_its_value() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(WarningCapture(events.clone()));
    tracing::subscriber::with_default(subscriber, || {
        resolve(
            "general_settings:\n  coordination_redis:\n    host: redis\n    max_connections: 47",
            &[],
        )
        .unwrap();
    });
    let warnings = events.lock().unwrap().join(" ");

    assert!(warnings.contains("max_connections"));
    assert!(!warnings.contains("47"));
}

#[derive(serde::Deserialize)]
struct Contract {
    cases: Vec<ContractCase>,
}

#[derive(serde::Deserialize)]
struct ContractCase {
    name: String,
    kwargs: serde_json::Value,
    environment: HashMap<String, String>,
    python: serde_json::Value,
}

const CONTRACT: &str = include_str!("redis_resolution.json");

// Settings redis-py accepts and Rust rejects at boot on purpose, each listed in crates/config/AGENTS.md
const DIVERGENCES: &[(&str, &str)] = &[
    ("ssl_ciphers", "ssl_ciphers"),
    ("unix_socket_path", "unix_socket_path"),
    (
        "a_url_query_setting_without_a_redis_rs_equivalent",
        "url ?socket_timeout",
    ),
];

fn tls_view(tls: Option<&TlsOptions>) -> serde_json::Value {
    tls.map_or(serde_json::Value::Null, |tls| {
        serde_json::json!({
            "ca_certs": tls.ca_certs,
            "certfile": tls.certfile,
            "keyfile": tls.keyfile,
            "verify_certificate": tls.verify_certificate,
            "check_hostname": tls.check_hostname,
        })
    })
}

fn nodes_view(nodes: &[ClusterNode]) -> Vec<String> {
    nodes
        .iter()
        .map(|node| format!("{}:{}", node.host, node.port))
        .collect()
}

fn expose(value: Option<&litellm_secret_value::SecretValue>) -> Option<&str> {
    value.map(litellm_secret_value::SecretValue::expose)
}

fn view(resolved: &ResolvedRedis) -> serde_json::Value {
    let target = match &resolved.endpoint {
        Endpoint::Host(host) => serde_json::json!({
            "mode": "host",
            "host": host.host,
            "port": host.port,
            "db": host.db,
            "tls": tls_view(host.tls.as_ref()),
        }),
        Endpoint::Url { url, tls } => serde_json::json!({
            "mode": "url",
            "url": url.expose(),
            "tls": tls_view(url.expose().starts_with("rediss://").then_some(tls)),
        }),
        Endpoint::Cluster(cluster) => serde_json::json!({
            "mode": "cluster",
            "nodes": nodes_view(&cluster.nodes),
            "tls": tls_view(cluster.tls.as_ref()),
        }),
        Endpoint::Sentinel(sentinel) => serde_json::json!({
            "mode": "sentinel",
            "nodes": nodes_view(&sentinel.sentinels),
            "service_name": sentinel.service_name,
            "sentinel_password": expose(sentinel.sentinel_password.as_ref()),
            "db": sentinel.db,
            "tls": tls_view(sentinel.tls.as_ref()),
        }),
    };
    let auth = match &resolved.auth {
        RedisAuth::Static { username, password } => serde_json::json!({
            "kind": "static",
            "username": expose(username.as_ref()),
            "password": expose(password.as_ref()),
        }),
        RedisAuth::Host => serde_json::json!({"kind": "host"}),
        RedisAuth::GcpIam { service_account } => {
            serde_json::json!({"kind": "gcp_iam", "service_account": service_account})
        }
        RedisAuth::AzureAd(azure) => {
            serde_json::json!({"kind": "azure_ad", "username": azure.username})
        }
        RedisAuth::AwsIam(aws) => serde_json::json!({
            "kind": "aws_iam",
            "user_name": aws.user_name,
            "cache_name": aws.cache_name,
            "region": aws.region,
            "serverless": aws.serverless,
        }),
    };
    let serde_json::Value::Object(target) = target else {
        unreachable!("every target view is an object")
    };
    let rest = serde_json::json!({
        "auth": auth,
        "socket_timeout": resolved.timeout.map_or(5.0, |timeout| timeout.as_secs_f64()),
        "socket_connect_timeout": resolved.connect_timeout.or(resolved.timeout).map_or(5.0, |timeout| timeout.as_secs_f64()),
    });
    let serde_json::Value::Object(rest) = rest else {
        unreachable!("the view is an object")
    };
    serde_json::Value::Object(target.into_iter().chain(rest).collect())
}

fn rust_outcome(case: &ContractCase) -> Result<serde_json::Value, RedisError> {
    let settings: RedisSettings = serde_json::from_value(case.kwargs.clone()).unwrap();
    resolve_settings(&settings, &case.environment).map(|resolved| view(&resolved))
}

fn matches_python_rejection(error: &RedisError, python: &serde_json::Value) -> bool {
    let rejection = &python["rejected"];
    match error {
        RedisError::Unsupported { setting } if setting == "ssl_ca_path" => {
            rejection["type"] == "TypeError"
                && rejection["message"]
                    == "Redis.__init__() got an unexpected keyword argument 'ssl_ca_path'"
        }
        RedisError::NoTarget => {
            rejection["type"] == "ValueError"
                && rejection["message"] == "Either 'host' or 'url' must be specified for redis."
        }
        RedisError::AwsIamWithoutTls | RedisError::AwsIamIncomplete { .. } => {
            rejection["type"] == "ValueError" && rejection["message"] == error.to_string()
        }
        RedisError::DbInCluster { setting } => {
            setting == "db"
                && rejection["type"] == "RedisClusterException"
                && rejection["message"] == "Argument 'db' must be 0 or None in cluster mode"
        }
        RedisError::SentinelWithHost { setting } => {
            rejection["type"] == "TypeError"
                && rejection["message"].as_str().is_some_and(|message| {
                    message.ends_with(&format!(
                        "got multiple values for keyword argument '{setting}'"
                    ))
                })
        }
        _ => false,
    }
}

fn mismatch(case: &ContractCase, contract: &Contract) -> Option<String> {
    let divergence = DIVERGENCES.iter().find(|(name, _)| *name == case.name);
    let python_rejects = case.python.get("rejected").is_some();
    match (rust_outcome(case), divergence) {
        (
            Err(RedisError::Unsupported { setting } | RedisError::Invalid { setting, .. }),
            Some((_, expected)),
        ) if setting == *expected && !python_rejects => None,
        (outcome, Some(_)) => Some(format!(
            "{}: listed as a divergence, got {outcome:?}",
            case.name
        )),
        (Err(ref error), None) if matches_python_rejection(error, &case.python) => None,
        (Err(RedisError::Unsupported { setting }), None) => Some(format!(
            "{}: Rust rejects {setting}, which Python accepts",
            case.name
        )),
        (Ok(view), None)
            if case.name == "retry_on_error_from_the_environment"
                && case.python["rejected"]["type"] == "TypeError"
                && case.python["rejected"]["message"] == "'bool' object is not iterable"
                && contract
                    .cases
                    .iter()
                    .any(|default| default.name == "host_defaults" && default.python == view) =>
        {
            None
        }
        (Ok(view), None) if view == case.python => None,
        (outcome, None) => Some(format!(
            "{}: rust {outcome:?}, python {}",
            case.name, case.python
        )),
    }
}

// redis_resolution.json comes from generate_redis_resolution.py running litellm/_redis.py, and
// tests/unit/test_redis.py fails when that file drifts from the Python, so both sides pin one contract
#[rstest]
fn resolution_matches_python_redis() {
    let contract: Contract = serde_json::from_str(CONTRACT).unwrap();

    let unknown_divergences: Vec<&str> = DIVERGENCES
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !contract.cases.iter().any(|case| case.name == *name))
        .collect();
    let mismatches: Vec<String> = contract
        .cases
        .iter()
        .filter_map(|case| mismatch(case, &contract))
        .collect();

    assert_eq!(unknown_divergences, Vec::<&str>::new());

    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

fn snapshot(values: &[(&str, &str)]) -> SecretSnapshot {
    let values: HashMap<&str, &str> = values.iter().copied().collect();
    SecretSnapshot::read(RedisSettings::default().secret_names(), &|name: &str| {
        values.get(name).map(|value| (*value).to_owned())
    })
}

#[rstest]
fn secret_manager_values_reach_the_resolver_through_the_snapshot() {
    let settings: RedisSettings =
        serde_json::from_str(r#"{"host": "os.environ/COORD_HOST"}"#).unwrap();
    let secrets = SecretSnapshot::read(settings.secret_names(), &|name: &str| match name {
        "COORD_HOST" => Some("from-the-manager".to_owned()),
        "REDIS_PASSWORD" => Some("manager-secret".to_owned()),
        _ => None,
    });

    let resolved = settings.resolve(&secrets, &|_: &str| None).unwrap();

    assert!(matches!(&resolved.endpoint, Endpoint::Host(host) if host.host == "from-the-manager"));
    assert_eq!(resolved.auth, static_auth(None, Some("manager-secret")));
}

#[rstest]
#[case::from_the_environment(&[], &[("REDIS_USERNAME", "environment-user")], Some("environment-user"))]
#[case::never_from_the_secret_manager(&[("REDIS_USERNAME", "manager-user")], &[], None)]
fn the_azure_username_comes_from_the_process_environment_like_python(
    #[case] secrets: &[(&str, &str)],
    #[case] environment: &[(&str, &str)],
    #[case] expected: Option<&str>,
) {
    let settings: RedisSettings =
        serde_json::from_str(r#"{"host": "redis", "azure_redis_ad_token": "true"}"#).unwrap();
    let environment: HashMap<&str, &str> = environment.iter().copied().collect();

    let resolved = settings
        .resolve(&snapshot(secrets), &|name: &str| {
            environment.get(name).map(|value| (*value).to_owned())
        })
        .unwrap();

    assert!(
        matches!(&resolved.auth, RedisAuth::AzureAd(azure) if azure.username.as_deref() == expected),
        "{:?}",
        resolved.auth
    );
}

#[rstest]
#[case::coordination_drops_the_null(
    "general_settings:\n  coordination_redis:\n    host: redis\n    ssl: null",
    false,
    true
)]
#[case::cache_passes_an_unnamed_null_on(
    "litellm_settings:\n  cache_params:\n    type: redis\n    host: redis\n    ssl: null",
    true,
    false
)]
#[case::cache_drops_a_null_it_names(
    "litellm_settings:\n  cache_params:\n    type: redis\n    host: null\n    ssl: true",
    true,
    true
)]
fn a_null_overrides_the_environment_only_where_python_passes_it_on(
    #[case] yaml: &str,
    #[case] cache: bool,
    #[case] tls: bool,
) {
    let environment = [("REDIS_HOST", "environment"), ("REDIS_SSL", "true")];
    let resolved = match cache {
        true => resolve_cache_in(yaml, &environment),
        false => resolve(yaml, &environment),
    }
    .unwrap();

    let Endpoint::Host(host) = resolved.endpoint else {
        panic!("expected a host endpoint, got {:?}", resolved.endpoint);
    };
    assert_eq!(host.tls.is_some(), tls);
}
