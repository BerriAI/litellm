use std::{
    collections::HashMap,
    fmt::Write,
    sync::{Arc, Mutex},
};

use litellm_config::{
    AwsIamAuth, AzureAdAuth, CacheSettings, Config, Expected, RedisAuth, RedisEndpoint, RedisError, RedisNode,
    RedisSettings, RedisTls, ResolvedRedis,
};
use rstest::rstest;
use tracing::{Event, Subscriber, field::Visit};
use tracing_subscriber::{Layer, layer::Context, prelude::*};

fn resolve(
    yaml: &str,
    environment: &[(&str, &str)],
) -> Result<litellm_config::ResolvedRedis, RedisError> {
    let environment: HashMap<String, String> = environment
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    let config = Config::from_yaml(yaml).unwrap();
    config
        .general_settings
        .coordination_redis
        .as_ref()
        .unwrap_or(&RedisSettings::default())
        .resolve(&|name| environment.get(name).cloned())
}

fn resolve_cache(yaml: &str) -> Result<ResolvedRedis, RedisError> {
    let config = Config::from_yaml(yaml).unwrap();
    match config.litellm_settings.cache_params.unwrap() {
        CacheSettings::Redis(settings) => settings.resolve_cache(&|_| None),
        CacheSettings::Other(_) => panic!("expected a Redis cache"),
    }
}

fn cluster_ports(resolved: &ResolvedRedis) -> Vec<u16> {
    match &resolved.endpoint {
        RedisEndpoint::Cluster(cluster) => cluster.nodes.iter().map(|node| node.port).collect(),
        endpoint => panic!("expected a cluster, got {endpoint:?}"),
    }
}

#[rstest]
#[case::host(
    "general_settings:\n  coordination_redis:\n    host: redis\n    port: 6380\n    db: 2\n    ssl: true\n    namespace: app\n    socket_timeout: 1.5",
    &[],
    "host:redis:6380:2:true:app:1500"
)]
#[case::environment_url(
    "model_list: []",
    &[("REDIS_URL", "redis://environment:6379")],
    "url:redis://environment:6379"
)]
#[case::explicit_host_drops_environment_url(
    "general_settings:\n  coordination_redis:\n    host: configured",
    &[("REDIS_URL", "redis://environment:6379")],
    "host:configured:6379:0:false::none"
)]
#[case::yaml_cluster(
    "general_settings:\n  coordination_redis:\n    startup_nodes:\n      - host: node-a\n        port: 7000\n      - host: node-b\n        port: \"7001\"",
    &[],
    "cluster:node-a:7000,node-b:7001"
)]
#[case::json_environment_cluster(
    "model_list: []",
    &[("REDIS_CLUSTER_NODES", r#"[{"host":"node-a","port":"7000"},{"host":"node-b","port":7001}]"#)],
    "cluster:node-a:7000,node-b:7001"
)]
#[case::missing_reference_keeps_environment_fallback(
    "general_settings:\n  coordination_redis:\n    host: os.environ/MISSING_HOST",
    &[("REDIS_HOST", "environment")],
    "host:environment:6379:0:false::none"
)]
#[case::configured_values_beat_the_same_environment_values(
    "general_settings:\n  coordination_redis:\n    host: configured\n    port: 6380\n    db: 3",
    &[("REDIS_HOST", "environment"), ("REDIS_PORT", "6390"), ("REDIS_DB", "9")],
    "host:configured:6380:3:false::none"
)]
#[case::environment_url_beats_the_environment_host(
    "model_list: []",
    &[("REDIS_URL", "redis://environment:6379"), ("REDIS_HOST", "ignored")],
    "url:redis://environment:6379"
)]
#[case::startup_nodes_beat_a_url(
    "general_settings:\n  coordination_redis:\n    url: redis://single:6379\n    startup_nodes:\n      - host: node-a\n        port: 7000",
    &[],
    "cluster:node-a:7000"
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
        RedisEndpoint::Url { url, .. } => format!("url:{}", url.expose()),
        RedisEndpoint::Sentinel(sentinel) => format!("sentinel:{}", sentinel.service_name),
        RedisEndpoint::Host(host) => format!(
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
        RedisEndpoint::Cluster(cluster) => format!(
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
#[case::retry_on_timeout_from_environment(
    "model_list: []",
    &[("REDIS_HOST", "redis"), ("REDIS_RETRY_ON_TIMEOUT", "true")],
    "REDIS_RETRY_ON_TIMEOUT"
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
        username: username.map(litellm_auth_types::SecretValue::new),
        password: password.map(litellm_auth_types::SecretValue::new),
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
    assert!(matches!(resolved.endpoint, RedisEndpoint::Host(host) if host.tls.is_some()));
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
    Some(RedisTls { ca_certs: None, certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::every_option(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    ssl_certfile: /client.pem\n    ssl_keyfile: /client.key\n    ssl_cert_reqs: none\n    ssl_check_hostname: true",
    &[],
    Some(RedisTls { ca_certs: Some("/ca.pem".to_owned()), certfile: Some("/client.pem".to_owned()), keyfile: Some("/client.key".to_owned()), verify_certificate: false, check_hostname: true })
)]
#[case::options_from_the_environment(
    "    ssl: true",
    &[("REDIS_SSL_CA_CERTS", "/env-ca.pem"), ("REDIS_SSL_CERT_REQS", "optional")],
    Some(RedisTls { ca_certs: Some("/env-ca.pem".to_owned()), certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::gcp_ca_certs_replace_the_configured_ones(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    gcp_service_account: account\n    gcp_ssl_ca_certs: /gcp-ca.pem",
    &[],
    Some(RedisTls { ca_certs: Some("/gcp-ca.pem".to_owned()), certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::options_without_ssl_stay_plaintext("    ssl_ca_certs: /ca.pem", &[], None)]
fn tls_options_resolve_like_redis_py(
    #[case] block: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: Option<RedisTls>,
) {
    let yaml = format!("general_settings:\n  coordination_redis:\n    host: redis\n{block}");

    let resolved = resolve(&yaml, environment).unwrap();

    assert!(matches!(resolved.endpoint, RedisEndpoint::Host(host) if host.tls == expected));
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
fn sentinel_nodes_resolve_to_a_sentinel_endpoint(#[case] yaml: &str, #[case] environment: &[(&str, &str)]) {
    let RedisEndpoint::Sentinel(sentinel) = resolve(yaml, environment).unwrap().endpoint else {
        panic!("expected a sentinel endpoint");
    };

    assert_eq!(
        sentinel.sentinels,
        vec![
            RedisNode { host: "sentinel-a".to_owned(), port: 26379 },
            RedisNode { host: "sentinel-b".to_owned(), port: 26380 },
        ]
    );
    assert_eq!(sentinel.service_name, "primary");
    assert_eq!(sentinel.sentinel_password.as_ref().map(|password| password.expose()), Some("sentinel-secret"));
    assert_eq!(sentinel.db, 2);
}

#[rstest]
#[case::a_url_beats_sentinels(
    "general_settings:\n  coordination_redis:\n    url: redis://single:6379\n    sentinel_nodes: [[s, 26379]]\n    service_name: primary",
    "url:redis://single:6379"
)]
#[case::sentinels_beat_a_host(
    "general_settings:\n  coordination_redis:\n    host: ignored\n    sentinel_nodes: [[s, 26379]]\n    service_name: primary",
    "sentinel:primary"
)]
fn sentinel_precedence(#[case] yaml: &str, #[case] expected: &str) {
    let resolved = resolve(yaml, &[]).unwrap();
    let description = match resolved.endpoint {
        RedisEndpoint::Url { url, .. } => format!("url:{}", url.expose()),
        RedisEndpoint::Sentinel(sentinel) => format!("sentinel:{}", sentinel.service_name),
        other => panic!("unexpected {other:?}"),
    };

    assert_eq!(description, expected);
}

#[rstest]
fn sentinels_without_a_service_name_fail() {
    let error = resolve("general_settings:\n  coordination_redis:\n    sentinel_nodes: [[s, 26379]]", &[]).unwrap_err();

    assert!(matches!(error, RedisError::NoSentinelService), "{error:?}");
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
#[case::port_from_the_environment("model_list: []", &[("REDIS_HOST", "redis"), ("REDIS_PORT", "abc")], "REDIS_PORT")]
fn invalid_values_name_their_source(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] source: &str,
) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(matches!(&error, RedisError::Invalid { setting, .. } if setting == source), "{error:?}");
}

#[rstest]
#[case::falls_back_to_the_socket_timeout("    socket_timeout: 1.5", Some(1500), None)]
#[case::explicit_connect_timeout("    socket_timeout: 1.5\n    socket_connect_timeout: 0.25", Some(1500), Some(250))]
fn connect_timeout_is_resolved_separately(
    #[case] block: &str,
    #[case] timeout_ms: Option<u128>,
    #[case] connect_ms: Option<u128>,
) {
    let resolved = resolve(&format!("general_settings:\n  coordination_redis:\n    host: redis\n{block}"), &[]).unwrap();

    assert_eq!(resolved.timeout.map(|timeout| timeout.as_millis()), timeout_ms);
    assert_eq!(resolved.connect_timeout.map(|timeout| timeout.as_millis()), connect_ms);
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

    assert!(matches!(resolved.endpoint, RedisEndpoint::Host(_)));
}

#[rstest]
fn a_missing_canonical_startup_nodes_reference_shadows_the_cache_alias() {
    let resolved = resolve_cache(
        "litellm_settings:\n  cache_params:\n    type: redis\n    host: cache\n    startup_nodes: os.environ/MISSING_NODES\n    redis_startup_nodes:\n      - host: alias\n        port: 7100",
    )
    .unwrap();

    assert!(matches!(resolved.endpoint, RedisEndpoint::Host(_)));
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
#[case::not_a_number("slow")]
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
fn defaults_socket_timeout_for_the_gateway() {
    let resolved = resolve(
        "general_settings:\n  coordination_redis:\n    host: redis",
        &[],
    )
    .unwrap();

    assert_eq!(resolved.timeout, None);
}

#[rstest]
fn debug_redacts_redis_secrets_and_url() {
    let config = Config::from_yaml(
        "general_settings:\n  coordination_redis:\n    url: redis://user:password@redis:6379\n    username: credential-user\n    password: credential-password",
    )
    .unwrap();
    let settings = config.general_settings.coordination_redis.unwrap();
    let resolved = settings.resolve(&|_| None).unwrap();
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

    assert!(matches!(resolved.endpoint, RedisEndpoint::Host(_)));
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

#[derive(Debug, PartialEq)]
struct ContractView {
    target: String,
    db: Option<i64>,
    auth: String,
    tls: Option<String>,
    namespace: Option<String>,
    socket_timeout: f64,
    connect_timeout: Option<f64>,
}

impl ContractView {
    fn new(
        target: &str,
        db: Option<i64>,
        auth: &str,
        tls: Option<&str>,
        namespace: Option<&str>,
        socket_timeout: f64,
        connect_timeout: Option<f64>,
    ) -> Self {
        Self {
            target: target.to_owned(),
            db,
            auth: auth.to_owned(),
            tls: tls.map(str::to_owned),
            namespace: namespace.map(str::to_owned),
            socket_timeout,
            connect_timeout,
        }
    }
}

#[derive(Debug)]
enum Outcome {
    Resolves(ContractView),
    Invalid,
    Rejected(&'static str),
}

fn joined(nodes: &[RedisNode]) -> String {
    nodes
        .iter()
        .map(|node| format!("{}:{}", node.host, node.port))
        .collect::<Vec<_>>()
        .join(",")
}

fn tls_view(tls: Option<&RedisTls>) -> Option<String> {
    tls.map(|tls| {
        format!(
            "ca={},verify={},hostname={}",
            tls.ca_certs.as_deref().unwrap_or("-"),
            tls.verify_certificate,
            tls.check_hostname
        )
    })
}

fn view(resolved: ResolvedRedis) -> ContractView {
    let (target, db, tls) = match &resolved.endpoint {
        RedisEndpoint::Host(host) => (format!("host:{}:{}", host.host, host.port), Some(host.db), tls_view(host.tls.as_ref())),
        RedisEndpoint::Url { url, tls } => {
            let tls = url.expose().starts_with("rediss://").then_some(tls);
            (format!("url:{}", url.expose()), None, tls_view(tls))
        }
        RedisEndpoint::Cluster(cluster) => (format!("cluster:{}", joined(&cluster.nodes)), None, tls_view(cluster.tls.as_ref())),
        RedisEndpoint::Sentinel(sentinel) => (
            format!("sentinel:{}:{}", sentinel.service_name, joined(&sentinel.sentinels)),
            Some(sentinel.db),
            tls_view(sentinel.tls.as_ref()),
        ),
    };
    let secret = |value: &Option<litellm_auth_types::SecretValue>| {
        value.as_ref().map_or("-".to_owned(), |value| value.expose().to_owned())
    };
    let auth = match &resolved.auth {
        RedisAuth::Static { username, password } => format!("static:{}:{}", secret(username), secret(password)),
        RedisAuth::Host => "host".to_owned(),
        RedisAuth::GcpIam { service_account } => format!("gcp:{service_account}"),
        RedisAuth::AzureAd(azure) => format!("azure:{}", azure.username.as_deref().unwrap_or("-")),
        RedisAuth::AwsIam(aws) => format!("aws:{}:{}:{}:{}", aws.user_name, aws.cache_name, aws.region, aws.serverless),
    };
    ContractView {
        target,
        db,
        auth,
        tls,
        namespace: resolved.namespace,
        socket_timeout: resolved.timeout.map_or(5.0, |timeout| timeout.as_secs_f64()),
        connect_timeout: resolved.connect_timeout.map(|timeout| timeout.as_secs_f64()),
    }
}

// Each expected value matched litellm/_redis.py `_get_redis_client_logic` plus `_credential_provider_auth_kwargs`
// (redis-py 5.3.1) on main as of 2026-10-29, through RedisCache's kwargs split. Python and Rust pods share one
// Redis during the switch, so Rust keeps resolving these settings the way Python did
#[rstest]
#[case::host_defaults(
    r#"{"host": "redis"}"#,
    &[],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "static:-:-", None, None, 5.0, None))
)]
#[case::host_with_every_basic_setting(
    r#"{"host": "redis", "port": 6380, "db": 2, "ssl": true, "username": "user", "password": "pass", "namespace": "app", "socket_timeout": 1.5}"#,
    &[],
    Outcome::Resolves(ContractView::new("host:redis:6380", Some(2), "static:user:pass", Some("ca=-,verify=true,hostname=false"), Some("app"), 1.5, None))
)]
#[case::environment_host_port_and_password(
    r#"{}"#,
    &[("REDIS_HOST", "environment"), ("REDIS_PORT", "6390"), ("REDIS_PASSWORD", "environment-secret"), ("REDIS_SSL", "True")],
    Outcome::Resolves(ContractView::new("host:environment:6390", Some(0), "static:-:environment-secret", Some("ca=-,verify=true,hostname=false"), None, 5.0, None))
)]
#[case::environment_url(
    r#"{}"#,
    &[("REDIS_URL", "redis://environment:6379/1")],
    Outcome::Resolves(ContractView::new("url:redis://environment:6379/1", None, "static:-:-", None, None, 5.0, None))
)]
#[case::an_explicit_host_drops_the_environment_url(
    r#"{"host": "configured"}"#,
    &[("REDIS_URL", "redis://environment:6379")],
    Outcome::Resolves(ContractView::new("host:configured:6379", Some(0), "static:-:-", None, None, 5.0, None))
)]
#[case::a_url_drops_the_password_setting_and_keeps_the_username(
    r#"{"url": "rediss://cache:6380", "username": "user", "password": "dropped"}"#,
    &[],
    Outcome::Resolves(ContractView::new("url:rediss://cache:6380", None, "static:user:-", Some("ca=-,verify=true,hostname=false"), None, 5.0, None))
)]
#[case::startup_nodes_beat_a_url(
    r#"{"url": "redis://single:6379", "startup_nodes": [{"host": "node-a", "port": 7000}], "password": "cluster-secret"}"#,
    &[],
    Outcome::Resolves(ContractView::new("cluster:node-a:7000", None, "static:-:cluster-secret", None, None, 5.0, None))
)]
#[case::environment_cluster_nodes(
    r#"{}"#,
    &[("REDIS_CLUSTER_NODES", "[{\"host\": \"node-a\", \"port\": \"7000\"}, {\"host\": \"node-b\", \"port\": 7001}]")],
    Outcome::Resolves(ContractView::new("cluster:node-a:7000,node-b:7001", None, "static:-:-", None, None, 5.0, None))
)]
#[case::sentinel_from_settings(
    r#"{"sentinel_nodes": [["sentinel-a", 26379], ["sentinel-b", 26380]], "service_name": "primary", "sentinel_password": "sentinel-secret", "password": "data-secret"}"#,
    &[],
    Outcome::Resolves(ContractView::new("sentinel:primary:sentinel-a:26379,sentinel-b:26380", Some(0), "static:-:data-secret", None, None, 5.0, None))
)]
#[case::sentinel_from_the_environment(
    r#"{}"#,
    &[("REDIS_SENTINEL_NODES", "[[\"sentinel-a\", 26379]]"), ("REDIS_SERVICE_NAME", "primary")],
    Outcome::Resolves(ContractView::new("sentinel:primary:sentinel-a:26379", Some(0), "static:-:-", None, None, 5.0, None))
)]
#[case::sentinels_beat_a_host(
    r#"{"host": "ignored", "sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"}"#,
    &[],
    Outcome::Resolves(ContractView::new("sentinel:primary:sentinel-a:26379", Some(0), "static:-:-", None, None, 5.0, None))
)]
#[case::a_url_beats_sentinels(
    r#"{"url": "redis://single:6379", "sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"}"#,
    &[],
    Outcome::Resolves(ContractView::new("url:redis://single:6379", None, "static:-:-", None, None, 5.0, None))
)]
#[case::an_explicit_credential_provider_beats_cloud_auth(
    r#"{"host": "redis", "password": "dropped", "credential_provider": true, "gcp_service_account": "account", "aws_iam_auth": true}"#,
    &[("REDIS_AZURE_AD_TOKEN", "true")],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "host", None, None, 5.0, None))
)]
#[case::gcp_beats_azure_and_aws(
    r#"{"host": "redis", "password": "dropped", "azure_redis_ad_token": "True", "aws_iam_auth": true}"#,
    &[("REDIS_GCP_SERVICE_ACCOUNT", "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com")],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "gcp:projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com", None, None, 5.0, None))
)]
#[case::gcp_ca_certs_replace_the_configured_ones_under_ssl(
    r#"{"host": "redis", "ssl": true, "ssl_ca_certs": "/configured-ca.pem", "gcp_service_account": "account", "gcp_ssl_ca_certs": "/gcp-ca.pem"}"#,
    &[],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "gcp:account", Some("ca=/gcp-ca.pem,verify=true,hostname=false"), None, 5.0, None))
)]
#[case::azure_takes_the_environment_username(
    r#"{"host": "redis", "username": "dropped", "azure_redis_ad_token": true, "azure_client_id": "client", "azure_tenant_id": "tenant", "azure_client_secret": "secret", "aws_iam_auth": true}"#,
    &[("REDIS_USERNAME", "environment-user")],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "azure:environment-user", None, None, 5.0, None))
)]
#[case::azure_without_a_username_sends_only_the_token(
    r#"{"host": "redis", "azure_redis_ad_token": true, "azure_client_id": "client", "azure_tenant_id": "tenant", "azure_client_secret": "secret"}"#,
    &[],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "azure:-", None, None, 5.0, None))
)]
#[case::aws_iam_over_tls_with_the_environment_region(
    r#"{"host": "cache-1", "ssl": true, "password": "dropped", "aws_iam_auth": "yes", "aws_iam_user_name": "iam-user", "aws_iam_cache_name": "cache-1"}"#,
    &[("AWS_DEFAULT_REGION", "us-east-1")],
    Outcome::Resolves(ContractView::new("host:cache-1:6379", Some(0), "aws:iam-user:cache-1:us-east-1:false", Some("ca=-,verify=true,hostname=false"), None, 5.0, None))
)]
#[case::aws_iam_serverless_over_a_rediss_url(
    r#"{"url": "rediss://cache-1:6379", "aws_iam_auth": true, "aws_iam_user_name": "iam-user", "aws_iam_cache_name": "cache-1", "aws_iam_region": "eu-west-1", "aws_iam_serverless": true}"#,
    &[],
    Outcome::Resolves(ContractView::new("url:rediss://cache-1:6379", None, "aws:iam-user:cache-1:eu-west-1:true", Some("ca=-,verify=true,hostname=false"), None, 5.0, None))
)]
#[case::aws_iam_without_tls_fails(
    r#"{"host": "cache-1", "aws_iam_auth": true, "aws_iam_user_name": "iam-user", "aws_iam_cache_name": "cache-1", "aws_iam_region": "us-east-1"}"#,
    &[],
    Outcome::Invalid
)]
#[case::aws_iam_without_a_region_fails(
    r#"{"host": "cache-1", "ssl": true, "aws_iam_auth": true, "aws_iam_user_name": "iam-user", "aws_iam_cache_name": "cache-1"}"#,
    &[],
    Outcome::Invalid
)]
#[case::tls_options_from_settings(
    r#"{"host": "redis", "ssl": true, "ssl_ca_certs": "/ca.pem", "ssl_cert_reqs": "none", "ssl_check_hostname": true}"#,
    &[],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "static:-:-", Some("ca=/ca.pem,verify=false,hostname=true"), None, 5.0, None))
)]
#[case::tls_options_from_the_environment(
    r#"{"host": "redis"}"#,
    &[("REDIS_SSL", "true"), ("REDIS_SSL_CA_CERTS", "/env-ca.pem"), ("REDIS_SSL_CERT_REQS", "optional")],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "static:-:-", Some("ca=/env-ca.pem,verify=true,hostname=false"), None, 5.0, None))
)]
#[case::connect_timeout_is_separate_from_the_socket_timeout(
    r#"{"host": "redis", "socket_timeout": 2.5, "socket_connect_timeout": 0.25}"#,
    &[],
    Outcome::Resolves(ContractView::new("host:redis:6379", Some(0), "static:-:-", None, None, 2.5, Some(0.25)))
)]
#[case::environment_references_resolve(
    r#"{"host": "os.environ/COORD_HOST", "password": "os.environ/COORD_PASSWORD", "namespace": "responses"}"#,
    &[("COORD_HOST", "referenced"), ("COORD_PASSWORD", "referenced-secret")],
    Outcome::Resolves(ContractView::new("host:referenced:6379", Some(0), "static:-:referenced-secret", None, Some("responses"), 5.0, None))
)]
#[case::no_target_fails(
    r#"{"port": 6379}"#,
    &[],
    Outcome::Invalid
)]
#[case::ssl_ciphers_are_accepted_by_python_and_rejected_by_rust(
    r#"{"host": "redis", "ssl": true, "ssl_ciphers": "HIGH"}"#,
    &[],
    Outcome::Rejected("ssl_ciphers")
)]
#[case::retry_on_timeout_is_accepted_by_python_and_rejected_by_rust(
    r#"{"host": "redis", "retry_on_timeout": true}"#,
    &[],
    Outcome::Rejected("retry_on_timeout")
)]
fn resolution_matches_python_redis(
    #[case] kwargs: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: Outcome,
) {
    let settings: RedisSettings = serde_json::from_str(kwargs).unwrap();
    let environment: HashMap<&str, &str> = environment.iter().copied().collect();

    let result = settings.resolve(&|name| environment.get(name).map(|value| (*value).to_owned()));

    match expected {
        Outcome::Resolves(view_expected) => assert_eq!(view(result.unwrap()), view_expected),
        Outcome::Invalid => assert!(
            matches!(&result, Err(error) if !matches!(error, RedisError::Unsupported { .. })),
            "{result:?}"
        ),
        Outcome::Rejected(setting) => {
            assert!(matches!(&result, Err(RedisError::Unsupported { setting: rejected }) if rejected == setting), "{result:?}")
        }
    }
}
