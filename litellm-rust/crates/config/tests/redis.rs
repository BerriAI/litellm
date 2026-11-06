use std::{
    collections::HashMap,
    fmt::Write,
    sync::{Arc, Mutex},
};

use litellm_config::{CacheSettings, Config, Error, RedisEndpoint, RedisSettings, ResolvedRedis};
use rstest::rstest;
use tracing::{Event, Subscriber, field::Visit};
use tracing_subscriber::{Layer, layer::Context, prelude::*};

fn resolve(
    yaml: &str,
    environment: &[(&str, &str)],
) -> Result<litellm_config::ResolvedRedis, Error> {
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

fn resolve_cache(yaml: &str) -> Result<ResolvedRedis, Error> {
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
        RedisEndpoint::Url(url) => format!("url:{}", url.expose()),
        RedisEndpoint::Host(host) => format!(
            "host:{}:{}:{}:{}:{}:{}",
            host.host,
            host.port,
            host.db,
            host.tls,
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
#[case::custom_tls_file(
    "general_settings:\n  coordination_redis:\n    host: redis\n    ssl_ca_certs: /ca.pem",
    &[],
    "ssl_ca_certs"
)]
#[case::gcp_auth_from_config(
    "general_settings:\n  coordination_redis:\n    host: redis\n    gcp_service_account: account",
    &[],
    "gcp_service_account"
)]
#[case::gcp_auth_from_environment(
    "model_list: []",
    &[("REDIS_HOST", "redis"), ("REDIS_GCP_SERVICE_ACCOUNT", "account")],
    "REDIS_GCP_SERVICE_ACCOUNT"
)]
#[case::azure_auth_from_config(
    "general_settings:\n  coordination_redis:\n    host: redis\n    azure_redis_ad_token: true",
    &[],
    "azure_redis_ad_token"
)]
#[case::azure_auth_from_environment(
    "model_list: []",
    &[("REDIS_HOST", "redis"), ("REDIS_AZURE_AD_TOKEN", "true")],
    "REDIS_AZURE_AD_TOKEN"
)]
#[case::aws_auth_from_config(
    "general_settings:\n  coordination_redis:\n    host: redis\n    aws_iam_auth: true",
    &[],
    "aws_iam_auth"
)]
#[case::aws_auth_from_environment(
    "model_list: []",
    &[("REDIS_HOST", "redis"), ("REDIS_AWS_IAM_AUTH", "true")],
    "REDIS_AWS_IAM_AUTH"
)]
#[case::sentinel_from_config(
    "general_settings:\n  coordination_redis:\n    sentinel_nodes: [[redis, 26379]]",
    &[],
    "sentinel_nodes"
)]
#[case::sentinel_from_environment(
    "model_list: []",
    &[("REDIS_SENTINEL_NODES", "[[\"redis\",26379]]")],
    "REDIS_SENTINEL_NODES"
)]
fn rejected_settings_name_their_source(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: &str,
) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(matches!(error, Error::RedisSetting(setting) if setting == expected));
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

    let RedisEndpoint::Host(host) = resolved.endpoint else {
        panic!("expected a host endpoint");
    };
    assert_eq!(
        host.password.as_ref().map(|password| password.expose()),
        Some("environment-secret")
    );
    assert!(host.tls);
}

#[rstest]
#[case::only_a_port("general_settings:\n  coordination_redis:\n    port: 6379", &[])]
#[case::a_structured_host_with_an_environment_target(
    "general_settings:\n  coordination_redis:\n    host: []",
    &[("REDIS_HOST", "environment"), ("REDIS_URL", "redis://environment:6379")]
)]
fn a_block_without_a_usable_target_fails(#[case] yaml: &str, #[case] environment: &[(&str, &str)]) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(matches!(
        error,
        Error::RedisInvalid {
            setting: "host",
            ..
        }
    ));
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
            ("REDIS_SSL_CERT_REQS", "private-option-value"),
        ],
    )
    .unwrap_err();
    let message = error.to_string();

    assert!(message.contains("REDIS_SSL_CERT_REQS"));
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
        Error::RedisInvalid {
            setting: "socket_timeout",
            ..
        }
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
