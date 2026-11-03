use std::collections::{BTreeMap, HashMap};

use litellm_config::{
    CacheSettings, Config, ConfiguredRedis, Expected, RedisError, SecretSnapshot, Value,
};
use litellm_redis_types::{
    Auth, AwsIamAuth, AzureAdAuth, HealthCheck, Node, ProviderHandle, Resolution, StaticAuth,
    Target, TlsOptions,
};
use rstest::rstest;
use serde_json::json;

fn resolution(
    configured: &ConfiguredRedis,
    environment: &HashMap<String, String>,
) -> Result<Resolution, RedisError> {
    let lookup = |name: &str| environment.get(name).cloned();
    configured.resolve(
        &SecretSnapshot::read(configured.secret_names(), &lookup),
        &lookup,
    )
}

fn redis_cache_kwargs(kwargs: serde_json::Value) -> ConfiguredRedis {
    ConfiguredRedis::redis_cache_kwargs(
        serde_json::from_value::<BTreeMap<String, Value>>(kwargs).unwrap(),
    )
}

const PROVIDER: ProviderHandle = ProviderHandle::new(7);

// python-bridge hands a credential_provider object over as a handle, so the contract cases do the same
fn resolve_kwargs(
    kwargs: serde_json::Value,
    environment: &HashMap<String, String>,
) -> Result<Resolution, RedisError> {
    let has_provider = kwargs
        .get("credential_provider")
        .is_some_and(|provider| !provider.is_null());
    let configured = redis_cache_kwargs(kwargs);
    let configured = match has_provider {
        true => configured.with_credential_provider(PROVIDER),
        false => configured,
    };
    resolution(&configured, environment)
}

fn coordination_resolution(
    yaml: &str,
    environment: &[(&str, &str)],
) -> Result<Resolution, RedisError> {
    let environment: HashMap<String, String> = environment
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    let config = Config::from_yaml(yaml).unwrap();
    let configured = config
        .general_settings
        .coordination_redis
        .unwrap_or_default()
        .coordination()?;
    resolution(&configured, &environment)
}

fn resolve(yaml: &str, environment: &[(&str, &str)]) -> Result<Resolution, RedisError> {
    coordination_resolution(yaml, environment)
}

fn resolve_cache(yaml: &str) -> Result<Resolution, RedisError> {
    resolve_cache_in(yaml, &[])
}

fn resolve_cache_in(yaml: &str, environment: &[(&str, &str)]) -> Result<Resolution, RedisError> {
    let lookup = |name: &str| {
        environment
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_owned())
    };
    let config = Config::from_yaml(yaml).unwrap();
    let configured = match config.litellm_settings.cache_params.unwrap() {
        CacheSettings::Redis(settings) => settings.cache_params(),
        CacheSettings::Other(_) => panic!("expected a Redis cache"),
    };
    configured.resolve(
        &SecretSnapshot::read(configured.secret_names(), &lookup),
        &lookup,
    )
}

fn cluster_ports(resolved: &Resolution) -> Vec<u16> {
    match &resolved.target {
        Target::Cluster(cluster) => cluster.nodes.iter().map(|node| node.port).collect(),
        target => panic!("expected a cluster, got {target:?}"),
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
    let description = match resolved.target {
        Target::Unix(socket) => format!("unix:{}", socket.path.display()),
        Target::Url(url) => format!("url:{}", url.url.expose()),
        Target::Sentinel(sentinel) => format!("sentinel:{}", sentinel.service_name),
        Target::Standalone(host) => format!(
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
        Target::Cluster(cluster) => format!(
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
fn unsupported_settings_resolve_and_are_named_by_their_source(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: &str,
) {
    let resolution = coordination_resolution(yaml, environment).unwrap();

    assert_eq!(
        resolution.target,
        Target::Standalone(litellm_redis_types::Standalone {
            host: "redis".into(),
            port: 6379,
            db: 0,
            tls: None,
        })
    );
    let [ignored] = resolution.ignored.as_slice() else {
        panic!("expected one ignored setting, got {:?}", resolution.ignored);
    };
    assert_eq!(ignored.setting, expected);
    assert!(!ignored.consequence.is_empty());
}

fn static_auth(username: Option<&str>, password: Option<&str>) -> Auth {
    Auth::Static(StaticAuth {
        username: username.map(litellm_secret_value::SecretValue::new),
        password: password.map(litellm_secret_value::SecretValue::new),
    })
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
    assert!(matches!(resolved.target, Target::Standalone(host) if host.tls.is_some()));
}

fn azure(username: Option<&str>) -> Auth {
    Auth::AzureAd(AzureAdAuth {
        client_id: Some("client".to_owned()),
        tenant_id: None,
        client_secret: None,
        username: username.map(str::to_owned),
    })
}

fn aws(serverless: bool) -> Auth {
    Auth::AwsIam(AwsIamAuth {
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
#[case::gcp_beats_azure_and_aws(
    "general_settings:\n  coordination_redis:\n    host: redis\n    password: pass\n    azure_redis_ad_token: \"True\"\n    aws_iam_auth: true",
    &[("REDIS_GCP_SERVICE_ACCOUNT", "projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com")],
    Auth::GcpIam { service_account: "projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com".to_owned() }
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
    #[case] expected: Auth,
) {
    assert_eq!(resolve(yaml, environment).unwrap().auth, expected);
}

#[rstest]
fn a_provider_handle_beats_every_cloud_auth_and_static_credentials() {
    let configured = redis_cache_kwargs(json!({
        "host": "redis", "password": "pass", "credential_provider": "host-object",
        "gcp_service_account": "account", "aws_iam_auth": true, "azure_redis_ad_token": "true",
    }))
    .with_credential_provider(PROVIDER);

    let resolved = resolution(&configured, &HashMap::new()).unwrap();

    assert_eq!(resolved.auth, Auth::Provider(PROVIDER));
}

#[rstest]
#[case::coordination(
    "general_settings:\n  coordination_redis:\n    host: redis\n    credential_provider: host-object",
    &[]
)]
#[case::without_a_target(
    "general_settings:\n  coordination_redis:\n    credential_provider: host-object",
    &[("REDIS_HOST", "redis")]
)]
fn a_credential_provider_without_a_host_handle_fails(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
) {
    let error = resolve(yaml, environment).unwrap_err();

    assert!(
        matches!(error, RedisError::ProviderWithoutHandle),
        "{error}"
    );
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
    Some(TlsOptions { ca_certs: None, ca_data: None, ca_path: None, certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::every_option(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    ssl_certfile: /client.pem\n    ssl_keyfile: /client.key\n    ssl_cert_reqs: none\n    ssl_check_hostname: true",
    &[],
    Some(TlsOptions { ca_certs: Some("/ca.pem".into()), ca_data: None, ca_path: None, certfile: Some("/client.pem".into()), keyfile: Some("/client.key".into()), verify_certificate: false, check_hostname: true })
)]
#[case::options_from_the_environment(
    "    ssl: true",
    &[("REDIS_SSL_CA_CERTS", "/env-ca.pem"), ("REDIS_SSL_CERT_REQS", "optional")],
    Some(TlsOptions { ca_certs: Some("/env-ca.pem".into()), ca_data: None, ca_path: None, certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::gcp_ca_certs_replace_the_configured_ones(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    gcp_service_account: account\n    gcp_ssl_ca_certs: /gcp-ca.pem",
    &[],
    Some(TlsOptions { ca_certs: Some("/gcp-ca.pem".into()), ca_data: None, ca_path: None, certfile: None, keyfile: None, verify_certificate: true, check_hostname: false })
)]
#[case::options_without_ssl_stay_plaintext("    ssl_ca_certs: /ca.pem", &[], None)]
fn tls_options_resolve_like_redis_py(
    #[case] block: &str,
    #[case] environment: &[(&str, &str)],
    #[case] expected: Option<TlsOptions>,
) {
    let yaml = format!("general_settings:\n  coordination_redis:\n    host: redis\n{block}");

    let resolved = resolve(&yaml, environment).unwrap();

    assert!(matches!(resolved.target, Target::Standalone(host) if host.tls == expected));
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
    let Target::Sentinel(sentinel) = resolve(yaml, environment).unwrap().target else {
        panic!("expected a sentinel endpoint");
    };

    assert_eq!(
        sentinel.monitors,
        vec![
            Node {
                host: "sentinel-a".to_owned(),
                port: 26379
            },
            Node {
                host: "sentinel-b".to_owned(),
                port: 26380
            },
        ]
    );
    assert_eq!(sentinel.service_name, "primary");
    assert_eq!(
        sentinel.monitor_auth,
        Some(StaticAuth {
            username: None,
            password: Some(litellm_secret_value::SecretValue::new("sentinel-secret")),
        })
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
    let description = match resolved.target {
        Target::Url(url) => format!("url:{}", url.url.expose()),
        Target::Sentinel(sentinel) => format!("sentinel:{}", sentinel.service_name),
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

    assert!(matches!(resolved.target, Target::Standalone(_)));
}

#[rstest]
fn a_missing_canonical_startup_nodes_reference_shadows_the_cache_alias() {
    let resolved = resolve_cache(
        "litellm_settings:\n  cache_params:\n    type: redis\n    host: cache\n    startup_nodes: os.environ/MISSING_NODES\n    redis_startup_nodes:\n      - host: alias\n        port: 7100",
    )
    .unwrap();

    assert!(matches!(resolved.target, Target::Standalone(_)));
}

#[rstest]
fn unsupported_environment_settings_never_carry_the_value() {
    let resolution = coordination_resolution(
        "model_list: []",
        &[
            ("REDIS_HOST", "redis"),
            ("REDIS_SSL_CIPHERS", "private-option-value"),
        ],
    )
    .unwrap();
    let ignored = format!("{:?}", resolution.ignored);

    assert!(ignored.contains("REDIS_SSL_CIPHERS"), "{ignored}");
    assert!(!ignored.contains("private-option-value"), "{ignored}");
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
        "general_settings:\n  coordination_redis:\n    url: redis://user:password@redis:6379\n    username: credential-user\n    password: credential-password\n    sentinel_kwargs: {password: credential-passthrough}",
    )
    .unwrap();
    let settings = config.general_settings.coordination_redis.unwrap();
    let configured = settings.coordination().unwrap();
    let resolved = resolution(&configured, &HashMap::new()).unwrap();
    let debug = format!("{settings:?} {configured:?} {resolved:?}");

    assert!(!debug.contains("redis://user:password"));
    assert!(!debug.contains("credential-user"));
    assert!(!debug.contains("credential-password"));
    assert!(!debug.contains("credential-passthrough"));
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

    assert!(matches!(resolved.target, Target::Standalone(_)));
}

#[rstest]
#[case::configured(
    "general_settings:\n  coordination_redis:\n    host: redis\n    max_connections: 47",
    &[],
    "max_connections"
)]
#[case::from_the_environment(
    "general_settings:\n  coordination_redis:\n    host: redis",
    &[("REDIS_MAX_CONNECTIONS", "47")],
    "REDIS_MAX_CONNECTIONS"
)]
fn ignored_settings_are_named_by_their_source(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] source: &str,
) {
    let resolution = coordination_resolution(yaml, environment).unwrap();

    let settings: Vec<&str> = resolution
        .ignored
        .iter()
        .map(|ignored| ignored.setting.as_str())
        .collect();
    assert_eq!(settings, vec![source]);
}

#[rstest]
#[case::configured(
    json!({"host": "redis", "client_name": "worker", "read_from_replicas": true, "connection_class": "SSLConnection", "retry_on_error": null}),
    &[],
    json!({"client_name": "worker", "read_from_replicas": true}),
    &["connection_class", "retry_on_error"],
)]
#[case::from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_CLIENT_NAME", "worker")],
    json!({"client_name": "worker"}),
    &[],
)]
fn settings_rust_does_not_model_reach_the_host_instead_of_being_dropped(
    #[case] kwargs: serde_json::Value,
    #[case] environment: &[(&str, &str)],
    #[case] passthrough: serde_json::Value,
    #[case] host_objects: &[&str],
) {
    let environment: HashMap<String, String> = environment
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();

    let resolved = resolve_kwargs(kwargs, &environment).unwrap();

    let passed: serde_json::Map<String, serde_json::Value> = resolved
        .passthrough
        .iter()
        .map(|(key, value)| (key.to_owned(), value.clone()))
        .collect();
    assert_eq!(serde_json::Value::Object(passed), passthrough);
    assert_eq!(
        resolved
            .host_objects
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        host_objects
    );
}

#[rstest]
#[case::read_on_a_plain_url(
    "redis://cache:6379/1?db=3&username=query-user&password=query-secret",
    Some(3),
    Some("query-user"),
    Some("query-secret")
)]
#[case::absent("rediss://cache:6380/1", None, None, None)]
fn a_url_query_carries_db_and_credentials_like_parse_url(
    #[case] url: &str,
    #[case] db: Option<i64>,
    #[case] username: Option<&str>,
    #[case] password: Option<&str>,
) {
    let resolved = resolve_kwargs(json!({"url": url}), &HashMap::new()).unwrap();

    let Target::Url(parsed) = resolved.target else {
        panic!("expected a url target, got {:?}", resolved.target);
    };
    assert_eq!(parsed.db, db);
    assert_eq!(expose(parsed.username.as_ref()), username);
    assert_eq!(expose(parsed.password.as_ref()), password);
    assert!(resolved.ignored.is_empty(), "{:?}", resolved.ignored);
}

#[rstest]
fn a_url_query_db_that_is_not_an_integer_fails_like_parse_url() {
    let error = resolve_kwargs(
        json!({"url": "redis://cache:6379?db=first"}),
        &HashMap::new(),
    )
    .unwrap_err();

    assert!(
        matches!(error, RedisError::Invalid { setting, expected: Expected::Integer } if setting == "url ?db")
    );
}

#[rstest]
#[case::tls_1_2_is_already_met(json!("TLSv1_2"), false)]
#[case::the_enum_name(json!("TLSVersion.TLSv1_2"), false)]
#[case::the_enum_value(json!(771), false)]
#[case::tls_1_3_is_not_enforced(json!("TLSv1_3"), true)]
fn ssl_min_version_warns_only_when_rustls_cannot_meet_it(
    #[case] version: serde_json::Value,
    #[case] warns: bool,
) {
    let resolved = resolve_kwargs(
        json!({"host": "redis", "ssl": true, "ssl_min_version": version}),
        &HashMap::new(),
    )
    .unwrap();

    assert_eq!(
        !resolved.ignored.is_empty(),
        warns,
        "{:?}",
        resolved.ignored
    );
}

fn tls_view(tls: Option<&TlsOptions>) -> serde_json::Value {
    tls.map_or(serde_json::Value::Null, |tls| {
        serde_json::json!({
            "ca_certs": tls.ca_certs,
            "ca_data": tls.ca_data,
            "ca_path": tls.ca_path,
            "certfile": tls.certfile,
            "keyfile": tls.keyfile,
            "verify_certificate": tls.verify_certificate,
            "check_hostname": tls.check_hostname,
        })
    })
}

fn nodes_view(nodes: &[Node]) -> Vec<String> {
    nodes
        .iter()
        .map(|node| format!("{}:{}", node.host, node.port))
        .collect()
}

fn expose(value: Option<&litellm_secret_value::SecretValue>) -> Option<&str> {
    value.map(litellm_secret_value::SecretValue::expose)
}

fn view(resolved: &Resolution) -> serde_json::Value {
    let target = match &resolved.target {
        Target::Unix(socket) => serde_json::json!({
            "mode": "unix", "path": socket.path, "tls": null,
        }),
        Target::Standalone(host) => serde_json::json!({
            "mode": "host",
            "host": host.host,
            "port": host.port,
            "db": host.db,
            "tls": tls_view(host.tls.as_ref()),
        }),
        Target::Url(url) => serde_json::json!({
            "mode": "url",
            "url": url.url.expose(),
            "tls": tls_view(url.tls.as_ref()),
        }),
        Target::Cluster(cluster) => serde_json::json!({
            "mode": "cluster",
            "nodes": nodes_view(&cluster.nodes),
            "tls": tls_view(cluster.tls.as_ref()),
        }),
        Target::Sentinel(sentinel) => serde_json::json!({
            "mode": "sentinel",
            "nodes": nodes_view(&sentinel.monitors),
            "service_name": sentinel.service_name,
            "sentinel_password": expose(sentinel.monitor_auth.as_ref().and_then(|auth| auth.password.as_ref())),
            "db": sentinel.db,
            "tls": tls_view(sentinel.tls.as_ref()),
        }),
    };
    let auth = match &resolved.auth {
        Auth::Static(StaticAuth { username, password }) => serde_json::json!({
            "kind": "static",
            "username": expose(username.as_ref()),
            "password": expose(password.as_ref()),
        }),
        Auth::Provider(_) => serde_json::json!({"kind": "host"}),
        Auth::GcpIam { service_account } => {
            serde_json::json!({"kind": "gcp_iam", "service_account": service_account})
        }
        Auth::AzureAd(azure) => {
            serde_json::json!({"kind": "azure_ad", "username": azure.username})
        }
        Auth::AwsIam(aws) => serde_json::json!({
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
        "health_check_interval": match (&resolved.target, resolved.health_check) {
            (_, HealthCheck::Every(interval)) => Some(interval.as_secs_f64()),
            (Target::Cluster(_), HealthCheck::Unset) => Some(25.0),
            (_, HealthCheck::Unset | HealthCheck::Disabled) => None,
        },
    });
    let serde_json::Value::Object(rest) = rest else {
        unreachable!("the view is an object")
    };
    serde_json::Value::Object(target.into_iter().chain(rest).collect())
}

fn expected_view(fields: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(fields) = fields else {
        unreachable!("expected views are objects")
    };
    let defaults = json!({
        "auth": {"kind": "static", "username": null, "password": null},
        "tls": null,
        "socket_timeout": 5.0,
        "socket_connect_timeout": 5.0,
        "health_check_interval": null,
    });
    let serde_json::Value::Object(defaults) = defaults else {
        unreachable!("the defaults are an object")
    };
    serde_json::Value::Object(defaults.into_iter().chain(fields).collect())
}

#[rstest]
#[case::host_defaults(
    json!({"host": "redis"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::empty_cluster_nodes_use_the_environment(
    json!({"startup_nodes": []}),
    &[("REDIS_CLUSTER_NODES", r#"[{"host": "node-a", "port": 7000}]"#)],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::empty_sentinel_nodes_use_the_environment(
    json!({"sentinel_nodes": [], "service_name": "primary"}),
    &[("REDIS_SENTINEL_NODES", r#"[["sentinel-a", 26379]]"#)],
    Ok(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379"], "service_name": "primary", "sentinel_password": null, "db": 0}))),
)]
#[case::empty_service_name_uses_the_environment(
    json!({"sentinel_nodes": [["sentinel-a", 26379]], "service_name": ""}),
    &[("REDIS_SERVICE_NAME", "primary")],
    Ok(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379"], "service_name": "primary", "sentinel_password": null, "db": 0}))),
)]
#[case::plaintext_ignores_tls_requirements(
    json!({"host": "redis", "ssl_cert_reqs": "strict"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::an_ambiguous_flag_is_refused(
    json!({"host": "redis", "ssl": "on"}),
    &[],
    Err("invalid Redis setting ssl: expected a boolean"),
)]
#[case::empty_sentinel_nodes_without_the_environment_name_no_target(
    json!({"sentinel_nodes": [], "service_name": "primary"}),
    &[],
    Err("either host or url must be specified for redis"),
)]
#[case::ssl_ca_path_is_unsupported(
    json!({"host": "redis", "ssl": true, "ssl_ca_path": "/ca"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": "/ca", "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::host_with_every_basic_setting(
    json!({"db": 2, "host": "redis", "password": "pass", "port": 6380, "socket_timeout": 1.5, "ssl": true, "username": "user"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6380, "db": 2, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "static", "username": "user", "password": "pass"}, "socket_timeout": 1.5, "socket_connect_timeout": 1.5}))),
)]
#[case::environment_host_port_and_password(
    json!({}),
    &[("REDIS_HOST", "environment"), ("REDIS_PASSWORD", "environment-secret"), ("REDIS_PORT", "6390"), ("REDIS_SSL", "True")],
    Ok(expected_view(json!({"mode": "host", "host": "environment", "port": 6390, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "static", "username": null, "password": "environment-secret"}}))),
)]
#[case::string_numbers_are_coerced(
    json!({"db": "3", "host": "redis", "port": "6380", "socket_timeout": "2.5"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6380, "db": 3, "socket_timeout": 2.5, "socket_connect_timeout": 2.5}))),
)]
#[case::padded_and_underscored_numbers_parse_like_python_int_and_float(
    json!({"host": "redis", "port": "6380\n", "db": " 2 ", "socket_timeout": "1_0.5\n"}),
    &[],
    Ok(expected_view(json!({"db": 2, "host": "redis", "mode": "host", "port": 6380, "socket_timeout": 10.5, "socket_connect_timeout": 10.5}))),
)]
#[case::padded_environment_numbers_parse_like_python_int(
    json!({}),
    &[("REDIS_HOST", "redis"), ("REDIS_PORT", "6_380"), ("REDIS_DB", "3\n")],
    Ok(expected_view(json!({"db": 3, "host": "redis", "mode": "host", "port": 6380}))),
)]
#[case::socket_timeout_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_SOCKET_TIMEOUT", "0.75")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "socket_timeout": 0.75, "socket_connect_timeout": 0.75}))),
)]
#[case::environment_url(
    json!({}),
    &[("REDIS_URL", "redis://environment:6379/1")],
    Ok(expected_view(json!({"mode": "url", "url": "redis://environment:6379/1"}))),
)]
#[case::environment_url_beats_the_environment_host(
    json!({}),
    &[("REDIS_HOST", "ignored"), ("REDIS_URL", "redis://from-url:6379")],
    Ok(expected_view(json!({"mode": "url", "url": "redis://from-url:6379"}))),
)]
#[case::an_explicit_host_drops_the_environment_url(
    json!({"host": "configured"}),
    &[("REDIS_URL", "redis://environment:6379")],
    Ok(expected_view(json!({"mode": "host", "host": "configured", "port": 6379, "db": 0}))),
)]
#[case::a_url_drops_host_db_and_password_and_keeps_the_username(
    json!({"db": 4, "host": "dropped", "password": "dropped", "url": "rediss://cache:6380", "username": "user"}),
    &[],
    Ok(expected_view(json!({"mode": "url", "url": "rediss://cache:6380", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "static", "username": "user", "password": null}}))),
)]
#[case::a_url_ignores_tls_settings_outside_the_url(
    json!({"ssl_ca_certs": "/ca.pem", "ssl_cert_reqs": "none", "ssl_check_hostname": true, "url": "rediss://cache:6380"}),
    &[],
    Ok(expected_view(json!({"mode": "url", "url": "rediss://cache:6380", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::startup_nodes_beat_a_url(
    json!({"password": "cluster-secret", "startup_nodes": [{"host": "node-a", "port": 7000}], "url": "redis://single:6379"}),
    &[],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "auth": {"kind": "static", "username": null, "password": "cluster-secret"}, "health_check_interval": 25.0}))),
)]
#[case::environment_cluster_nodes(
    json!({}),
    &[("REDIS_CLUSTER_NODES", r#"[{"host": "node-a", "port": "7000"}, {"host": "node-b", "port": 7001}]"#)],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000", "node-b:7001"], "health_check_interval": 25.0}))),
)]
#[case::cluster_tls_options(
    json!({"ssl": "true", "ssl_ca_certs": "/ca.pem", "ssl_check_hostname": true, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "tls": {"ca_certs": "/ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": true}, "health_check_interval": 25.0}))),
)]
#[case::sentinel_from_settings(
    json!({"password": "data-secret", "sentinel_nodes": [["sentinel-a", 26379], ["sentinel-b", 26380]], "sentinel_password": "sentinel-secret", "service_name": "primary"}),
    &[],
    Ok(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379", "sentinel-b:26380"], "service_name": "primary", "sentinel_password": "sentinel-secret", "db": 0, "auth": {"kind": "static", "username": null, "password": "data-secret"}}))),
)]
#[case::sentinel_from_the_environment(
    json!({}),
    &[("REDIS_SENTINEL_NODES", r#"[["sentinel-a", 26379]]"#), ("REDIS_SENTINEL_PASSWORD", "sentinel-secret"), ("REDIS_SERVICE_NAME", "primary")],
    Ok(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379"], "service_name": "primary", "sentinel_password": "sentinel-secret", "db": 0}))),
)]
#[case::a_host_breaks_sentinel(
    json!({"host": "ignored", "sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"}),
    &[],
    Err("host cannot be combined with sentinel_nodes, which redis-py rejects at startup"),
)]
#[case::a_url_beats_sentinels(
    json!({"sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary", "url": "redis://single:6379"}),
    &[],
    Ok(expected_view(json!({"mode": "url", "url": "redis://single:6379"}))),
)]
#[case::an_explicit_credential_provider_beats_cloud_auth(
    json!({"aws_iam_auth": true, "credential_provider": true, "gcp_service_account": "account", "host": "redis", "password": "dropped"}),
    &[("REDIS_AZURE_AD_TOKEN", "true")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "host"}}))),
)]
#[case::gcp_beats_azure_and_aws(
    json!({"aws_iam_auth": true, "azure_redis_ad_token": "True", "host": "redis", "password": "dropped"}),
    &[("REDIS_GCP_SERVICE_ACCOUNT", "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "gcp_iam", "service_account": "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"}}))),
)]
#[case::gcp_ca_certs_replace_the_configured_ones_under_ssl(
    json!({"gcp_service_account": "account", "gcp_ssl_ca_certs": "/gcp-ca.pem", "host": "redis", "ssl": true, "ssl_ca_certs": "/configured-ca.pem"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": "/gcp-ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "gcp_iam", "service_account": "account"}}))),
)]
#[case::azure_beats_aws_and_takes_the_environment_username(
    json!({"aws_iam_auth": true, "azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis", "username": "dropped"}),
    &[("REDIS_USERNAME", "environment-user")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "azure_ad", "username": "environment-user"}}))),
)]
#[case::azure_without_a_username_sends_only_the_token(
    json!({"azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "azure_ad", "username": null}}))),
)]
#[case::azure_needs_the_literal_true(
    json!({"azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": "1", "azure_tenant_id": "tenant", "host": "redis"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::aws_iam_over_tls_with_the_default_region(
    json!({"aws_iam_auth": "yes", "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "password": "dropped", "ssl": true}),
    &[("AWS_DEFAULT_REGION", "us-east-1")],
    Ok(expected_view(json!({"mode": "host", "host": "cache-1", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "aws_iam", "user_name": "iam-user", "cache_name": "cache-1", "region": "us-east-1", "serverless": false}}))),
)]
#[case::aws_region_beats_the_default_region(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "ssl": true}),
    &[("AWS_DEFAULT_REGION", "us-east-1"), ("AWS_REGION", "eu-central-1")],
    Ok(expected_view(json!({"mode": "host", "host": "cache-1", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "aws_iam", "user_name": "iam-user", "cache_name": "cache-1", "region": "eu-central-1", "serverless": false}}))),
)]
#[case::aws_iam_serverless_over_a_rediss_url_drops_its_userinfo(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_region": "eu-west-1", "aws_iam_serverless": true, "aws_iam_user_name": "iam-user", "url": "rediss://user:pass@cache-1:6379"}),
    &[],
    Ok(expected_view(json!({"mode": "url", "url": "rediss://cache-1:6379", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "aws_iam", "user_name": "iam-user", "cache_name": "cache-1", "region": "eu-west-1", "serverless": true}}))),
)]
#[case::aws_iam_without_tls_fails(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_region": "us-east-1", "aws_iam_user_name": "iam-user", "host": "cache-1"}),
    &[],
    Err("AWS ElastiCache IAM Redis authentication requires TLS"),
)]
#[case::aws_iam_without_a_region_fails(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "ssl": true}),
    &[],
    Err("AWS ElastiCache IAM Redis authentication requires: aws_iam_region"),
)]
#[case::tls_options_from_settings(
    json!({"host": "redis", "ssl": true, "ssl_ca_certs": "/ca.pem", "ssl_cert_reqs": "none", "ssl_check_hostname": true}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": "/ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": false, "check_hostname": true}}))),
)]
#[case::tls_options_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_SSL", "true"), ("REDIS_SSL_CA_CERTS", "/env-ca.pem"), ("REDIS_SSL_CERT_REQS", "optional")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": "/env-ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::connect_timeout_is_separate_from_the_socket_timeout(
    json!({"host": "redis", "socket_connect_timeout": 0.25, "socket_timeout": 2.5}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "socket_timeout": 2.5, "socket_connect_timeout": 0.25}))),
)]
#[case::environment_references_resolve(
    json!({"host": "os.environ/COORD_HOST", "password": "os.environ/COORD_PASSWORD"}),
    &[("COORD_HOST", "referenced"), ("COORD_PASSWORD", "referenced-secret")],
    Ok(expected_view(json!({"mode": "host", "host": "referenced", "port": 6379, "db": 0, "auth": {"kind": "static", "username": null, "password": "referenced-secret"}}))),
)]
#[case::no_target_fails(
    json!({"port": 6379}),
    &[],
    Err("either host or url must be specified for redis"),
)]
#[case::client_certificates(
    json!({"host": "redis", "ssl": true, "ssl_certfile": "/client.pem", "ssl_keyfile": "/client.key"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": "/client.pem", "keyfile": "/client.key", "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::a_rediss_url_takes_tls_options_from_its_query(
    json!({"url": "rediss://cache:6380/0?ssl_cert_reqs=none&ssl_ca_certs=/ca.pem&ssl_check_hostname=yes"}),
    &[],
    Ok(expected_view(json!({"mode": "url", "url": "rediss://cache:6380/0?ssl_cert_reqs=none&ssl_ca_certs=/ca.pem&ssl_check_hostname=yes", "tls": {"ca_certs": "/ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": false, "check_hostname": true}}))),
)]
#[case::a_url_query_setting_without_a_redis_rs_equivalent(
    json!({"url": "redis://cache:6379/0?socket_timeout=2"}),
    &[],
    Ok(expected_view(json!({"mode": "url", "url": "redis://cache:6379/0?socket_timeout=2"}))),
)]
#[case::tls_settings_outside_a_url_never_fail_it(
    json!({"db": "99999999999999999999", "ssl_cert_reqs": "strict", "url": "rediss://cache:6380"}),
    &[],
    Ok(expected_view(json!({"mode": "url", "url": "rediss://cache:6380", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::db_is_not_read_for_a_cluster(
    json!({"db": 99_999_999_999_i64, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Err("db must be 0 in cluster mode"),
)]
#[case::cluster_db_true(
    json!({"db": true, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Err("db must be 0 in cluster mode"),
)]
#[case::cluster_db_false(
    json!({"db": false, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::cluster_db_fraction(
    json!({"db": 0.5, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Err("db must be 0 in cluster mode"),
)]
#[case::cluster_db_float_zero(
    json!({"db": 0.0, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::cluster_db_nonempty_list(
    json!({"db": [1], "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Err("db must be 0 in cluster mode"),
)]
#[case::cluster_db_empty_list(
    json!({"db": [], "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::ssl_ciphers(
    json!({"host": "redis", "ssl": true, "ssl_ciphers": "HIGH"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::retry_on_timeout(
    json!({"host": "redis", "retry_on_timeout": true}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::malformed_port(
    json!({"host": "redis", "port": "abc"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::malformed_db(
    json!({"db": "first", "host": "redis"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::socket_timeout_not_a_number(
    json!({"host": "redis", "socket_timeout": "slow"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::malformed_port_from_the_environment(
    json!({}),
    &[("REDIS_HOST", "redis"), ("REDIS_PORT", "abc")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::protocol_3(
    json!({"host": "redis", "protocol": 3}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::retry_on_error_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_RETRY_ON_ERROR", "true")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::unix_socket_path(
    json!({"host": "redis", "unix_socket_path": "/tmp/redis.sock"}),
    &[],
    Ok(expected_view(json!({"mode": "unix", "path": "/tmp/redis.sock"}))),
)]
#[case::unix_socket_path_keeps_tls_settings_out_of_the_target(
    json!({"host": "redis", "ssl": true, "ssl_ca_certs": "/ca.pem", "unix_socket_path": "/tmp/redis.sock"}),
    &[],
    Ok(expected_view(json!({"mode": "unix", "path": "/tmp/redis.sock"}))),
)]
#[case::ssl_ca_data(
    json!({"host": "redis", "ssl": true, "ssl_ca_data": "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----", "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::ssl_ca_path_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_SSL", "true"), ("REDIS_SSL_CA_PATH", "/etc/ssl/private-cas")],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": "/etc/ssl/private-cas", "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::health_check_interval_is_whole_seconds(
    json!({"health_check_interval": "7", "host": "redis"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "health_check_interval": 7.0}))),
)]
#[case::health_check_interval_zero_is_off(
    json!({"health_check_interval": 0, "host": "redis"}),
    &[],
    Ok(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::health_check_interval_zero_keeps_a_cluster_off(
    json!({"health_check_interval": 0, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Ok(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"]}))),
)]
#[case::a_cluster_rejects_a_nonzero_db(
    json!({"db": 3, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    Err("db must be 0 in cluster mode"),
)]
#[case::unix_socket_path_without_a_host_fails_like_python(
    json!({"unix_socket_path": "/tmp/redis.sock"}),
    &[],
    Err("either host or url must be specified for redis"),
)]
fn resolves_like_python_redis(
    #[case] kwargs: serde_json::Value,
    #[case] environment: &[(&str, &str)],
    #[case] expected: Result<serde_json::Value, &str>,
) {
    let environment: HashMap<String, String> = environment
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();

    let outcome = resolve_kwargs(kwargs, &environment)
        .map(|resolved| view(&resolved))
        .map_err(|error| error.to_string());

    assert_eq!(outcome, expected.map_err(str::to_owned));
}

fn snapshot(values: &[(&str, &str)]) -> SecretSnapshot {
    let values: HashMap<&str, &str> = values.iter().copied().collect();
    SecretSnapshot::read(
        ConfiguredRedis::environment_only().secret_names(),
        &|name: &str| values.get(name).map(|value| (*value).to_owned()),
    )
}

#[rstest]
fn secret_manager_values_reach_the_resolver_through_the_snapshot() {
    let settings = redis_cache_kwargs(json!({"host": "os.environ/COORD_HOST"}));
    let secrets = SecretSnapshot::read(settings.secret_names(), &|name: &str| match name {
        "COORD_HOST" => Some("from-the-manager".to_owned()),
        "REDIS_PASSWORD" => Some("manager-secret".to_owned()),
        _ => None,
    });

    let resolved = settings.resolve(&secrets, &|_: &str| None).unwrap();

    assert!(
        matches!(&resolved.target, Target::Standalone(host) if host.host == "from-the-manager")
    );
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
    let settings = redis_cache_kwargs(json!({"host": "redis", "azure_redis_ad_token": "true"}));
    let environment: HashMap<&str, &str> = environment.iter().copied().collect();

    let resolved = settings
        .resolve(&snapshot(secrets), &|name: &str| {
            environment.get(name).map(|value| (*value).to_owned())
        })
        .unwrap();

    assert!(
        matches!(&resolved.auth, Auth::AzureAd(azure) if azure.username.as_deref() == expected),
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

    let Target::Standalone(host) = resolved.target else {
        panic!("expected a standalone target, got {:?}", resolved.target);
    };
    assert_eq!(host.tls.is_some(), tls);
}
