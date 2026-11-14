use std::collections::{BTreeMap, HashMap};

use litellm_config::{CacheSettings, Config, Value};
use litellm_redis_resolve::Expected;
use litellm_redis_types::{
    Auth, AwsIamAuth, AzureAdAuth, Node, ProviderHandle, Resolution, StaticAuth, Target, TlsOptions,
};
use litellm_secret_value::SecretValue;
use rstest::rstest;
use serde_json::json;

use super::{configured::ConfiguredRedis, secrets::SecretSnapshot};
use crate::Error;

fn resolution(
    configured: &ConfiguredRedis,
    environment: &HashMap<String, String>,
) -> Result<Resolution, Error> {
    let lookup = |name: &str| environment.get(name).cloned();
    configured.resolve(
        &SecretSnapshot::read(configured.secret_names(), &lookup),
        &lookup,
    )
}

const PROVIDER: ProviderHandle = ProviderHandle::new(7);

fn redis_cache_kwargs(kwargs: serde_json::Value) -> ConfiguredRedis {
    let has_provider = kwargs
        .get("credential_provider")
        .is_some_and(|provider| !provider.is_null());
    let configured = ConfiguredRedis::redis_cache_kwargs(
        serde_json::from_value::<BTreeMap<String, Value>>(kwargs).unwrap(),
    );
    match has_provider {
        true => configured.with_credential_provider(PROVIDER),
        false => configured,
    }
}

fn coordination_resolution(yaml: &str, environment: &[(&str, &str)]) -> Result<Resolution, Error> {
    let environment: HashMap<String, String> = environment
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    let config = Config::from_yaml(yaml).unwrap();
    let configured = config
        .general_settings
        .coordination_redis
        .unwrap_or_default();
    let configured = ConfiguredRedis::coordination(&configured)?;
    resolution(&configured, &environment)
}

fn resolve(yaml: &str, environment: &[(&str, &str)]) -> Result<Resolution, Error> {
    coordination_resolution(yaml, environment)
}

fn resolve_cache(yaml: &str) -> Result<Resolution, Error> {
    resolve_cache_in(yaml, &[])
}

fn resolve_cache_in(yaml: &str, environment: &[(&str, &str)]) -> Result<Resolution, Error> {
    let lookup = |name: &str| {
        environment
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_owned())
    };
    let config = Config::from_yaml(yaml).unwrap();
    let configured = match config.litellm_settings.cache_params.unwrap() {
        CacheSettings::Redis(settings) => ConfiguredRedis::cache_params(&settings),
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

#[rstest]
#[case::kwarg(
    "general_settings:\n  coordination_redis:\n    host: redis\n    ssl: true\n    ssl_validate_ocsp: true",
    &[],
    "ssl_validate_ocsp"
)]
#[case::stapled_from_environment(
    "model_list: []",
    &[("REDIS_HOST", "redis"), ("REDIS_SSL", "true"), ("REDIS_SSL_VALIDATE_OCSP_STAPLED", "yes")],
    "REDIS_SSL_VALIDATE_OCSP_STAPLED"
)]
#[case::rediss_url_query(
    "general_settings:\n  coordination_redis:\n    url: rediss://cache:6380/0?ssl_validate_ocsp=true",
    &[],
    "url ?ssl_validate_ocsp"
)]
fn an_ocsp_check_fails_instead_of_connecting_without_it(
    #[case] yaml: &str,
    #[case] environment: &[(&str, &str)],
    #[case] setting: &str,
) {
    let error = coordination_resolution(yaml, environment).unwrap_err();

    assert!(
        matches!(&error, Error::Redis(litellm_redis_resolve::Error::Unenforceable { setting: named, .. }) if named == setting),
        "{error:?}"
    );
}

#[rstest]
#[case::kwarg("    host: redis\n    ssl: true\n    ssl_validate_ocsp: false")]
#[case::rediss_url_query("    url: rediss://cache:6380/0?ssl_validate_ocsp=false")]
fn an_ocsp_flag_set_to_false_asks_for_nothing(#[case] block: &str) {
    let yaml = format!("general_settings:\n  coordination_redis:\n{block}");

    let resolution = coordination_resolution(&yaml, &[]).unwrap();

    assert!(resolution.ignored.is_empty(), "{:?}", resolution.ignored);
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
        matches!(
            error,
            Error::Redis(litellm_redis_resolve::Error::ProviderWithoutHandle)
        ),
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
    Some(TlsOptions { ca_certs: None, ca_data: None, ca_path: None, certfile: None, keyfile: None, key_password: None, verify_certificate: true, check_hostname: false })
)]
#[case::every_option(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    ssl_certfile: /client.pem\n    ssl_keyfile: /client.key\n    ssl_password: key-secret\n    ssl_cert_reqs: none\n    ssl_check_hostname: true",
    &[],
    Some(TlsOptions { ca_certs: Some("/ca.pem".into()), ca_data: None, ca_path: None, certfile: Some("/client.pem".into()), keyfile: Some("/client.key".into()), key_password: Some(SecretValue::new("key-secret")), verify_certificate: false, check_hostname: true })
)]
#[case::options_from_the_environment(
    "    ssl: true",
    &[("REDIS_SSL_CA_CERTS", "/env-ca.pem"), ("REDIS_SSL_CERT_REQS", "optional"), ("REDIS_SSL_PASSWORD", "env-key-secret")],
    Some(TlsOptions { ca_certs: Some("/env-ca.pem".into()), ca_data: None, ca_path: None, certfile: None, keyfile: None, key_password: Some(SecretValue::new("env-key-secret")), verify_certificate: true, check_hostname: false })
)]
#[case::gcp_ca_certs_replace_the_configured_ones(
    "    ssl: true\n    ssl_ca_certs: /ca.pem\n    gcp_service_account: account\n    gcp_ssl_ca_certs: /gcp-ca.pem",
    &[],
    Some(TlsOptions { ca_certs: Some("/gcp-ca.pem".into()), ca_data: None, ca_path: None, certfile: None, keyfile: None, key_password: None, verify_certificate: true, check_hostname: false })
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
        Error::Redis(litellm_redis_resolve::Error::Invalid { setting, expected: Expected::CertRequirement }) if setting == "ssl_cert_reqs"
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

    assert!(
        matches!(
            error,
            Error::Redis(litellm_redis_resolve::Error::NoSentinelService)
        ),
        "{error:?}"
    );
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
        matches!(&error, Error::Redis(litellm_redis_resolve::Error::SentinelWithHost { setting }) if setting == source),
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
        matches!(&error, Error::Redis(litellm_redis_resolve::Error::Invalid { setting, expected: Expected::SentinelNodes }) if setting == source),
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
        matches!(&error, Error::Redis(litellm_redis_resolve::Error::Invalid { setting, .. }) if setting == source),
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

    assert!(
        matches!(error, Error::Redis(litellm_redis_resolve::Error::NoTarget)),
        "{error:?}"
    );
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
        Error::Redis(litellm_redis_resolve::Error::Invalid { setting, expected: Expected::Seconds }) if setting == "socket_timeout"
    ));
}

#[rstest]
fn debug_redacts_redis_secrets_and_url() {
    let config = Config::from_yaml(
        "general_settings:\n  coordination_redis:\n    url: redis://user:password@redis:6379\n    username: credential-user\n    password: credential-password\n    sentinel_kwargs: {password: credential-passthrough}",
    )
    .unwrap();
    let settings = config.general_settings.coordination_redis.unwrap();
    let configured = ConfiguredRedis::coordination(&settings).unwrap();
    let resolved = resolution(&configured, &HashMap::new()).unwrap();
    let debug = format!("{settings:?} {configured:?} {resolved:?}");

    assert!(!debug.contains("redis://user:password"));
    assert!(!debug.contains("credential-user"));
    assert!(!debug.contains("credential-password"));
    assert!(!debug.contains("credential-passthrough"));
    assert!(debug.contains("redacted") || debug.contains("REDACTED"));
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

#[rstest]
#[case::empty_cluster_nodes_use_the_environment(
    json!({"startup_nodes": []}),
    &[("REDIS_CLUSTER_NODES", r#"[{"host": "node-a", "port": 7000}]"#)],
    json!({"startup_nodes": "[{\"host\": \"node-a\", \"port\": 7000}]"}),
    &[("startup_nodes", "REDIS_CLUSTER_NODES")],
)]
#[case::empty_sentinel_nodes_use_the_environment(
    json!({"sentinel_nodes": [], "service_name": "primary"}),
    &[("REDIS_SENTINEL_NODES", r#"[["sentinel-a", 26379]]"#)],
    json!({"sentinel_nodes": "[[\"sentinel-a\", 26379]]", "service_name": "primary"}),
    &[("sentinel_nodes", "REDIS_SENTINEL_NODES")],
)]
#[case::empty_service_name_uses_the_environment(
    json!({"sentinel_nodes": [["sentinel-a", 26379]], "service_name": ""}),
    &[("REDIS_SERVICE_NAME", "primary")],
    json!({"sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"}),
    &[("service_name", "REDIS_SERVICE_NAME")],
)]
#[case::environment_host_port_and_password(
    json!({}),
    &[("REDIS_HOST", "environment"), ("REDIS_PASSWORD", "environment-secret"), ("REDIS_PORT", "6390"), ("REDIS_SSL", "True")],
    json!({"port": "6390", "ssl": true, "host": "environment", "password": "environment-secret"}),
    &[("host", "REDIS_HOST"), ("password", "REDIS_PASSWORD"), ("port", "REDIS_PORT"), ("ssl", "REDIS_SSL")],
)]
#[case::padded_environment_numbers_parse_like_python_int(
    json!({}),
    &[("REDIS_HOST", "redis"), ("REDIS_PORT", "6_380"), ("REDIS_DB", "3\n")],
    json!({"port": "6_380", "host": "redis", "db": "3\n"}),
    &[("db", "REDIS_DB"), ("host", "REDIS_HOST"), ("port", "REDIS_PORT")],
)]
#[case::socket_timeout_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_SOCKET_TIMEOUT", "0.75")],
    json!({"socket_timeout": "0.75", "host": "redis"}),
    &[("socket_timeout", "REDIS_SOCKET_TIMEOUT")],
)]
#[case::environment_url(
    json!({}),
    &[("REDIS_URL", "redis://environment:6379/1")],
    json!({"url": "redis://environment:6379/1"}),
    &[("url", "REDIS_URL")],
)]
#[case::environment_url_beats_the_environment_host(
    json!({}),
    &[("REDIS_HOST", "ignored"), ("REDIS_URL", "redis://from-url:6379")],
    json!({"host": "ignored", "url": "redis://from-url:6379"}),
    &[("host", "REDIS_HOST"), ("url", "REDIS_URL")],
)]
#[case::an_explicit_host_drops_the_environment_url(
    json!({"host": "configured"}),
    &[("REDIS_URL", "redis://environment:6379")],
    json!({"host": "configured"}),
    &[],
)]
#[case::environment_cluster_nodes(
    json!({}),
    &[("REDIS_CLUSTER_NODES", r#"[{"host": "node-a", "port": "7000"}, {"host": "node-b", "port": 7001}]"#)],
    json!({"startup_nodes": "[{\"host\": \"node-a\", \"port\": \"7000\"}, {\"host\": \"node-b\", \"port\": 7001}]"}),
    &[("startup_nodes", "REDIS_CLUSTER_NODES")],
)]
#[case::sentinel_from_the_environment(
    json!({}),
    &[("REDIS_SENTINEL_NODES", r#"[["sentinel-a", 26379]]"#), ("REDIS_SENTINEL_PASSWORD", "sentinel-secret"), ("REDIS_SERVICE_NAME", "primary")],
    json!({"sentinel_nodes": "[[\"sentinel-a\", 26379]]", "sentinel_password": "sentinel-secret", "service_name": "primary"}),
    &[("sentinel_nodes", "REDIS_SENTINEL_NODES"), ("sentinel_password", "REDIS_SENTINEL_PASSWORD"), ("service_name", "REDIS_SERVICE_NAME")],
)]
#[case::an_explicit_credential_provider_beats_cloud_auth(
    json!({"aws_iam_auth": true, "credential_provider": true, "gcp_service_account": "account", "host": "redis", "password": "dropped"}),
    &[("REDIS_AZURE_AD_TOKEN", "true")],
    json!({"aws_iam_auth": true, "credential_provider": true, "gcp_service_account": "account", "host": "redis", "password": "dropped"}),
    &[],
)]
#[case::gcp_beats_azure_and_aws(
    json!({"aws_iam_auth": true, "azure_redis_ad_token": "True", "host": "redis", "password": "dropped"}),
    &[("REDIS_GCP_SERVICE_ACCOUNT", "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com")],
    json!({"gcp_service_account": "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com", "aws_iam_auth": true, "azure_redis_ad_token": "True", "host": "redis", "password": "dropped"}),
    &[("gcp_service_account", "REDIS_GCP_SERVICE_ACCOUNT")],
)]
#[case::azure_beats_aws_and_takes_the_environment_username(
    json!({"aws_iam_auth": true, "azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis", "username": "dropped"}),
    &[("REDIS_USERNAME", "environment-user")],
    json!({"aws_iam_auth": true, "azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis", "username": "dropped"}),
    &[],
)]
#[case::an_empty_environment_username_sends_only_the_token(
    json!({"azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis"}),
    &[("REDIS_USERNAME", "")],
    json!({"azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis", "username": ""}),
    &[("username", "REDIS_USERNAME")],
)]
#[case::aws_iam_over_tls_with_the_default_region(
    json!({"aws_iam_auth": "yes", "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "password": "dropped", "ssl": true}),
    &[("AWS_DEFAULT_REGION", "us-east-1")],
    json!({"aws_iam_auth": "yes", "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "password": "dropped", "ssl": true, "aws_iam_region": "us-east-1"}),
    &[("aws_iam_region", "AWS_DEFAULT_REGION")],
)]
#[case::aws_region_beats_the_default_region(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "ssl": true}),
    &[("AWS_DEFAULT_REGION", "us-east-1"), ("AWS_REGION", "eu-central-1")],
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "ssl": true, "aws_iam_region": "eu-central-1"}),
    &[("aws_iam_region", "AWS_REGION")],
)]
#[case::a_null_cert_requirement_skips_verification(
    json!({"host": "redis", "ssl": true, "ssl_cert_reqs": null}),
    &[("REDIS_SSL_CERT_REQS", "required")],
    json!({"host": "redis", "ssl": true, "ssl_cert_reqs": null}),
    &[],
)]
#[case::a_cert_requirement_spelled_like_python_none_fails(
    json!({"host": "redis"}),
    &[("REDIS_SSL", "true"), ("REDIS_SSL_CERT_REQS", "None")],
    json!({"ssl": true, "ssl_cert_reqs": "None", "host": "redis"}),
    &[("ssl", "REDIS_SSL"), ("ssl_cert_reqs", "REDIS_SSL_CERT_REQS")],
)]
#[case::tls_options_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_SSL", "true"), ("REDIS_SSL_CA_CERTS", "/env-ca.pem"), ("REDIS_SSL_CERT_REQS", "optional")],
    json!({"ssl_ca_certs": "/env-ca.pem", "ssl": true, "ssl_cert_reqs": "optional", "host": "redis"}),
    &[("ssl", "REDIS_SSL"), ("ssl_ca_certs", "REDIS_SSL_CA_CERTS"), ("ssl_cert_reqs", "REDIS_SSL_CERT_REQS")],
)]
#[case::environment_references_resolve(
    json!({"host": "os.environ/COORD_HOST", "password": "os.environ/COORD_PASSWORD"}),
    &[("COORD_HOST", "referenced"), ("COORD_PASSWORD", "referenced-secret")],
    json!({"host": "referenced", "password": "referenced-secret"}),
    &[],
)]
#[case::malformed_port_from_the_environment(
    json!({}),
    &[("REDIS_HOST", "redis"), ("REDIS_PORT", "abc")],
    json!({"port": "abc", "host": "redis"}),
    &[("host", "REDIS_HOST"), ("port", "REDIS_PORT")],
)]
#[case::retry_on_error_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_RETRY_ON_ERROR", "true")],
    json!({"retry_on_error": true, "host": "redis"}),
    &[("retry_on_error", "REDIS_RETRY_ON_ERROR")],
)]
#[case::ssl_ca_path_from_the_environment(
    json!({"host": "redis"}),
    &[("REDIS_SSL", "true"), ("REDIS_SSL_CA_PATH", "/etc/ssl/private-cas")],
    json!({"ssl": true, "ssl_ca_path": "/etc/ssl/private-cas", "host": "redis"}),
    &[("ssl", "REDIS_SSL"), ("ssl_ca_path", "REDIS_SSL_CA_PATH")],
)]
fn merges_like_python_redis(
    #[case] kwargs: serde_json::Value,
    #[case] environment: &[(&str, &str)],
    #[case] merged: serde_json::Value,
    #[case] sources: &[(&str, &str)],
) {
    let configured = redis_cache_kwargs(kwargs);
    let lookup = |name: &str| {
        environment
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_owned())
    };

    let kwargs = configured
        .merged(&SecretSnapshot::read(configured.secret_names(), &lookup))
        .unwrap();

    let values: serde_json::Map<String, serde_json::Value> = kwargs
        .iter()
        .map(|(kwarg, entry)| (kwarg.clone(), entry.value.clone()))
        .collect();
    let renamed: Vec<(&str, &str)> = kwargs
        .iter()
        .filter(|(kwarg, entry)| *kwarg != &entry.source)
        .map(|(kwarg, entry)| (kwarg.as_str(), entry.source.as_str()))
        .collect();
    assert_eq!(serde_json::Value::Object(values), merged);
    assert_eq!(renamed, sources);
}
