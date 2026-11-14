use std::collections::BTreeMap;

use litellm_redis_resolve::{Entry, Error, Expected, Kwargs, PythonException, interpret, kwargs};
use litellm_redis_types::{
    Auth, HealthCheck, Node, ProviderHandle, Resolution, StaticAuth, Target, TlsOptions,
};
use rstest::rstest;
use serde_json::json;

use Outcome::{Fails, Refuses, Resolves};
use PythonException::{RedisClusterException, TypeError, ValueError};

const PROVIDER: ProviderHandle = ProviderHandle::new(7);

fn merged(fields: serde_json::Value, sources: &[(&str, &str)]) -> Kwargs {
    let fields = serde_json::from_value::<BTreeMap<String, serde_json::Value>>(fields).unwrap();
    kwargs(fields)
        .into_iter()
        .map(|(kwarg, entry)| {
            let source = sources
                .iter()
                .find(|(named, _)| *named == kwarg)
                .map_or(entry.source, |(_, source)| (*source).to_owned());
            (kwarg, Entry { source, ..entry })
        })
        .collect()
}

// a host hands its credential_provider object over as a handle, so the cases do the same
fn interpret_merged(
    fields: serde_json::Value,
    sources: &[(&str, &str)],
    redis_username: Option<&str>,
) -> Result<Resolution, Error> {
    let has_provider = fields
        .get("credential_provider")
        .is_some_and(|provider| !provider.is_null());
    interpret(
        &merged(fields, sources),
        has_provider.then_some(PROVIDER),
        redis_username,
    )
}

fn interpret_kwargs(fields: serde_json::Value) -> Result<Resolution, Error> {
    interpret_merged(fields, &[], None)
}

/// What a contract case expects of the resolver, next to what its Python mirror expects of redis-py.
enum Outcome {
    Resolves(serde_json::Value),
    /// Rust refuses a config Python connects with, a disagreement on purpose.
    Refuses(&'static str),
    /// redis-py 5.3.1 already raises on the config, building the client or its first connection, so failing
    /// here breaks no deployment that works today. The class is what it raises building the client, and `None`
    /// when it only raises once a command connects.
    Fails(&'static str, Option<PythonException>),
}

#[rstest]
fn a_provider_handle_beats_every_cloud_auth_and_static_credentials() {
    let resolved = interpret_kwargs(json!({
        "host": "redis", "password": "pass", "credential_provider": "host-object",
        "gcp_service_account": "account", "aws_iam_auth": true, "azure_redis_ad_token": "true",
    }))
    .unwrap();

    assert_eq!(resolved.auth, Auth::Provider(PROVIDER));
}

#[rstest]
fn a_rediss_url_reads_ssl_password_from_its_query() {
    let resolved =
        interpret_kwargs(json!({"url": "rediss://cache:6380/0?ssl_password=query-key-secret"}))
            .unwrap();

    let Target::Url(parsed) = resolved.target else {
        panic!("expected a url target, got {:?}", resolved.target);
    };
    let key_password = parsed.tls.and_then(|tls| tls.key_password);
    assert_eq!(expose(key_password.as_ref()), Some("query-key-secret"));
    assert!(resolved.ignored.is_empty(), "{:?}", resolved.ignored);
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
    let resolved = interpret_kwargs(json!({"url": url})).unwrap();

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
    let error = interpret_kwargs(json!({"url": "redis://cache:6379?db=first"})).unwrap_err();

    assert!(matches!(error, Error::UrlQuery { setting } if setting == "url ?db"));
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
    let resolved =
        interpret_kwargs(json!({"host": "redis", "ssl": true, "ssl_min_version": version}))
            .unwrap();

    assert_eq!(
        !resolved.ignored.is_empty(),
        warns,
        "{:?}",
        resolved.ignored
    );
}

#[rstest]
#[case::unset(json!({"host": "redis"}), true)]
#[case::empty_like_unset(json!({"host": "redis", "socket_keepalive": ""}), true)]
#[case::enabled(json!({"host": "redis", "socket_keepalive": "yes"}), true)]
#[case::disabled(json!({"host": "redis", "socket_keepalive": false}), false)]
#[case::disabled_as_text(json!({"host": "redis", "socket_keepalive": "0"}), false)]
#[case::disabled_on_a_cluster(json!({"startup_nodes": [{"host": "node-a", "port": 7000}], "socket_keepalive": "false"}), false)]
fn keepalive_stays_on_unless_socket_keepalive_turns_it_off(
    #[case] kwargs: serde_json::Value,
    #[case] keepalive: bool,
) {
    let resolved = interpret_kwargs(kwargs).unwrap();

    assert_eq!(resolved.keepalive, keepalive);
    assert!(resolved.ignored.is_empty(), "{:?}", resolved.ignored);
}

#[rstest]
fn a_socket_keepalive_no_python_surface_agrees_on_is_invalid() {
    let error = interpret_kwargs(json!({"host": "redis", "socket_keepalive": "on"})).unwrap_err();

    assert!(
        matches!(&error, Error::Invalid { setting, expected: Expected::Boolean } if setting == "socket_keepalive"),
        "{error:?}"
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
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::empty_cluster_nodes_use_the_environment(
    json!({"startup_nodes": "[{\"host\": \"node-a\", \"port\": 7000}]"}),
    &[("startup_nodes", "REDIS_CLUSTER_NODES")],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::empty_sentinel_nodes_use_the_environment(
    json!({"sentinel_nodes": "[[\"sentinel-a\", 26379]]", "service_name": "primary"}),
    &[("sentinel_nodes", "REDIS_SENTINEL_NODES")],
    None,
    Resolves(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379"], "service_name": "primary", "sentinel_password": null, "db": 0}))),
)]
#[case::empty_service_name_uses_the_environment(
    json!({"sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"}),
    &[("service_name", "REDIS_SERVICE_NAME")],
    None,
    Resolves(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379"], "service_name": "primary", "sentinel_password": null, "db": 0}))),
)]
#[case::plaintext_ignores_tls_requirements(
    json!({"host": "redis", "ssl_cert_reqs": "strict"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::an_ambiguous_flag_is_refused(
    json!({"host": "redis", "ssl": "on"}),
    &[],
    None,
    Refuses("invalid Redis setting ssl: expected a boolean"),
)]
#[case::empty_sentinel_nodes_without_the_environment_name_no_target(
    json!({"sentinel_nodes": [], "service_name": "primary"}),
    &[],
    None,
    Fails("either host or url must be specified for redis", Some(ValueError)),
)]
#[case::ssl_ca_path_is_unsupported(
    json!({"host": "redis", "ssl": true, "ssl_ca_path": "/ca"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": "/ca", "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::ssl_validate_ocsp_is_unsupported(
    json!({"host": "redis", "ssl": true, "ssl_validate_ocsp": true}),
    &[],
    None,
    Fails("ssl_validate_ocsp asks for OCSP revocation checks, which rustls cannot perform, so Redis is not connected without them", Some(TypeError)),
)]
#[case::ssl_include_verify_flags_is_unsupported(
    json!({"host": "redis", "ssl": true, "ssl_include_verify_flags": ["VERIFY_CRL_CHECK_LEAF"]}),
    &[],
    None,
    Fails("ssl_include_verify_flags asks for X.509 verify flags such as CRL checking, which rustls cannot perform, so Redis is not connected without them", Some(TypeError)),
)]
#[case::ssl_password_is_unsupported(
    json!({"host": "redis", "ssl": true, "ssl_certfile": "/client.pem", "ssl_password": "key-secret"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": "/client.pem", "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::host_with_every_basic_setting(
    json!({"db": 2, "host": "redis", "password": "pass", "port": 6380, "socket_timeout": 1.5, "ssl": true, "username": "user"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6380, "db": 2, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "static", "username": "user", "password": "pass"}, "socket_timeout": 1.5, "socket_connect_timeout": 1.5}))),
)]
#[case::environment_host_port_and_password(
    json!({"port": "6390", "ssl": true, "host": "environment", "password": "environment-secret"}),
    &[("host", "REDIS_HOST"), ("password", "REDIS_PASSWORD"), ("port", "REDIS_PORT"), ("ssl", "REDIS_SSL")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "environment", "port": 6390, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "static", "username": null, "password": "environment-secret"}}))),
)]
#[case::string_numbers_are_coerced(
    json!({"db": "3", "host": "redis", "port": "6380", "socket_timeout": "2.5"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6380, "db": 3, "socket_timeout": 2.5, "socket_connect_timeout": 2.5}))),
)]
#[case::padded_and_underscored_numbers_parse_like_python_int_and_float(
    json!({"host": "redis", "port": "6380\n", "db": " 2 ", "socket_timeout": "1_0.5\n"}),
    &[],
    None,
    Resolves(expected_view(json!({"db": 2, "host": "redis", "mode": "host", "port": 6380, "socket_timeout": 10.5, "socket_connect_timeout": 10.5}))),
)]
#[case::padded_environment_numbers_parse_like_python_int(
    json!({"port": "6_380", "host": "redis", "db": "3\n"}),
    &[("db", "REDIS_DB"), ("host", "REDIS_HOST"), ("port", "REDIS_PORT")],
    None,
    Resolves(expected_view(json!({"db": 3, "host": "redis", "mode": "host", "port": 6380}))),
)]
#[case::socket_timeout_from_the_environment(
    json!({"socket_timeout": "0.75", "host": "redis"}),
    &[("socket_timeout", "REDIS_SOCKET_TIMEOUT")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "socket_timeout": 0.75, "socket_connect_timeout": 0.75}))),
)]
#[case::environment_url(
    json!({"url": "redis://environment:6379/1"}),
    &[("url", "REDIS_URL")],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "redis://environment:6379/1"}))),
)]
#[case::environment_url_beats_the_environment_host(
    json!({"host": "ignored", "url": "redis://from-url:6379"}),
    &[("host", "REDIS_HOST"), ("url", "REDIS_URL")],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "redis://from-url:6379"}))),
)]
#[case::an_explicit_host_drops_the_environment_url(
    json!({"host": "configured"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "configured", "port": 6379, "db": 0}))),
)]
#[case::a_url_drops_host_db_and_password_and_keeps_the_username(
    json!({"db": 4, "host": "dropped", "password": "dropped", "url": "rediss://cache:6380", "username": "user"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "rediss://cache:6380", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "static", "username": "user", "password": null}}))),
)]
#[case::a_url_ignores_tls_settings_outside_the_url(
    json!({"ssl_ca_certs": "/ca.pem", "ssl_cert_reqs": "none", "ssl_check_hostname": true, "url": "rediss://cache:6380"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "rediss://cache:6380", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::startup_nodes_beat_a_url(
    json!({"password": "cluster-secret", "startup_nodes": [{"host": "node-a", "port": 7000}], "url": "redis://single:6379"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "auth": {"kind": "static", "username": null, "password": "cluster-secret"}, "health_check_interval": 25.0}))),
)]
#[case::environment_cluster_nodes(
    json!({"startup_nodes": "[{\"host\": \"node-a\", \"port\": \"7000\"}, {\"host\": \"node-b\", \"port\": 7001}]"}),
    &[("startup_nodes", "REDIS_CLUSTER_NODES")],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000", "node-b:7001"], "health_check_interval": 25.0}))),
)]
#[case::cluster_tls_options(
    json!({"ssl": "true", "ssl_ca_certs": "/ca.pem", "ssl_check_hostname": true, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "tls": {"ca_certs": "/ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": true}, "health_check_interval": 25.0}))),
)]
#[case::sentinel_from_settings(
    json!({"password": "data-secret", "sentinel_nodes": [["sentinel-a", 26379], ["sentinel-b", 26380]], "sentinel_password": "sentinel-secret", "service_name": "primary"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379", "sentinel-b:26380"], "service_name": "primary", "sentinel_password": "sentinel-secret", "db": 0, "auth": {"kind": "static", "username": null, "password": "data-secret"}}))),
)]
#[case::sentinel_from_the_environment(
    json!({"sentinel_nodes": "[[\"sentinel-a\", 26379]]", "sentinel_password": "sentinel-secret", "service_name": "primary"}),
    &[("sentinel_nodes", "REDIS_SENTINEL_NODES"), ("sentinel_password", "REDIS_SENTINEL_PASSWORD"), ("service_name", "REDIS_SERVICE_NAME")],
    None,
    Resolves(expected_view(json!({"mode": "sentinel", "nodes": ["sentinel-a:26379"], "service_name": "primary", "sentinel_password": "sentinel-secret", "db": 0}))),
)]
#[case::a_host_breaks_sentinel(
    json!({"host": "ignored", "sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"}),
    &[],
    None,
    Fails("host cannot be combined with sentinel_nodes, which redis-py rejects at startup", Some(TypeError)),
)]
#[case::a_url_beats_sentinels(
    json!({"sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary", "url": "redis://single:6379"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "redis://single:6379"}))),
)]
#[case::an_explicit_credential_provider_beats_cloud_auth(
    json!({"aws_iam_auth": true, "credential_provider": true, "gcp_service_account": "account", "host": "redis", "password": "dropped"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "host"}}))),
)]
#[case::gcp_beats_azure_and_aws(
    json!({"gcp_service_account": "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com", "aws_iam_auth": true, "azure_redis_ad_token": "True", "host": "redis", "password": "dropped"}),
    &[("gcp_service_account", "REDIS_GCP_SERVICE_ACCOUNT")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "gcp_iam", "service_account": "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"}}))),
)]
#[case::gcp_ca_certs_replace_the_configured_ones_under_ssl(
    json!({"gcp_service_account": "account", "gcp_ssl_ca_certs": "/gcp-ca.pem", "host": "redis", "ssl": true, "ssl_ca_certs": "/configured-ca.pem"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": "/gcp-ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "gcp_iam", "service_account": "account"}}))),
)]
#[case::azure_beats_aws_and_takes_the_environment_username(
    json!({"aws_iam_auth": true, "azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis", "username": "dropped"}),
    &[],
    Some("environment-user"),
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "azure_ad", "username": "environment-user"}}))),
)]
#[case::an_empty_environment_username_sends_only_the_token(
    json!({"azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis", "username": ""}),
    &[("username", "REDIS_USERNAME")],
    Some(""),
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "azure_ad", "username": null}}))),
)]
#[case::azure_without_a_username_sends_only_the_token(
    json!({"azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": true, "azure_tenant_id": "tenant", "host": "redis"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "auth": {"kind": "azure_ad", "username": null}}))),
)]
#[case::azure_needs_the_literal_true(
    json!({"azure_client_id": "client", "azure_client_secret": "secret", "azure_redis_ad_token": "1", "azure_tenant_id": "tenant", "host": "redis"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::aws_iam_over_tls_with_the_default_region(
    json!({"aws_iam_auth": "yes", "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "password": "dropped", "ssl": true, "aws_iam_region": "us-east-1"}),
    &[("aws_iam_region", "AWS_DEFAULT_REGION")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "cache-1", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "aws_iam", "user_name": "iam-user", "cache_name": "cache-1", "region": "us-east-1", "serverless": false}}))),
)]
#[case::aws_region_beats_the_default_region(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "ssl": true, "aws_iam_region": "eu-central-1"}),
    &[("aws_iam_region", "AWS_REGION")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "cache-1", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "aws_iam", "user_name": "iam-user", "cache_name": "cache-1", "region": "eu-central-1", "serverless": false}}))),
)]
#[case::aws_iam_serverless_over_a_rediss_url_drops_its_userinfo(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_region": "eu-west-1", "aws_iam_serverless": true, "aws_iam_user_name": "iam-user", "url": "rediss://user:pass@cache-1:6379"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "rediss://cache-1:6379", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}, "auth": {"kind": "aws_iam", "user_name": "iam-user", "cache_name": "cache-1", "region": "eu-west-1", "serverless": true}}))),
)]
#[case::aws_iam_without_tls_fails(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_region": "us-east-1", "aws_iam_user_name": "iam-user", "host": "cache-1"}),
    &[],
    None,
    Fails("AWS ElastiCache IAM Redis authentication requires TLS", Some(ValueError)),
)]
#[case::aws_iam_without_a_region_fails(
    json!({"aws_iam_auth": true, "aws_iam_cache_name": "cache-1", "aws_iam_user_name": "iam-user", "host": "cache-1", "ssl": true}),
    &[],
    None,
    Fails("AWS ElastiCache IAM Redis authentication requires: aws_iam_region", Some(ValueError)),
)]
#[case::tls_options_from_settings(
    json!({"host": "redis", "ssl": true, "ssl_ca_certs": "/ca.pem", "ssl_cert_reqs": "none", "ssl_check_hostname": true}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": "/ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": false, "check_hostname": true}}))),
)]
#[case::a_null_cert_requirement_skips_verification(
    json!({"host": "redis", "ssl": true, "ssl_cert_reqs": null}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": false, "check_hostname": false}}))),
)]
#[case::a_verify_mode_of_none_skips_verification(
    json!({"host": "redis", "ssl": true, "ssl_cert_reqs": 0}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": false, "check_hostname": false}}))),
)]
#[case::a_verify_mode_of_optional_verifies(
    json!({"host": "redis", "ssl": true, "ssl_cert_reqs": 1}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::a_cert_requirement_spelled_like_python_none_fails(
    json!({"ssl": true, "ssl_cert_reqs": "None", "host": "redis"}),
    &[("ssl", "REDIS_SSL"), ("ssl_cert_reqs", "REDIS_SSL_CERT_REQS")],
    None,
    Fails("invalid Redis setting REDIS_SSL_CERT_REQS: expected none, optional, required, null or an ssl.VerifyMode value", None),
)]
#[case::tls_options_from_the_environment(
    json!({"ssl_ca_certs": "/env-ca.pem", "ssl": true, "ssl_cert_reqs": "optional", "host": "redis"}),
    &[("ssl", "REDIS_SSL"), ("ssl_ca_certs", "REDIS_SSL_CA_CERTS"), ("ssl_cert_reqs", "REDIS_SSL_CERT_REQS")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": "/env-ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::connect_timeout_is_separate_from_the_socket_timeout(
    json!({"host": "redis", "socket_connect_timeout": 0.25, "socket_timeout": 2.5}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "socket_timeout": 2.5, "socket_connect_timeout": 0.25}))),
)]
#[case::environment_references_resolve(
    json!({"host": "referenced", "password": "referenced-secret"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "referenced", "port": 6379, "db": 0, "auth": {"kind": "static", "username": null, "password": "referenced-secret"}}))),
)]
#[case::no_target_fails(
    json!({"port": 6379}),
    &[],
    None,
    Fails("either host or url must be specified for redis", Some(ValueError)),
)]
#[case::client_certificates(
    json!({"host": "redis", "ssl": true, "ssl_certfile": "/client.pem", "ssl_keyfile": "/client.key"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": "/client.pem", "keyfile": "/client.key", "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::a_rediss_url_takes_tls_options_from_its_query(
    json!({"url": "rediss://cache:6380/0?ssl_cert_reqs=none&ssl_ca_certs=/ca.pem&ssl_check_hostname=yes"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "rediss://cache:6380/0?ssl_cert_reqs=none&ssl_ca_certs=/ca.pem&ssl_check_hostname=yes", "tls": {"ca_certs": "/ca.pem", "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": false, "check_hostname": true}}))),
)]
#[case::a_url_query_carries_socket_timeout(
    json!({"url": "redis://cache:6379/0?socket_timeout=2"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "redis://cache:6379/0?socket_timeout=2", "socket_timeout": 2.0, "socket_connect_timeout": 2.0}))),
)]
#[case::url_query_settings_win_over_the_kwargs(
    json!({"health_check_interval": 4, "socket_connect_timeout": 7, "socket_timeout": 1, "url": "redis://cache:6379/0?socket_connect_timeout=0.5&health_check_interval=9&socket_timeout=3"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "redis://cache:6379/0?socket_connect_timeout=0.5&health_check_interval=9&socket_timeout=3", "socket_timeout": 3.0, "socket_connect_timeout": 0.5, "health_check_interval": 9.0}))),
)]
#[case::a_repeated_url_query_parameter_takes_its_first_value(
    json!({"url": "redis://cache:6379/0?socket_timeout=2&socket_timeout=9"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "redis://cache:6379/0?socket_timeout=2&socket_timeout=9", "socket_timeout": 2.0, "socket_connect_timeout": 2.0}))),
)]
#[case::a_url_query_number_parse_url_cannot_read(
    json!({"url": "redis://cache:6379/0?socket_timeout=slow"}),
    &[],
    None,
    Fails("invalid Redis setting url ?socket_timeout: redis-py's parse_url cannot read it as a number", Some(ValueError)),
)]
#[case::tls_settings_outside_a_url_never_fail_it(
    json!({"db": "99999999999999999999", "ssl_cert_reqs": "strict", "url": "rediss://cache:6380"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "url", "url": "rediss://cache:6380", "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::db_is_not_read_for_a_cluster(
    json!({"db": 99_999_999_999_i64, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Fails("db must be 0 in cluster mode", Some(RedisClusterException)),
)]
#[case::cluster_db_true(
    json!({"db": true, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Fails("db must be 0 in cluster mode", Some(RedisClusterException)),
)]
#[case::cluster_db_false(
    json!({"db": false, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::cluster_db_fraction(
    json!({"db": 0.5, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Fails("db must be 0 in cluster mode", Some(RedisClusterException)),
)]
#[case::cluster_db_float_zero(
    json!({"db": 0.0, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::cluster_db_nonempty_list(
    json!({"db": [1], "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Fails("db must be 0 in cluster mode", Some(RedisClusterException)),
)]
#[case::cluster_db_empty_list(
    json!({"db": [], "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"], "health_check_interval": 25.0}))),
)]
#[case::ssl_ciphers(
    json!({"host": "redis", "ssl": true, "ssl_ciphers": "HIGH"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::retry_on_timeout(
    json!({"host": "redis", "retry_on_timeout": true}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::malformed_port(
    json!({"host": "redis", "port": "abc"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::malformed_db(
    json!({"db": "first", "host": "redis"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::socket_timeout_not_a_number(
    json!({"host": "redis", "socket_timeout": "slow"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::malformed_port_from_the_environment(
    json!({"port": "abc", "host": "redis"}),
    &[("host", "REDIS_HOST"), ("port", "REDIS_PORT")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::protocol_3(
    json!({"host": "redis", "protocol": 3}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::retry_on_error_from_the_environment(
    json!({"retry_on_error": true, "host": "redis"}),
    &[("retry_on_error", "REDIS_RETRY_ON_ERROR")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::unix_socket_path(
    json!({"host": "redis", "unix_socket_path": "/tmp/redis.sock"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "unix", "path": "/tmp/redis.sock"}))),
)]
#[case::unix_socket_path_keeps_tls_settings_out_of_the_target(
    json!({"host": "redis", "ssl": true, "ssl_ca_certs": "/ca.pem", "unix_socket_path": "/tmp/redis.sock"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "unix", "path": "/tmp/redis.sock"}))),
)]
#[case::ssl_ca_data(
    json!({"host": "redis", "ssl": true, "ssl_ca_data": "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----", "ca_path": null, "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::ssl_ca_path_from_the_environment(
    json!({"ssl": true, "ssl_ca_path": "/etc/ssl/private-cas", "host": "redis"}),
    &[("ssl", "REDIS_SSL"), ("ssl_ca_path", "REDIS_SSL_CA_PATH")],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "tls": {"ca_certs": null, "ca_data": null, "ca_path": "/etc/ssl/private-cas", "certfile": null, "keyfile": null, "verify_certificate": true, "check_hostname": false}}))),
)]
#[case::health_check_interval_is_whole_seconds(
    json!({"health_check_interval": "7", "host": "redis"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0, "health_check_interval": 7.0}))),
)]
#[case::health_check_interval_zero_is_off(
    json!({"health_check_interval": 0, "host": "redis"}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "host", "host": "redis", "port": 6379, "db": 0}))),
)]
#[case::health_check_interval_zero_keeps_a_cluster_off(
    json!({"health_check_interval": 0, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Resolves(expected_view(json!({"mode": "cluster", "nodes": ["node-a:7000"]}))),
)]
#[case::a_cluster_rejects_a_nonzero_db(
    json!({"db": 3, "startup_nodes": [{"host": "node-a", "port": 7000}]}),
    &[],
    None,
    Fails("db must be 0 in cluster mode", Some(RedisClusterException)),
)]
#[case::unix_socket_path_without_a_host_fails_like_python(
    json!({"unix_socket_path": "/tmp/redis.sock"}),
    &[],
    None,
    Fails("either host or url must be specified for redis", Some(ValueError)),
)]
fn resolves_like_python_redis(
    #[case] merged: serde_json::Value,
    #[case] sources: &[(&str, &str)],
    #[case] redis_username: Option<&str>,
    #[case] expected: Outcome,
) {
    let outcome = interpret_merged(merged, sources, redis_username);
    let rendered = outcome
        .as_ref()
        .map(view)
        .map_err(|error| error.to_string());

    match expected {
        Resolves(expected) => assert_eq!(rendered, Ok(expected)),
        Refuses(message) => assert_eq!(rendered, Err(message.to_owned())),
        Fails(message, raised) => {
            assert_eq!(rendered, Err(message.to_owned()));
            assert_eq!(
                outcome
                    .err()
                    .and_then(|error| error.raised_building_the_python_client()),
                raised
            );
        }
    }
}
