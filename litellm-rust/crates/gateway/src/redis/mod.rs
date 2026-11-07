use std::{sync::Arc, time::Duration};

use litellm_circuit_breaker::Config as BreakerConfig;
use litellm_config::{CacheSettings, Config, RedisSettings, Value};
use litellm_core_utils::settings::Lookup;
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{HostServices, RedisHosts, SharedEnvironment, TracingEvents};
use litellm_redis_resolve::Expected;
use litellm_redis_types::Resolution;

use crate::Error;

mod configured;
mod layers;
mod secrets;
#[cfg(test)]
mod surface_tests;

use configured::ConfiguredRedis;
use secrets::SecretSnapshot;

pub(crate) struct Resources {
    pub(crate) http: litellm_http::Client,
    pub(crate) lookup: SharedEnvironment,
}

const ENVIRONMENT_PING_TIMEOUT: Duration = Duration::from_secs(2);
type Params = Resolution;

pub struct RedisClients {
    pub coordination: Option<RedisClient>,
    pub transaction_buffer: Option<RedisClient>,
    pub response_cache: Option<RedisClient>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Plan {
    coordination: Option<Target>,
    transaction_buffer_fallback: Option<Params>,
    response_cache: Option<Params>,
}

#[derive(Debug, PartialEq)]
enum Target {
    Configured(Params),
    Environment(Params),
}

pub(crate) fn plan(config: &Config, lookup: &dyn Lookup) -> Result<Plan, Error> {
    let response_cache = redis_cache_params(config)
        .map(|settings| resolve(&ConfiguredRedis::cache_params(settings), lookup))
        .transpose()?;
    let coordination = match &config.general_settings.coordination_redis {
        Some(settings) => Some(Target::Configured(resolve(
            &ConfiguredRedis::coordination(settings)?,
            lookup,
        )?)),
        None => match (&response_cache, config.litellm_settings.cache == Some(true)) {
            (Some(cache), _) => Some(Target::Configured(cache.clone())),
            (None, true) => environment(lookup)?.map(Target::Configured),
            (None, false) => optional_environment(lookup).map(Target::Environment),
        },
    };
    let transaction_buffer_fallback = match (&coordination, use_transaction_buffer(config)) {
        (Some(Target::Configured(_)), _) | (_, false) => None,
        (Some(Target::Environment(_)) | None, true) => environment(lookup)?,
    };
    Ok(Plan {
        coordination,
        transaction_buffer_fallback,
        response_cache,
    })
}

pub(crate) async fn connect(
    config: &Config,
    lookup: &dyn Lookup,
    resources: &Resources,
) -> Result<RedisClients, Error> {
    let plan = plan(config, lookup)?;
    let breaker = breaker_config(lookup)?;
    let events: Arc<dyn RedisEvents> = Arc::new(TracingEvents::new(lookup));
    let hosts = RedisHosts::default();
    let client = |params: Params| -> Result<RedisClient, Error> {
        Ok(hosts.client(
            params,
            HostServices {
                events: Some(events.clone()),
                http: resources.http.clone(),
                environment: resources.lookup.clone(),
                credential_provider: None,
            },
            breaker,
        )?)
    };
    let coordination = match plan.coordination {
        Some(Target::Configured(params)) => Some(client(params)?),
        Some(Target::Environment(params)) => match client(params) {
            Ok(client) => reachable(client).await,
            Err(_) => {
                tracing::warn!(
                    "REDIS_* environment variables did not form a valid Redis client, cross-pod coordination stays in memory"
                );
                None
            }
        },
        None => None,
    };
    let transaction_buffer = match (&coordination, plan.transaction_buffer_fallback) {
        (Some(coordination), _) => Some(coordination.clone()),
        (None, Some(params)) => Some(client(params)?),
        (None, None) => None,
    };
    let response_cache = plan.response_cache.map(client).transpose()?;
    Ok(RedisClients {
        coordination,
        transaction_buffer,
        response_cache,
    })
}

fn redis_cache_params(config: &Config) -> Option<&RedisSettings> {
    let settings = &config.litellm_settings;
    let params = settings.cache_params.as_ref()?;
    match (settings.cache, params) {
        (Some(true), CacheSettings::Redis(redis)) => Some(redis),
        _ => None,
    }
}

fn resolve(configured: &ConfiguredRedis, lookup: &dyn Lookup) -> Result<Params, Error> {
    let secrets = SecretSnapshot::read(configured.secret_names(), lookup);
    let resolution = configured.resolve(&secrets, lookup)?;
    for ignored in &resolution.ignored {
        tracing::warn!(
            "Redis setting {} is not supported by the Rust client and is ignored: {}",
            ignored.setting,
            ignored.consequence
        );
    }
    Ok(resolution)
}

fn environment(lookup: &dyn Lookup) -> Result<Option<Params>, Error> {
    match environment_has_target(lookup) {
        true => Ok(Some(resolve(&ConfiguredRedis::environment_only(), lookup)?)),
        false => Ok(None),
    }
}

fn environment_has_target(lookup: &dyn Lookup) -> bool {
    [
        "REDIS_HOST",
        "REDIS_URL",
        "REDIS_CLUSTER_NODES",
        "REDIS_SENTINEL_NODES",
    ]
    .iter()
    .any(|name| lookup.get(name).is_some())
}

fn optional_environment(lookup: &dyn Lookup) -> Option<Params> {
    match environment(lookup) {
        Ok(params) => params,
        Err(error) => {
            tracing::warn!(error = %error, "REDIS_* environment variables are invalid, cross-pod coordination stays in memory");
            None
        }
    }
}

fn use_transaction_buffer(config: &Config) -> bool {
    match config
        .general_settings
        .additional_fields
        .get("use_redis_transaction_buffer")
    {
        Some(Value::Bool(enabled)) => *enabled,
        Some(Value::String(text)) => matches!(
            text.to_ascii_lowercase().as_str(),
            "true" | "1" | "yes" | "on"
        ),
        _ => false,
    }
}

async fn reachable(client: RedisClient) -> Option<RedisClient> {
    match tokio::time::timeout(ENVIRONMENT_PING_TIMEOUT, client.ping()).await {
        Ok(Ok(())) => Some(client),
        Ok(Err(error)) => {
            tracing::warn!(%error, "REDIS_* environment variables named a Redis that is not reachable, cross-pod coordination stays in memory");
            None
        }
        Err(_) => {
            tracing::warn!(
                "REDIS_* environment variables named a Redis that did not answer PING in 2s, cross-pod coordination stays in memory"
            );
            None
        }
    }
}

fn parse_seconds(text: &str) -> Option<Duration> {
    text.parse::<f64>()
        .ok()
        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
}

fn breaker_config(lookup: &dyn Lookup) -> Result<Option<BreakerConfig>, Error> {
    let enabled = lookup
        .get("REDIS_CIRCUIT_BREAKER_ENABLED")
        .is_none_or(|value| value.eq_ignore_ascii_case("true"));
    if !enabled {
        return Ok(None);
    }
    let defaults = BreakerConfig::default();
    Ok(Some(BreakerConfig {
        failure_threshold: environment_value(
            lookup,
            "REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            Expected::Integer,
            |text| text.parse().ok(),
        )?
        .unwrap_or(defaults.failure_threshold),
        recovery_timeout: environment_value(
            lookup,
            "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
            Expected::Seconds,
            parse_seconds,
        )?
        .unwrap_or(defaults.recovery_timeout),
        timeout_min_duration: environment_value(
            lookup,
            "REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION",
            Expected::Seconds,
            parse_seconds,
        )?
        .unwrap_or(defaults.timeout_min_duration),
    }))
}

fn environment_value<T>(
    lookup: &dyn Lookup,
    name: &str,
    expected: Expected,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, litellm_redis_resolve::Error> {
    lookup
        .get(name)
        .map(|value| {
            parse(&value).ok_or_else(|| litellm_redis_resolve::Error::Invalid {
                setting: name.to_owned(),
                expected,
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use litellm_core_utils::settings::Lookup;
    use std::{collections::HashMap, time::Duration};

    use litellm_config::Config;
    use litellm_redis_client::{Standalone, StaticAuth, commands};
    use litellm_redis_testing::RedisServer;
    use litellm_redis_types::{Auth, HealthCheck};
    use rstest::rstest;

    use std::sync::Arc;

    use super::{Params, Plan, Resources, Target, plan};
    use crate::Error;

    async fn connect(config: &Config, lookup: &dyn Lookup) -> Result<super::RedisClients, Error> {
        let resources = Resources {
            http: litellm_http::Client::for_test(),
            lookup: Arc::new(|_: &str| None),
        };
        super::connect(config, lookup, &resources).await
    }

    fn host(name: &str, port: u16) -> Params {
        Params {
            target: litellm_redis_client::Target::Standalone(Standalone {
                host: name.to_owned(),
                port,
                tls: None,
                db: 0,
            }),
            auth: Auth::Static(StaticAuth::default()),
            namespace: None,
            timeout: None,
            connect_timeout: None,
            health_check: HealthCheck::Unset,
            keepalive: true,
            ignored: Vec::new(),
        }
    }

    fn resolved(yaml: &str, environment: &[(&str, &str)]) -> Result<Plan, Error> {
        let environment: HashMap<String, String> = environment
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        plan(&Config::from_yaml(yaml).unwrap(), &|name: &str| {
            environment.get(name).cloned()
        })
    }

    fn configured(coordination: Params, response_cache: Option<Params>) -> Plan {
        Plan {
            coordination: Some(Target::Configured(coordination)),
            transaction_buffer_fallback: None,
            response_cache,
        }
    }

    const NO_REDIS: Plan = Plan {
        coordination: None,
        transaction_buffer_fallback: None,
        response_cache: None,
    };

    #[rstest]
    #[case::nothing_configured("model_list: []", &[], NO_REDIS)]
    #[case::coordination_block_wins_over_the_cache_redis(
        "general_settings:\n  coordination_redis:\n    host: coord\nlitellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache",
        &[],
        configured(host("coord", 6379), Some(host("cache", 6379)))
    )]
    #[case::cache_params_without_cache_true_are_unused(
        "litellm_settings:\n  cache_params:\n    type: redis\n    host: cache",
        &[],
        NO_REDIS
    )]
    #[case::non_redis_cache_takes_coordination_from_the_environment_without_a_ping(
        "litellm_settings:\n  cache: true\n  cache_params:\n    type: local",
        &[("REDIS_HOST", "env")],
        configured(host("env", 6379), None)
    )]
    #[case::environment_only_coordination_is_probed(
        "model_list: []",
        &[("REDIS_HOST", "env")],
        Plan { coordination: Some(Target::Environment(host("env", 6379))), transaction_buffer_fallback: None, response_cache: None }
    )]
    #[case::the_transaction_buffer_falls_back_to_the_environment_only_when_enabled(
        "general_settings:\n  use_redis_transaction_buffer: \"true\"",
        &[("REDIS_HOST", "env")],
        Plan { coordination: Some(Target::Environment(host("env", 6379))), transaction_buffer_fallback: Some(host("env", 6379)), response_cache: None }
    )]
    fn resolves_each_redis_role_like_the_proxy(
        #[case] yaml: &str,
        #[case] environment: &[(&str, &str)],
        #[case] expected: Plan,
    ) {
        assert_eq!(resolved(yaml, environment).unwrap(), expected);
    }

    #[rstest]
    fn redis_migration_regression_preserves_borrowed_cache_namespace() {
        let yaml = "litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache\n    namespace: responses\n    ttl: 600";
        let cache = Params {
            namespace: Some("responses".to_owned()),
            ..host("cache", 6379)
        };

        assert_eq!(
            resolved(yaml, &[]).unwrap(),
            configured(cache.clone(), Some(cache))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn borrowed_coordination_and_transaction_buffer_share_the_cache_namespace() {
        let server = RedisServer::start().await.unwrap();
        let config = Config::from_yaml(&format!(
            "litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: 127.0.0.1\n    port: {}\n    namespace: responses",
            port_of(&server)
        ))
        .unwrap();
        let clients = connect(&config, &|_: &str| None).await.unwrap();
        let coordination = clients.coordination.unwrap();
        let buffer = clients.transaction_buffer.unwrap();
        let cache = clients.response_cache.unwrap();
        let unnamespaced = connect(&coordination_config(&server), &|_: &str| None)
            .await
            .unwrap()
            .coordination
            .unwrap();

        coordination
            .send(commands::set("rate-limit", "1", Duration::from_secs(60)))
            .await
            .unwrap();
        buffer
            .send(commands::set("spend", "2", Duration::from_secs(60)))
            .await
            .unwrap();

        assert_eq!(
            cache
                .send(commands::get::<String>("rate-limit"))
                .await
                .unwrap()
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            cache
                .send(commands::get::<String>("spend"))
                .await
                .unwrap()
                .as_deref(),
            Some("2")
        );
        assert_eq!(
            unnamespaced
                .send(commands::get::<String>("responses:rate-limit"))
                .await
                .unwrap()
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            unnamespaced
                .send(commands::get::<String>("responses:spend"))
                .await
                .unwrap()
                .as_deref(),
            Some("2")
        );
        assert!(
            unnamespaced
                .send(commands::get::<String>("rate-limit"))
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            unnamespaced
                .send(commands::get::<String>("spend"))
                .await
                .unwrap()
                .is_none()
        );
    }

    #[rstest]
    #[case::malformed_cluster_nodes(&[("REDIS_CLUSTER_NODES", "[{\"host\": ")])]
    fn redis_migration_regression_malformed_optional_environment_is_absent(
        #[case] environment: &[(&str, &str)],
    ) {
        assert_eq!(resolved("model_list: []", environment).unwrap(), NO_REDIS);
    }

    #[rstest]
    #[case::optional_environment("model_list: []", &[("REDIS_HOST", "env"), ("REDIS_PORT", "not-a-port")])]
    #[case::transaction_buffer_environment(
        "general_settings:\n  use_redis_transaction_buffer: true",
        &[("REDIS_HOST", "env"), ("REDIS_PORT", "not-a-port")]
    )]
    #[case::response_cache(
        "litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: env\n    port: not-a-port",
        &[]
    )]
    fn a_malformed_port_outside_the_coordination_block_falls_back_to_6379_like_redis_py(
        #[case] yaml: &str,
        #[case] environment: &[(&str, &str)],
    ) {
        let plan = resolved(yaml, environment).unwrap();

        let target = match (plan.coordination, plan.transaction_buffer_fallback) {
            (Some(Target::Configured(params) | Target::Environment(params)), _)
            | (None, Some(params)) => params,
            (None, None) => panic!("expected a Redis target"),
        };
        assert_eq!(target.target, host("env", 6379).target);
    }

    #[rstest]
    #[case::port("port: not-a-port", "port")]
    #[case::ssl("ssl: maybe", "ssl")]
    fn a_malformed_coordination_block_fails_like_the_pydantic_model(
        #[case] setting: &str,
        #[case] expected: &str,
    ) {
        let error = resolved(
            &format!("general_settings:\n  coordination_redis:\n    host: coord\n    {setting}"),
            &[],
        )
        .unwrap_err();

        assert!(
            matches!(error, Error::Redis(litellm_redis_resolve::Error::Invalid { setting, .. }) if setting == expected)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn redis_migration_regression_invalid_optional_environment_url_is_absent() {
        let config = Config::from_yaml("model_list: []").unwrap();

        let clients = connect(&config, &|name: &str| {
            (name == "REDIS_URL").then(|| "not a redis URL".to_owned())
        })
        .await
        .unwrap();

        assert!(clients.coordination.is_none());
    }

    fn port_of(server: &RedisServer) -> String {
        server.url().rsplit(':').next().unwrap().to_owned()
    }

    fn coordination_config(server: &RedisServer) -> Config {
        Config::from_yaml(&format!(
            "general_settings:\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\n    socket_timeout: 0.5",
            port_of(server)
        ))
        .unwrap()
    }

    async fn eventually_pings(client: &litellm_redis_client::RedisClient) -> bool {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        while tokio::time::Instant::now() < deadline {
            if client.ping().await.is_ok() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        false
    }

    #[rstest]
    #[tokio::test]
    async fn boot_does_not_wait_for_a_configured_redis_that_is_down() {
        let server = RedisServer::start().await.unwrap();
        server.stop().await.unwrap();

        let clients = connect(&coordination_config(&server), &|_: &str| None)
            .await
            .unwrap();
        let coordination = clients.coordination.unwrap();
        let while_down = coordination.ping().await;
        server.resume().await.unwrap();

        assert!(while_down.is_err());
        assert!(eventually_pings(&coordination).await);
    }

    #[rstest]
    #[tokio::test]
    async fn the_coordination_client_survives_a_redis_restart_and_backs_the_transaction_buffer() {
        let server = RedisServer::start().await.unwrap();
        let clients = connect(&coordination_config(&server), &|_: &str| None)
            .await
            .unwrap();
        let coordination = clients.coordination.unwrap();
        coordination
            .send(commands::set("written", "1", Duration::from_secs(60)))
            .await
            .unwrap();

        server.restart().await.unwrap();

        assert!(eventually_pings(&coordination).await);
        let buffer = clients.transaction_buffer.unwrap();
        assert!(buffer.ping().await.is_ok());
    }

    #[rstest]
    #[tokio::test]
    async fn an_environment_only_tls_redis_is_reached_with_its_ca() {
        crate::install_crypto_provider();
        let server = litellm_redis_testing::RedisTlsServer::start(false)
            .await
            .unwrap();
        let ca = server.files().ca.display().to_string();
        let port = server.port().to_string();
        let config = Config::from_yaml("model_list: []").unwrap();

        let clients = connect(&config, &|name: &str| match name {
            "REDIS_HOST" => Some(server.host().to_owned()),
            "REDIS_PORT" => Some(port.clone()),
            "REDIS_SSL" => Some("true".to_owned()),
            "REDIS_SSL_CA_CERTS" => Some(ca.clone()),
            _ => None,
        })
        .await
        .unwrap();

        assert!(clients.coordination.is_some());
    }

    #[rstest]
    #[tokio::test]
    async fn an_unreachable_environment_only_redis_leaves_coordination_in_memory() {
        let server = RedisServer::start().await.unwrap();
        let port = port_of(&server);
        server.stop().await.unwrap();
        let config = Config::from_yaml("model_list: []").unwrap();

        let clients = connect(&config, &|name: &str| match name {
            "REDIS_HOST" => Some("127.0.0.1".to_owned()),
            "REDIS_PORT" => Some(port.clone()),
            _ => None,
        })
        .await
        .unwrap();

        assert!(clients.coordination.is_none());
        assert!(clients.transaction_buffer.is_none());
    }

    #[rstest]
    #[tokio::test]
    async fn a_reachable_environment_only_redis_becomes_coordination() {
        let server = RedisServer::start().await.unwrap();
        let port = port_of(&server);
        let config = Config::from_yaml("model_list: []").unwrap();

        let clients = connect(&config, &|name: &str| match name {
            "REDIS_HOST" => Some("127.0.0.1".to_owned()),
            "REDIS_PORT" => Some(port.clone()),
            _ => None,
        })
        .await
        .unwrap();

        assert!(clients.coordination.unwrap().ping().await.is_ok());
    }
}
