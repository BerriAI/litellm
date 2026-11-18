use std::{sync::Arc, time::Duration};

use litellm_circuit_breaker::Config as BreakerConfig;
use litellm_config::{CacheSettings, Config, RedisSettings};
use litellm_env::Lookup;
use litellm_http::ClientVariant;
use litellm_python_compat::flags::{lower_is_true, str_to_bool_or_truthy};
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{HostServices, RedisHosts, TracingEvents};
use litellm_redis_resolve::{Expected, Surface};
use litellm_redis_types::Resolution;
use litellm_secrets::SecretResolver;
use serde::Deserialize;

use crate::{Error, Resources};

const ENVIRONMENT_PING_TIMEOUT: Duration = Duration::from_secs(2);
type Params = Resolution;

#[derive(Clone, Default)]
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

pub(crate) async fn plan(config: &Config, secrets: &SecretResolver) -> Result<Plan, Error> {
    let response_cache = match redis_cache_params(config) {
        Some(settings) => resolve(Surface::CacheParams(settings), secrets).await?,
        None => None,
    };
    let coordination = match &config.general_settings.coordination_redis {
        Some(settings) => resolve(Surface::Coordination(settings), secrets)
            .await?
            .map(Target::Configured),
        None => match (&response_cache, config.litellm_settings.cache == Some(true)) {
            (Some(cache), _) => Some(Target::Configured(cache.clone())),
            (None, true) => resolve(Surface::Environment, secrets)
                .await?
                .map(Target::Configured),
            (None, false) => optional_environment(secrets).await.map(Target::Environment),
        },
    };
    let transaction_buffer_fallback = match (&coordination, use_transaction_buffer(config)) {
        (Some(Target::Configured(_)), _) | (_, false) => None,
        (Some(Target::Environment(_)) | None, true) => {
            resolve(Surface::Environment, secrets).await?
        }
    };
    Ok(Plan {
        coordination,
        transaction_buffer_fallback,
        response_cache,
    })
}

pub(crate) async fn connect(config: &Config, resources: &Resources) -> Result<RedisClients, Error> {
    let plan = plan(config, &resources.secrets).await?;
    let environment: &dyn Lookup = resources.environment.as_ref();
    let breaker = breaker_config(environment)?;
    let events: Arc<dyn RedisEvents> = Arc::new(TracingEvents::new(environment));
    let http = resources
        .core
        .pool
        .client(&resources.http, ClientVariant::Provider)?;
    let hosts = RedisHosts::default();
    let client = |params: Params| -> Result<RedisClient, Error> {
        Ok(hosts.client(
            params,
            HostServices {
                events: Some(events.clone()),
                http: http.clone(),
                environment: resources.environment.clone(),
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

async fn resolve(surface: Surface<'_>, secrets: &SecretResolver) -> Result<Option<Params>, Error> {
    let resolution = litellm_redis_resolve::resolve(surface, secrets).await?;
    for ignored in resolution.iter().flat_map(|resolution| &resolution.ignored) {
        tracing::warn!(
            "Redis setting {} is not supported by the Rust client and is ignored: {}",
            ignored.setting,
            ignored.consequence
        );
    }
    Ok(resolution)
}

async fn optional_environment(secrets: &SecretResolver) -> Option<Params> {
    match resolve(Surface::Environment, secrets).await {
        Ok(params) => params,
        Err(error) => {
            tracing::warn!(error = %error, "REDIS_* environment variables are invalid, cross-pod coordination stays in memory");
            None
        }
    }
}

fn use_transaction_buffer(config: &Config) -> bool {
    config
        .general_settings
        .additional_fields
        .get("use_redis_transaction_buffer")
        .and_then(|value| litellm_python_compat::Value::deserialize(value).ok())
        .is_some_and(|value| str_to_bool_or_truthy(&value))
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
        .is_none_or(|value| lower_is_true(&value));
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
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use litellm_config::Config;
    use litellm_redis_client::{Standalone, StaticAuth, commands};
    use litellm_redis_testing::RedisServer;
    use litellm_redis_types::{Auth, HealthCheck};
    use rstest::rstest;

    use super::{Params, Plan, Target, plan, use_transaction_buffer};
    use crate::{Error, Resources};

    fn resources(config: &Config, environment: &[(&str, &str)]) -> Resources {
        let environment: HashMap<String, String> = environment
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        Resources::over(
            config,
            Arc::new(move |name: &str| environment.get(name).cloned()),
        )
        .unwrap()
    }

    async fn connect(
        config: &Config,
        environment: &[(&str, &str)],
    ) -> Result<super::RedisClients, Error> {
        super::connect(config, &resources(config, environment)).await
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

    async fn resolved(yaml: &str, environment: &[(&str, &str)]) -> Result<Plan, Error> {
        let config = Config::from_yaml(yaml).unwrap();
        plan(&config, &resources(&config, environment).secrets).await
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
    #[tokio::test]
    async fn resolves_each_redis_role_like_the_proxy(
        #[case] yaml: &str,
        #[case] environment: &[(&str, &str)],
        #[case] expected: Plan,
    ) {
        assert_eq!(resolved(yaml, environment).await.unwrap(), expected);
    }

    /// Expected values from `str_to_bool(value) if isinstance(value, str) else value` read as a condition,
    /// how `get_transaction_buffer_redis_cache` and `should_commit_spend_updates_to_redis` read the setting.
    #[rstest]
    #[case::text_true("\"true\"", true)]
    #[case::padded_text("\" True \"", true)]
    #[case::text_false("\"false\"", false)]
    #[case::yes_is_not_true("\"yes\"", false)]
    #[case::on_is_not_true("\"on\"", false)]
    #[case::text_one_is_not_true("\"1\"", false)]
    #[case::number_one("1", true)]
    #[case::number_zero("0", false)]
    #[case::boolean("true", true)]
    #[case::null("null", false)]
    fn the_transaction_buffer_flag_reads_like_str_to_bool(
        #[case] value: &str,
        #[case] expected: bool,
    ) {
        let config = Config::from_yaml(&format!(
            "general_settings:\n  use_redis_transaction_buffer: {value}"
        ))
        .unwrap();
        assert_eq!(use_transaction_buffer(&config), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn redis_migration_regression_preserves_borrowed_cache_namespace() {
        let yaml = "litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache\n    namespace: responses\n    ttl: 600";
        let cache = Params {
            namespace: Some("responses".to_owned()),
            ..host("cache", 6379)
        };

        assert_eq!(
            resolved(yaml, &[]).await.unwrap(),
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
        let clients = connect(&config, &[]).await.unwrap();
        let coordination = clients.coordination.unwrap();
        let buffer = clients.transaction_buffer.unwrap();
        let cache = clients.response_cache.unwrap();
        let unnamespaced = connect(&coordination_config(&server), &[])
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
    #[tokio::test]
    async fn redis_migration_regression_malformed_optional_environment_is_absent(
        #[case] environment: &[(&str, &str)],
    ) {
        assert_eq!(
            resolved("model_list: []", environment).await.unwrap(),
            NO_REDIS
        );
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
    #[tokio::test]
    async fn a_malformed_port_outside_the_coordination_block_falls_back_to_6379_like_redis_py(
        #[case] yaml: &str,
        #[case] environment: &[(&str, &str)],
    ) {
        let plan = resolved(yaml, environment).await.unwrap();

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
    #[tokio::test]
    async fn a_malformed_coordination_block_fails_like_the_pydantic_model(
        #[case] setting: &str,
        #[case] expected: &str,
    ) {
        let error = resolved(
            &format!("general_settings:\n  coordination_redis:\n    host: coord\n    {setting}"),
            &[],
        )
        .await
        .unwrap_err();

        assert!(
            matches!(error, Error::Redis(litellm_redis_resolve::Error::Invalid { setting, .. }) if setting == expected)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn redis_migration_regression_invalid_optional_environment_url_is_absent() {
        let config = Config::from_yaml("model_list: []").unwrap();

        let clients = connect(&config, &[("REDIS_URL", "not a redis URL")])
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

        let clients = connect(&coordination_config(&server), &[]).await.unwrap();
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
        let clients = connect(&coordination_config(&server), &[]).await.unwrap();
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

        let clients = connect(
            &config,
            &[
                ("REDIS_HOST", server.host()),
                ("REDIS_PORT", port.as_str()),
                ("REDIS_SSL", "true"),
                ("REDIS_SSL_CA_CERTS", ca.as_str()),
            ],
        )
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

        let clients = connect(
            &config,
            &[("REDIS_HOST", "127.0.0.1"), ("REDIS_PORT", port.as_str())],
        )
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

        let clients = connect(
            &config,
            &[("REDIS_HOST", "127.0.0.1"), ("REDIS_PORT", port.as_str())],
        )
        .await
        .unwrap();

        assert!(clients.coordination.unwrap().ping().await.is_ok());
    }
}
