use std::time::Duration;

use litellm_circuit_breaker::{Breaker, Config as BreakerConfig, State};
use litellm_config::{
    CacheSettings, Config, Lookup, RedisEndpoint, RedisSettings, ResolvedRedis, Value,
    environment_has_target,
};
use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, DEFAULT_TIMEOUT, Endpoint, HostEndpoint, Namespace, RedisClient,
    RedisConfig,
};

use crate::Error;

const ENVIRONMENT_PING_TIMEOUT: Duration = Duration::from_secs(2);
type Params = ResolvedRedis;

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

pub(crate) fn plan(config: &Config, lookup: Lookup<'_>) -> Result<Plan, Error> {
    let response_cache = redis_cache_params(config)
        .map(|settings| settings.resolve_cache(lookup))
        .transpose()?;
    let coordination = match &config.general_settings.coordination_redis {
        Some(settings) => Some(Target::Configured(settings.resolve(lookup)?)),
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

pub async fn connect(config: &Config, lookup: Lookup<'_>) -> Result<RedisClients, Error> {
    let plan = plan(config, lookup)?;
    let breaker = breaker_config(lookup)?;
    let coordination = match plan.coordination {
        Some(Target::Configured(params)) => Some(client(params, breaker)?),
        Some(Target::Environment(params)) => match client(params, breaker) {
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
        (None, Some(params)) => Some(client(params, breaker)?),
        (None, None) => None,
    };
    let response_cache = plan
        .response_cache
        .map(|params| client(params, breaker))
        .transpose()?;
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

fn environment(lookup: Lookup<'_>) -> Result<Option<Params>, Error> {
    match environment_has_target(lookup) {
        true => RedisSettings::default()
            .resolve(lookup)
            .map(Some)
            .map_err(Into::into),
        false => Ok(None),
    }
}

fn optional_environment(lookup: Lookup<'_>) -> Option<Params> {
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

fn client(params: Params, breaker: Option<BreakerConfig>) -> Result<RedisClient, Error> {
    tracing::info!(endpoint = ?params.endpoint, namespace = ?params.namespace, "redis client configured");
    let endpoint = match params.endpoint {
        RedisEndpoint::Url(url) => Endpoint::Url(url.expose().to_owned()),
        RedisEndpoint::Host(host) => Endpoint::Host(HostEndpoint {
            host: host.host,
            port: host.port,
            username: host.username.map(|value| value.expose().to_owned()),
            password: host.password.map(|value| value.expose().to_owned()),
            tls: host.tls,
            db: host.db,
        }),
        RedisEndpoint::Cluster(cluster) => Endpoint::Cluster(ClusterEndpoint {
            nodes: cluster
                .nodes
                .into_iter()
                .map(|node| ClusterNode {
                    host: node.host,
                    port: node.port,
                })
                .collect(),
            username: cluster.username.map(|value| value.expose().to_owned()),
            password: cluster.password.map(|value| value.expose().to_owned()),
            tls: cluster.tls,
        }),
    };
    Ok(RedisClient::new(RedisConfig {
        endpoint,
        namespace: params.namespace.map(Namespace::new).unwrap_or_default(),
        timeout: params.timeout.unwrap_or(DEFAULT_TIMEOUT),
        breaker: breaker.map(|config| Breaker::new(config, Box::new(log_transition))),
    })?)
}

fn log_transition(transition: &litellm_circuit_breaker::Transition<State>) {
    match transition.to {
        State::Open { .. } => tracing::warn!(
            generation = transition.generation,
            "redis circuit breaker opened, fast-failing redis calls"
        ),
        State::HalfOpen => tracing::info!(
            generation = transition.generation,
            "redis circuit breaker half open, probing redis"
        ),
        State::Closed { .. } => tracing::info!(
            generation = transition.generation,
            "redis circuit breaker closed, redis recovered"
        ),
    }
}

fn breaker_config(lookup: Lookup<'_>) -> Result<Option<BreakerConfig>, Error> {
    let enabled = lookup("REDIS_CIRCUIT_BREAKER_ENABLED")
        .is_none_or(|value| value.eq_ignore_ascii_case("true"));
    if !enabled {
        return Ok(None);
    }
    let defaults = BreakerConfig::default();
    let seconds = |name: &'static str, default: Duration| -> Result<Duration, Error> {
        lookup(name)
            .map(|value| {
                value
                    .parse::<f64>()
                    .ok()
                    .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                    .ok_or(Error::RedisInvalid {
                        setting: name,
                        reason: "must be a non-negative number of seconds",
                    })
            })
            .unwrap_or(Ok(default))
    };
    let failure_threshold = lookup("REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD")
        .map(|value| {
            value.parse().map_err(|_| Error::RedisInvalid {
                setting: "REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
                reason: "must be an integer",
            })
        })
        .unwrap_or(Ok(defaults.failure_threshold))?;
    Ok(Some(BreakerConfig {
        failure_threshold,
        recovery_timeout: seconds(
            "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
            defaults.recovery_timeout,
        )?,
        timeout_min_duration: seconds(
            "REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION",
            defaults.timeout_min_duration,
        )?,
    }))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use litellm_config::{Config, RedisEndpoint, RedisHost};
    use litellm_redis_testing::RedisServer;
    use rstest::rstest;

    use super::{Params, Plan, Target, connect, plan};
    use crate::Error;

    fn host(name: &str, port: u16) -> Params {
        Params {
            endpoint: RedisEndpoint::Host(RedisHost {
                host: name.to_owned(),
                port,
                username: None,
                password: None,
                tls: false,
                db: 0,
            }),
            namespace: None,
            timeout: None,
        }
    }

    fn resolved(yaml: &str, environment: &[(&str, &str)]) -> Result<Plan, Error> {
        let environment: HashMap<String, String> = environment
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        plan(&Config::from_yaml(yaml).unwrap(), &|name| {
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
        let clients = connect(&config, &|_| None).await.unwrap();
        let coordination = clients.coordination.unwrap();
        let buffer = clients.transaction_buffer.unwrap();
        let cache = clients.response_cache.unwrap();
        let unnamespaced = connect(&coordination_config(&server), &|_| None)
            .await
            .unwrap()
            .coordination
            .unwrap();

        coordination
            .set("rate-limit", "1", Duration::from_secs(60))
            .await
            .unwrap();
        buffer
            .set("spend", "2", Duration::from_secs(60))
            .await
            .unwrap();

        assert_eq!(
            cache.get::<String>("rate-limit").await.unwrap().as_deref(),
            Some("1")
        );
        assert_eq!(
            cache.get::<String>("spend").await.unwrap().as_deref(),
            Some("2")
        );
        assert_eq!(
            unnamespaced
                .get::<String>("responses:rate-limit")
                .await
                .unwrap()
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            unnamespaced
                .get::<String>("responses:spend")
                .await
                .unwrap()
                .as_deref(),
            Some("2")
        );
        assert!(
            unnamespaced
                .get::<String>("rate-limit")
                .await
                .unwrap()
                .is_none()
        );
        assert!(unnamespaced.get::<String>("spend").await.unwrap().is_none());
    }

    #[rstest]
    #[case::malformed_port(&[("REDIS_HOST", "env"), ("REDIS_PORT", "not-a-port")])]
    #[case::malformed_cluster_nodes(&[("REDIS_CLUSTER_NODES", "[{\"host\": ")])]
    fn redis_migration_regression_malformed_optional_environment_is_absent(
        #[case] environment: &[(&str, &str)],
    ) {
        assert_eq!(resolved("model_list: []", environment).unwrap(), NO_REDIS);
    }

    #[rstest]
    fn redis_migration_regression_explicit_malformed_setting_still_fails() {
        let error = resolved(
            "general_settings:\n  coordination_redis:\n    host: coord\n    port: not-a-port",
            &[],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Config(litellm_config::Error::RedisInvalid {
                setting: "port",
                ..
            })
        ));
    }

    #[rstest]
    fn redis_migration_regression_enabled_transaction_buffer_rejects_invalid_environment() {
        let error = resolved(
            "general_settings:\n  use_redis_transaction_buffer: true",
            &[("REDIS_HOST", "env"), ("REDIS_PORT", "not-a-port")],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Config(litellm_config::Error::RedisInvalid {
                setting: "port",
                ..
            })
        ));
    }

    #[rstest]
    fn redis_migration_regression_invalid_response_cache_still_fails() {
        let error = resolved(
            "litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache\n    port: not-a-port",
            &[],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Config(litellm_config::Error::RedisInvalid {
                setting: "port",
                ..
            })
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn redis_migration_regression_invalid_optional_environment_url_is_absent() {
        let config = Config::from_yaml("model_list: []").unwrap();

        let clients = connect(&config, &|name| {
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

        let clients = connect(&coordination_config(&server), &|_| None)
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
        let clients = connect(&coordination_config(&server), &|_| None)
            .await
            .unwrap();
        let coordination = clients.coordination.unwrap();
        coordination
            .set("written", "1", Duration::from_secs(60))
            .await
            .unwrap();

        server.restart().await.unwrap();

        assert!(eventually_pings(&coordination).await);
        let buffer = clients.transaction_buffer.unwrap();
        assert!(buffer.ping().await.is_ok());
    }

    #[rstest]
    #[tokio::test]
    async fn an_unreachable_environment_only_redis_leaves_coordination_in_memory() {
        let server = RedisServer::start().await.unwrap();
        let port = port_of(&server);
        server.stop().await.unwrap();
        let config = Config::from_yaml("model_list: []").unwrap();

        let clients = connect(&config, &|name| match name {
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

        let clients = connect(&config, &|name| match name {
            "REDIS_HOST" => Some("127.0.0.1".to_owned()),
            "REDIS_PORT" => Some(port.clone()),
            _ => None,
        })
        .await
        .unwrap();

        assert!(clients.coordination.unwrap().ping().await.is_ok());
    }
}
