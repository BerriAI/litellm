mod params;

use std::time::Duration;

use litellm_circuit_breaker::{Breaker, Config as BreakerConfig, State};
use litellm_config::{Config, Value};
use litellm_redis_client::{RedisClient, RedisConfig};

pub(crate) use params::Lookup;
use params::{Params, cache_block, environment_has_target, resolve};

use crate::Error;

const ENVIRONMENT_PING_TIMEOUT: Duration = Duration::from_secs(2);

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
        .map(|block| resolve(&cache_block(block), lookup))
        .transpose()?;
    let coordination = match &config.general_settings.coordination_redis {
        Some(block) => Some(Target::Configured(resolve(block, lookup)?)),
        None => match (&response_cache, config.litellm_settings.cache == Some(true)) {
            (Some(cache), _) => Some(Target::Configured(cache.clone())),
            (None, true) => environment(lookup)?.map(Target::Configured),
            (None, false) => environment(lookup)?.map(Target::Environment),
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
        Some(Target::Environment(params)) => reachable(client(params, breaker)?).await,
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

fn redis_cache_params(config: &Config) -> Option<&litellm_config::Object> {
    let settings = &config.litellm_settings;
    let params = settings.cache_params.as_ref()?;
    let is_redis = matches!(params.get("type"), Some(Value::String(kind)) if kind == "redis");
    (settings.cache == Some(true) && is_redis).then_some(params)
}

fn environment(lookup: Lookup<'_>) -> Result<Option<Params>, Error> {
    match environment_has_target(lookup) {
        true => resolve(&Default::default(), lookup).map(Some),
        false => Ok(None),
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
    tracing::info!(endpoint = ?params.endpoint, namespace = ?params.namespace.as_str(), "redis client configured");
    Ok(RedisClient::new(RedisConfig {
        endpoint: params.endpoint,
        namespace: params.namespace,
        timeout: params.timeout,
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

    use litellm_config::Config;
    use litellm_redis_client::{
        ClusterEndpoint, ClusterNode, DEFAULT_TIMEOUT, Endpoint, HostEndpoint, Namespace,
    };
    use litellm_redis_testing::RedisServer;
    use rstest::rstest;

    use super::{Plan, Target, connect, params::Params, plan};
    use crate::Error;

    fn host(name: &str, port: u16) -> Params {
        Params {
            endpoint: Endpoint::Host(HostEndpoint {
                host: name.to_owned(),
                port,
                username: None,
                password: None,
                tls: false,
                db: 0,
            }),
            namespace: Namespace::default(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    fn url(value: &str) -> Params {
        Params {
            endpoint: Endpoint::Url(value.to_owned()),
            namespace: Namespace::default(),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    fn cluster(ports: &[u16]) -> Params {
        Params {
            endpoint: Endpoint::Cluster(ClusterEndpoint {
                nodes: ports
                    .iter()
                    .map(|&port| ClusterNode {
                        host: "node".to_owned(),
                        port,
                    })
                    .collect(),
                username: None,
                password: None,
                tls: false,
            }),
            namespace: Namespace::default(),
            timeout: DEFAULT_TIMEOUT,
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
    #[case::coordination_block(
        "general_settings:\n  coordination_redis:\n    host: coord\n    port: 6380",
        &[],
        configured(host("coord", 6380), None)
    )]
    #[case::coordination_block_wins_over_the_cache_redis(
        "general_settings:\n  coordination_redis:\n    host: coord\nlitellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache",
        &[],
        configured(host("coord", 6379), Some(host("cache", 6379)))
    )]
    #[case::cache_redis_is_borrowed_for_coordination_without_its_namespace(
        "litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache\n    namespace: responses\n    ttl: 600",
        &[],
        configured(host("cache", 6379), Some(host("cache", 6379)))
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
        &[("REDIS_HOST", "env"), ("REDIS_PORT", "6390")],
        Plan { coordination: Some(Target::Environment(host("env", 6390))), transaction_buffer_fallback: None, response_cache: None }
    )]
    #[case::an_explicit_host_drops_the_environment_url(
        "general_settings:\n  coordination_redis:\n    host: coord",
        &[("REDIS_URL", "redis://env:6379")],
        configured(host("coord", 6379), None)
    )]
    #[case::an_explicit_host_keeps_environment_credentials(
        "general_settings:\n  coordination_redis:\n    host: coord",
        &[("REDIS_PASSWORD", "secret"), ("REDIS_SSL", "True")],
        configured(
            Params {
                endpoint: Endpoint::Host(HostEndpoint {
                    host: "coord".to_owned(),
                    port: 6379,
                    username: None,
                    password: Some("secret".to_owned()),
                    tls: true,
                    db: 0,
                }),
                namespace: Namespace::default(),
                timeout: DEFAULT_TIMEOUT,
            },
            None
        )
    )]
    #[case::environment_url_over_environment_host(
        "litellm_settings:\n  cache: true",
        &[("REDIS_URL", "redis://env:6379"), ("REDIS_HOST", "ignored")],
        configured(url("redis://env:6379"), None)
    )]
    #[case::startup_nodes_win_over_a_url(
        "general_settings:\n  coordination_redis:\n    url: redis://single:6379\n    startup_nodes:\n      - host: node\n        port: 7000",
        &[],
        configured(cluster(&[7000]), None)
    )]
    #[case::environment_cluster_nodes(
        "general_settings:\n  coordination_redis:\n    host: ignored",
        &[("REDIS_CLUSTER_NODES", r#"[{"host": "node", "port": "7000"}, {"host": "node", "port": 7001}]"#)],
        configured(cluster(&[7000, 7001]), None)
    )]
    #[case::cache_redis_startup_nodes(
        "litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    redis_startup_nodes:\n      - host: node\n        port: 7000",
        &[],
        configured(cluster(&[7000]), Some(cluster(&[7000])))
    )]
    #[case::os_environ_references_are_resolved(
        "general_settings:\n  coordination_redis:\n    host: os.environ/COORD_HOST\n    port: os.environ/COORD_PORT",
        &[("COORD_HOST", "from-env"), ("COORD_PORT", "6400")],
        configured(host("from-env", 6400), None)
    )]
    #[case::socket_timeout_and_namespace_come_through(
        "general_settings:\n  coordination_redis:\n    host: coord\n    namespace: litellm\n    socket_timeout: 1.5",
        &[],
        configured(
            Params { namespace: Namespace::new("litellm"), timeout: Duration::from_millis(1500), ..host("coord", 6379) },
            None
        )
    )]
    #[case::pool_settings_have_no_meaning_without_a_pool(
        "general_settings:\n  coordination_redis:\n    host: coord\n    max_connections: 50",
        &[("REDIS_HEALTH_CHECK_INTERVAL", "30")],
        configured(host("coord", 6379), None)
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
    #[case::unsupported_environment_kwarg("model_list: []", &[("REDIS_HOST", "env"), ("REDIS_SSL_CERT_REQS", "none")], "ssl_cert_reqs")]
    #[case::sentinel_in_the_block("general_settings:\n  coordination_redis:\n    sentinel_nodes: [[s, 26379]]\n    service_name: mymaster", &[], "sentinel_nodes")]
    #[case::sentinel_in_the_environment("model_list: []", &[("REDIS_SENTINEL_NODES", "[[\"s\", 26379]]")], "sentinel_nodes")]
    #[case::aws_iam("general_settings:\n  coordination_redis:\n    host: coord\n    aws_iam_auth: true", &[], "aws_iam_auth")]
    #[case::gcp_iam_from_the_environment("general_settings:\n  coordination_redis:\n    host: coord", &[("REDIS_GCP_SERVICE_ACCOUNT", "projects/-/serviceAccounts/x")], "gcp_service_account")]
    #[case::azure_ad("litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache\n    azure_redis_ad_token: \"true\"", &[], "azure_redis_ad_token")]
    #[case::custom_ca_on_the_cache("litellm_settings:\n  cache: true\n  cache_params:\n    type: redis\n    host: cache\n    ssl_ca_certs: /ca.pem", &[], "ssl_ca_certs")]
    fn unsupported_settings_fail_with_their_name(
        #[case] yaml: &str,
        #[case] environment: &[(&str, &str)],
        #[case] setting: &str,
    ) {
        let error = resolved(yaml, environment).unwrap_err();

        assert!(
            matches!(&error, Error::RedisSetting(name) if name == setting),
            "{error:?}"
        );
    }

    #[rstest]
    fn a_block_without_a_target_fails() {
        let error = resolved(
            "general_settings:\n  coordination_redis:\n    port: 6379",
            &[],
        )
        .unwrap_err();

        assert!(
            matches!(
                error,
                Error::RedisInvalid {
                    setting: "host",
                    ..
                }
            ),
            "{error:?}"
        );
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
