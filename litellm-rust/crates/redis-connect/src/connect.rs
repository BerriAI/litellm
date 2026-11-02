use std::time::Duration;

use std::sync::Arc;

use litellm_circuit_breaker::{Breaker, Config as BreakerConfig};
use litellm_config::{Expected, RedisAuth, RedisError, ResolvedRedis};
use litellm_core_utils::settings::Lookup;
use litellm_redis_client::{
    Credentials, CredentialsProvider, DEFAULT_TIMEOUT, Endpoint, Namespace, RedisClient,
    RedisConfig, RedisEvents, Timeouts,
};

use crate::{
    Error, RefreshingCredentials,
    sources::{AzureAd, ElastiCacheIam, GcpIam},
};

pub type SharedEnvironment = Arc<dyn Lookup + Send + Sync>;

pub struct HostServices {
    pub events: Option<Arc<dyn RedisEvents>>,
    pub breaker: Option<BreakerConfig>,
    pub http: litellm_http::Client,
    pub environment: SharedEnvironment,
}

// litellm/constants.py REDIS_CLUSTER_HEALTH_CHECK_INTERVAL, the default _redis.py sets for cluster clients only
const CLUSTER_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(25);

pub fn connect(resolved: ResolvedRedis, host: HostServices) -> Result<RedisClient, Error> {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    tracing::info!(endpoint = ?resolved.endpoint, namespace = ?resolved.namespace, "redis client configured");
    let credentials = credentials(resolved.auth, &host)?;
    let events = host.events;
    let breaker = host.breaker.map(|config| {
        let observer = events.clone();
        Breaker::new(
            config,
            Box::new(move |transition| {
                if let Some(events) = &observer {
                    events.breaker(transition);
                }
            }),
        )
    });
    let health_check = match &resolved.endpoint {
        Endpoint::Cluster(_) => resolved
            .health_check_interval
            .or(Some(CLUSTER_HEALTH_CHECK_INTERVAL)),
        _ => resolved.health_check_interval,
    }
    .filter(|interval| !interval.is_zero());
    Ok(RedisClient::new(RedisConfig {
        endpoint: resolved.endpoint,
        credentials,
        namespace: resolved.namespace.map(Namespace::new).unwrap_or_default(),
        timeouts: Timeouts::new(
            resolved.timeout.unwrap_or(DEFAULT_TIMEOUT),
            resolved.connect_timeout,
        ),
        breaker,
        events,
        health_check,
    })?)
}

fn credentials(auth: RedisAuth, host: &HostServices) -> Result<Credentials, Error> {
    let refreshing = |provider: RefreshingCredentials| {
        Credentials::Provider(CredentialsProvider(Arc::new(provider)))
    };
    Ok(match auth {
        RedisAuth::Static { username, password } => Credentials::Static { username, password },
        RedisAuth::Host => return Err(Error::NoHostCredentialProvider),
        RedisAuth::AzureAd(auth) => refreshing(RefreshingCredentials::new(AzureAd::new(
            auth,
            host.environment.clone(),
        ))),
        RedisAuth::AwsIam(auth) => refreshing(RefreshingCredentials::new(ElastiCacheIam::new(
            auth,
            &*host.environment,
        )?)),
        RedisAuth::GcpIam { service_account } => refreshing(RefreshingCredentials::new(
            GcpIam::new(host.http.clone(), service_account, host.environment.clone()),
        )),
    })
}

/// The breaker `RedisCache` builds from `REDIS_CIRCUIT_BREAKER_*`, or none when it is disabled.
pub fn breaker_config(lookup: &dyn Lookup) -> Result<Option<BreakerConfig>, RedisError> {
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

fn parse_seconds(text: &str) -> Option<Duration> {
    text.parse::<f64>()
        .ok()
        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
}

fn environment_value<T>(
    lookup: &dyn Lookup,
    name: &str,
    expected: Expected,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, RedisError> {
    lookup
        .get(name)
        .map(|value| {
            parse(&value).ok_or_else(|| RedisError::Invalid {
                setting: name.to_owned(),
                expected,
            })
        })
        .transpose()
}
