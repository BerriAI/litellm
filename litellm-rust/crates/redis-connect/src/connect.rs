use std::{sync::Arc, time::Duration};

use litellm_circuit_breaker::{Breaker, Config as BreakerConfig};
use litellm_core_utils::settings::Lookup;
use litellm_redis_client::{
    Credentials, CredentialsProvider, DEFAULT_TIMEOUT, Namespace, RedisClient, RedisConfig,
    RedisEvents, Target, Timeouts,
};
use litellm_redis_types::{Auth, HealthCheck, Resolution, StaticAuth};

use crate::{
    CredentialSource, Error, RefreshingCredentials,
    sources::{AzureAd, ElastiCacheIam, GcpIam},
};

pub type SharedEnvironment = Arc<dyn Lookup + Send + Sync>;

pub struct HostServices {
    pub events: Option<Arc<dyn RedisEvents>>,
    pub http: litellm_http::Client,
    pub environment: SharedEnvironment,
    pub credential_provider: Option<Arc<dyn CredentialSource>>,
}

// litellm/constants.py REDIS_CLUSTER_HEALTH_CHECK_INTERVAL, the default _redis.py sets for cluster clients only
const CLUSTER_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(25);

pub fn connect(
    resolved: Resolution,
    host: HostServices,
    breaker: Option<BreakerConfig>,
) -> Result<RedisClient, Error> {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    tracing::info!(target = ?resolved.target, namespace = ?resolved.namespace, "redis client configured");
    let credentials = credentials(resolved.auth, &host)?;
    let events = host.events;
    let breaker = breaker.map(|config| {
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
    let health_check = health_check(&resolved.target, resolved.health_check);
    Ok(RedisClient::new(RedisConfig {
        target: resolved.target,
        credentials,
        namespace: resolved.namespace.map(Namespace::new).unwrap_or_default(),
        timeouts: Timeouts::new(
            resolved.timeout.unwrap_or(DEFAULT_TIMEOUT),
            resolved.connect_timeout,
        ),
        breaker,
        events,
        health_check,
        keepalive: resolved.keepalive,
    })?)
}

fn health_check(target: &Target, configured: HealthCheck) -> Option<Duration> {
    match (target, configured) {
        (_, HealthCheck::Every(interval)) => Some(interval),
        (_, HealthCheck::Disabled) => None,
        (Target::Cluster(_), HealthCheck::Unset) => Some(CLUSTER_HEALTH_CHECK_INTERVAL),
        (_, HealthCheck::Unset) => None,
    }
}

fn credentials(auth: Auth, host: &HostServices) -> Result<Credentials, Error> {
    let refreshing = |provider: RefreshingCredentials| {
        Credentials::Provider(CredentialsProvider(Arc::new(provider)))
    };
    Ok(match auth {
        Auth::Static(StaticAuth { username, password }) => {
            Credentials::Static { username, password }
        }
        Auth::Provider(_) => match &host.credential_provider {
            Some(source) => refreshing(RefreshingCredentials::new(Arc::clone(source))),
            None => return Err(Error::NoHostCredentialProvider),
        },
        Auth::AzureAd(auth) => refreshing(RefreshingCredentials::new(AzureAd::new(
            auth,
            host.environment.clone(),
        ))),
        Auth::AwsIam(auth) => refreshing(RefreshingCredentials::new(ElastiCacheIam::new(
            auth,
            &*host.environment,
        )?)),
        Auth::GcpIam { service_account } => refreshing(RefreshingCredentials::new(GcpIam::new(
            host.http.clone(),
            service_account,
            host.environment.clone(),
        ))),
    })
}

#[cfg(test)]
mod tests {
    use litellm_redis_client::{Cluster, Node, Standalone};
    use rstest::rstest;

    use super::*;

    fn cluster() -> Target {
        Target::Cluster(Cluster {
            nodes: vec![Node {
                host: "node".to_owned(),
                port: 7000,
            }],
            tls: None,
        })
    }

    fn standalone() -> Target {
        Target::Standalone(Standalone {
            host: "redis".to_owned(),
            port: 6379,
            tls: None,
            db: 0,
        })
    }

    #[rstest]
    #[case::an_unset_cluster_takes_the_python_default(
        cluster(),
        HealthCheck::Unset,
        Some(CLUSTER_HEALTH_CHECK_INTERVAL)
    )]
    #[case::zero_keeps_a_cluster_off(cluster(), HealthCheck::Disabled, None)]
    #[case::an_unset_standalone_stays_off(standalone(), HealthCheck::Unset, None)]
    #[case::a_configured_interval_wins(
        cluster(),
        HealthCheck::Every(Duration::from_secs(5)),
        Some(Duration::from_secs(5))
    )]
    #[case::a_negative_interval_checks_every_command(
        standalone(),
        HealthCheck::Every(Duration::ZERO),
        Some(Duration::ZERO)
    )]
    fn health_checks_follow_redis_py(
        #[case] target: Target,
        #[case] configured: HealthCheck,
        #[case] expected: Option<Duration>,
    ) {
        assert_eq!(health_check(&target, configured), expected);
    }
}
