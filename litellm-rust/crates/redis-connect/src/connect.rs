use std::time::Duration;

use litellm_circuit_breaker::Breaker;
use litellm_redis_client::{
    Credentials, DEFAULT_TIMEOUT, Namespace, RedisClient, RedisConfig, Target, Timeouts,
};
use litellm_redis_types::{Auth, HealthCheck, Resolution, StaticAuth};

use crate::Error;

// litellm/constants.py REDIS_CLUSTER_HEALTH_CHECK_INTERVAL, the default _redis.py sets for cluster clients only
const CLUSTER_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(25);

pub fn connect(resolved: Resolution, breaker: Option<Breaker>) -> Result<RedisClient, Error> {
    tracing::info!(target = ?resolved.target, namespace = ?resolved.namespace, "redis client configured");
    let health_check = health_check(&resolved.target, resolved.health_check);
    Ok(RedisClient::new(RedisConfig {
        target: resolved.target,
        credentials: credentials(resolved.auth)?,
        namespace: resolved.namespace.map(Namespace::new).unwrap_or_default(),
        timeouts: Timeouts::new(
            resolved.timeout.unwrap_or(DEFAULT_TIMEOUT),
            resolved.connect_timeout,
        ),
        breaker,
        events: None,
        health_check,
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

fn credentials(auth: Auth) -> Result<Credentials, Error> {
    match auth {
        Auth::Static(StaticAuth { username, password }) => Ok(Credentials { username, password }),
        Auth::Provider(_) => Err(Error::NoHostCredentialProvider),
        Auth::GcpIam { .. } => Err(Error::CloudAuth {
            setting: "gcp_service_account",
        }),
        Auth::AzureAd(_) => Err(Error::CloudAuth {
            setting: "azure_redis_ad_token",
        }),
        Auth::AwsIam(_) => Err(Error::CloudAuth {
            setting: "aws_iam_auth",
        }),
    }
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
