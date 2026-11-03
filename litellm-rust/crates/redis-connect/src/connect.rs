use std::time::Duration;

use litellm_circuit_breaker::Breaker;
use litellm_config::{RedisAuth, ResolvedRedis};
use litellm_redis_client::{
    Credentials, DEFAULT_TIMEOUT, Endpoint, Namespace, RedisClient, RedisConfig, Timeouts,
};

use crate::Error;

// litellm/constants.py REDIS_CLUSTER_HEALTH_CHECK_INTERVAL, the default _redis.py sets for cluster clients only
const CLUSTER_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(25);

pub fn connect(resolved: ResolvedRedis, breaker: Option<Breaker>) -> Result<RedisClient, Error> {
    tracing::info!(endpoint = ?resolved.endpoint, namespace = ?resolved.namespace, "redis client configured");
    let health_check = match &resolved.endpoint {
        Endpoint::Cluster(_) => resolved
            .health_check_interval
            .or(Some(CLUSTER_HEALTH_CHECK_INTERVAL)),
        _ => resolved.health_check_interval,
    }
    .filter(|interval| !interval.is_zero());
    Ok(RedisClient::new(RedisConfig {
        endpoint: resolved.endpoint,
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

fn credentials(auth: RedisAuth) -> Result<Credentials, Error> {
    match auth {
        RedisAuth::Static { username, password } => Ok(Credentials { username, password }),
        RedisAuth::Host => Err(Error::NoHostCredentialProvider),
        RedisAuth::GcpIam { .. } => Err(Error::CloudAuth {
            setting: "gcp_service_account",
        }),
        RedisAuth::AzureAd(_) => Err(Error::CloudAuth {
            setting: "azure_redis_ad_token",
        }),
        RedisAuth::AwsIam(_) => Err(Error::CloudAuth {
            setting: "aws_iam_auth",
        }),
    }
}
