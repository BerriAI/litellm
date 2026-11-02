use litellm_circuit_breaker::Breaker;
use litellm_config::{RedisAuth, ResolvedRedis};
use litellm_redis_client::{
    Credentials, DEFAULT_TIMEOUT, Namespace, RedisClient, RedisConfig, Timeouts,
};

use crate::Error;

pub fn connect(resolved: ResolvedRedis, breaker: Option<Breaker>) -> Result<RedisClient, Error> {
    tracing::info!(endpoint = ?resolved.endpoint, namespace = ?resolved.namespace, "redis client configured");
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
