use litellm_circuit_breaker::Breaker;
use litellm_redis_client::{
    Credentials, DEFAULT_TIMEOUT, Namespace, RedisClient, RedisConfig, Timeouts,
};
use litellm_redis_types::{Auth, Resolution, StaticAuth};

use crate::Error;

pub fn connect(resolved: Resolution, breaker: Option<Breaker>) -> Result<RedisClient, Error> {
    tracing::info!(target = ?resolved.target, namespace = ?resolved.namespace, "redis client configured");
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
    })?)
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
