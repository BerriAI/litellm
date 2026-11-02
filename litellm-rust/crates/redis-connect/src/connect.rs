use std::sync::Arc;

use litellm_circuit_breaker::{Breaker, Config as BreakerConfig};
use litellm_config::{RedisAuth, ResolvedRedis};
use litellm_core_utils::settings::Lookup;
use litellm_redis_client::{
    Credentials, CredentialsProvider, DEFAULT_TIMEOUT, Namespace, RedisClient, RedisConfig,
    RedisEvents, Timeouts,
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

pub fn connect(resolved: ResolvedRedis, host: HostServices) -> Result<RedisClient, Error> {
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
