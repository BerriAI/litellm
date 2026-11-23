use std::{sync::Arc, time::Duration};

use axum::extract::FromRef;
use futures_util::StreamExt;
use litellm_auth_store::KeyDatabase;
use litellm_config::{Config, Environment};
use litellm_db::Pools;
use litellm_env::Lookup;
use litellm_gateway_auth::Auth;
use litellm_license::Entitlements;
use litellm_redis_client::RedisClient;
use litellm_redis_store::{
    Clock, Remote,
    auth_cache::{AuthCache, Settings},
    cache_invalidation::{self, CacheInvalidation},
    spend_counters::SpendCounters,
    system_clock,
};
use litellm_repositories::{
    Database, GatewayRequestRepository, KeyRepository, ObjectPermissionRepository,
};
use tokio::runtime::Handle;

use crate::{
    Error, Gateway, Inference, KeyLimiter, Metrics, RedisClients, Resources, db,
    gateway_requests::GatewayRequests,
    host::{
        self, NativeRows,
        auth::{AuthStore, KEY_TTL},
    },
    license::build_license,
    periodic::{self, PeriodicJob},
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
    /// Where a virtual key is read from: the auth cache in front of the proxy's database. `None`
    /// without a database.
    pub keys: Option<Arc<AuthStore>>,
    pub entitlements: Arc<Entitlements>,
    /// The proxy's `parallel_request_limiter_v3`, which the inference routes' callback stage runs
    /// and the limit check before the body asks.
    pub key_limiter: Arc<KeyLimiter>,
    /// The Prometheus logger's metrics, when the config lists it.
    pub metrics: Option<Arc<Metrics>>,
    /// The config's `environment_variables` in front of the process environment.
    pub environment: Arc<dyn Lookup + Send + Sync>,
    /// The gateway request counts the request volume middleware records. `None` without a
    /// database.
    pub gateway_requests: Option<Arc<GatewayRequests>>,
    /// The job that writes `gateway_requests` every `PROXY_BATCH_WRITE_AT` seconds.
    pub gateway_requests_flush: Option<Arc<PeriodicJob>>,
}

impl AppState {
    pub async fn build(config: &Config, environment: &Environment) -> Result<Self, Error> {
        Self::over(config, environment, &Resources::build(config)?).await
    }

    /// The state `config` describes, built from the `resources` made for it.
    pub async fn over(
        config: &Config,
        environment: &Environment,
        resources: &Resources,
    ) -> Result<Self, Error> {
        let redis = crate::redis::connect(config, resources).await?;
        let routing = crate::redis::routing(&redis, environment);
        let Inference {
            gateway: inference,
            key_limiter,
            metrics,
        } = crate::build_inference(config, environment, resources, routing, &redis).await?;
        let (database, connected) = db::connect(config, &resources.secrets)
            .await?
            .map(|(pools, repositories)| {
                let requests = Arc::new(GatewayRequests::new());
                let flush = gateway_requests_flush(
                    requests.clone(),
                    GatewayRequestRepository::new(repositories.clone()),
                    environment,
                );
                let keys = Arc::new(keys(config, &redis, repositories));
                (pools, (keys, (requests, Arc::new(flush))))
            })
            .unzip();
        let (keys, gateway_requests) = connected.unzip();
        let (gateway_requests, gateway_requests_flush) = gateway_requests.unzip();
        let entitlements = Arc::new(build_license(config, environment, resources).await);
        Ok(Self {
            inference,
            redis,
            database,
            keys,
            entitlements,
            key_limiter,
            metrics,
            environment: resources.environment.clone(),
            gateway_requests,
            gateway_requests_flush,
        })
    }

    /// Who the inference routes and `admit` authenticate, which the core decides over the native
    /// host: the master key of `config`, and with a database a virtual key it holds.
    pub(crate) fn auth(&self, config: &Config) -> Auth<NativeRows> {
        host::auth::auth(config, self.keys.clone())
    }

    /// Writes the gateway request counts still in memory, then closes the database pools.
    pub async fn close(&self) {
        if let Some(flush) = &self.gateway_requests_flush {
            flush.close().await;
        }
        if let Some(database) = &self.database {
            database.close().await;
        }
    }
}

/// `update_gateway_requests_job`: flushes `requests` through `repository` every batch write
/// interval and once more on shutdown.
fn gateway_requests_flush(
    requests: Arc<GatewayRequests>,
    repository: GatewayRequestRepository,
    environment: &Environment,
) -> PeriodicJob {
    let repository = Arc::new(repository);
    PeriodicJob::start(periodic::batch_write_interval(environment), move || {
        let (requests, repository) = (requests.clone(), repository.clone());
        async move { requests.flush(&repository).await }
    })
}

/// Where the gateway reads a virtual key: through the auth cache Python's `UserApiKeyCache` is,
/// this process's memory in front of the database, with the coordination Redis between them when
/// `enable_redis_auth_cache` attaches it. Every pod's evictions reach the memory whenever a
/// coordination Redis carries them, as Python subscribes with the flag on or off.
pub(crate) fn keys(config: &Config, redis: &RedisClients, database: Database) -> AuthStore {
    let database = KeyDatabase::new(
        Arc::new(KeyRepository::new(database.clone())),
        Arc::new(ObjectPermissionRepository::new(database)),
    );
    let cache = Arc::new(auth_cache(config, redis, system_clock()));
    if let Some(client) = &redis.coordination {
        hear_evictions(&cache, client);
    }
    AuthStore::new(cache, database)
}

/// Drops from this process's memory every key a pod evicts, Python or native, for as long as
/// `cache` lives. The store keeps a copy of each key it reads there, and without this a blocked
/// key would keep answering from it until the copy expired.
fn hear_evictions(cache: &Arc<AuthCache>, client: &RedisClient) {
    let channel = CacheInvalidation::new(
        client.clone(),
        client.namespace().as_str(),
        Handle::current(),
    );
    let mut evictions = channel.subscribe();
    let cache = Arc::downgrade(cache);
    let spend_counters = SpendCounters::new(None);
    tokio::spawn(async move {
        while let Some(invalidation) = evictions.next().await {
            let Some(cache) = cache.upgrade() else {
                return;
            };
            cache_invalidation::apply(&invalidation, &cache, &spend_counters);
        }
    });
}

/// Python's `UserApiKeyCache`, with the coordination Redis attached when
/// `enable_redis_auth_cache` is on. Its memory keeps a Redis hit for Python's
/// `default_in_memory_ttl`, and every write the gateway makes names its own TTL, so no Redis
/// write takes the default.
pub(crate) fn auth_cache(config: &Config, redis: &RedisClients, now: Clock) -> AuthCache {
    const NEVER_TAKEN: Duration = Duration::ZERO;
    let remote = redis
        .coordination
        .clone()
        .filter(|_| config.litellm_settings.enable_redis_auth_cache == Some(true))
        .map(|client| Remote {
            client,
            default_ttl: NEVER_TAKEN,
        });
    let settings = Settings {
        memory_ttl: Some(KEY_TTL),
        ..Settings::default()
    };
    AuthCache::with_clock(remote, settings, now)
}

impl FromRef<AppState> for RedisClients {
    fn from_ref(state: &AppState) -> Self {
        state.redis.clone()
    }
}

impl FromRef<AppState> for Option<Pools> {
    fn from_ref(state: &AppState) -> Self {
        state.database.clone()
    }
}
