use std::{sync::Arc, time::Duration};

use axum::extract::FromRef;
use futures_util::StreamExt;
use litellm_auth_store::KeyDatabase;
use litellm_config::{Config, Environment};
use litellm_db::Pools;
use litellm_env::{Lookup, ProcessEnvironment};
use litellm_gateway_auth::Auth;
use litellm_license::Entitlements;
use litellm_redis_client::RedisClient;
use litellm_redis_store::{
    Clock, Remote,
    auth_cache::{AuthCache, Settings},
    cache_invalidation::{self, CacheInvalidation},
    rate_limits::RateLimits,
    spend_counters::SpendCounters,
    system_clock,
};
use litellm_repositories::{Database, KeyRepository, ObjectPermissionRepository};
use tokio::runtime::Handle;

use crate::{
    Error, Gateway, Inference, KeyLimiter, RedisClients, Resources,
    callbacks::parallel_request_limiter,
    db,
    host::{
        self, NativeRows,
        auth::{AuthStore, KEY_TTL},
    },
    license::build_license,
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
    /// The config's `environment_variables` in front of the process environment.
    pub environment: Arc<dyn Lookup + Send + Sync>,
    /// What the key limits and model group capacity count in, in this process's memory when there
    /// is no coordination Redis.
    pub counters: Arc<RateLimits>,
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
        let counters = parallel_request_limiter::counters(redis.coordination.clone());
        let Inference {
            gateway: inference,
            key_limiter,
        } = crate::inference_over(
            config,
            environment,
            resources,
            routing,
            &redis,
            counters.clone(),
        )
        .await?;
        let (database, keys) = db::connect(config, &resources.secrets)
            .await?
            .map(|(pools, repositories)| (pools, Arc::new(keys(config, &redis, repositories))))
            .unzip();
        let entitlements = Arc::new(build_license(config, environment, resources).await);
        Ok(Self {
            inference,
            redis,
            database,
            keys,
            entitlements,
            key_limiter,
            environment: resources.environment.clone(),
            counters,
        })
    }

    /// The state `config` describes, built over what this one keeps for the life of the process:
    /// the Redis clients, the database pools and the keys read through them, the license, the
    /// HTTP client pool, the deployments' routing state and the limiters' counters. Only what
    /// `config` decides is built again, its environment, router and callbacks.
    pub async fn next(&self, config: &Config, environment: &Environment) -> Result<Self, Error> {
        let resources = Resources::sharing(
            config,
            Arc::new(ProcessEnvironment),
            self.inference.resources.clone(),
        )?;
        let Inference {
            gateway: inference,
            key_limiter,
        } = crate::inference_over(
            config,
            environment,
            &resources,
            self.inference.shared_routing(),
            &self.redis,
            self.counters.clone(),
        )
        .await?;
        Ok(Self {
            inference,
            key_limiter,
            environment: resources.environment,
            ..self.clone()
        })
    }

    /// Who the inference routes and `admit` authenticate, which the core decides over the native
    /// host: the master key of `config`, and with a database a virtual key it holds.
    pub(crate) fn auth(&self, config: &Config) -> Auth<NativeRows> {
        host::auth::auth(config, self.keys.clone())
    }

    pub async fn close(&self) {
        if let Some(database) = &self.database {
            database.close().await;
        }
    }
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
