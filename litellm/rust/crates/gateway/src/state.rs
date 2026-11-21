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
    Remote,
    auth_cache::{AuthCache, Settings},
    cache_invalidation::{self, CacheInvalidation},
    spend_counters::SpendCounters,
};
use litellm_repositories::{Database, KeyRepository, ObjectPermissionRepository};
use tokio::runtime::Handle;

use crate::{
    Error, Gateway, RedisClients, Resources, db,
    host::{
        self, NativeRows,
        auth::{AuthStore, Keys},
    },
    license::build_license,
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
    /// Where a virtual key is read from: the auth cache shared with the Python pods in front of
    /// the proxy's database, or that database alone. `None` without a database.
    pub keys: Option<Arc<Keys>>,
    pub entitlements: Arc<Entitlements>,
    /// The config's `environment_variables` in front of the process environment.
    pub environment: Arc<dyn Lookup + Send + Sync>,
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
        let inference = crate::build_inference(config, resources, routing).await?;
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
            environment: resources.environment.clone(),
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

/// How long Python keeps a key it cached, its `DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`.
const KEY_TTL: Duration = Duration::from_secs(60);

/// Where the gateway reads a virtual key: through the auth cache it shares with the Python pods
/// when `enable_redis_auth_cache` attaches one, else from the database on every request.
pub(crate) fn keys(config: &Config, redis: &RedisClients, database: Database) -> Keys {
    let database = KeyDatabase::new(
        Arc::new(KeyRepository::new(database.clone())),
        Arc::new(ObjectPermissionRepository::new(database)),
    );
    match (auth_cache(config, redis), &redis.coordination) {
        (Some(cache), Some(client)) => {
            let cache = Arc::new(cache);
            hear_evictions(&cache, client);
            Keys::Shared(Arc::new(AuthStore::new(cache, database, KEY_TTL)))
        }
        _ => Keys::Database(database),
    }
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

/// Python's `UserApiKeyCache` on the coordination Redis, which `enable_redis_auth_cache` attaches
/// it to. Every write the gateway makes names its own TTL, so none takes the default.
pub(crate) fn auth_cache(config: &Config, redis: &RedisClients) -> Option<AuthCache> {
    const NEVER_TAKEN: Duration = Duration::ZERO;
    let client = redis.coordination.clone()?;
    (config.litellm_settings.enable_redis_auth_cache == Some(true)).then(|| {
        AuthCache::new(
            Some(Remote {
                client,
                default_ttl: NEVER_TAKEN,
            }),
            Settings::default(),
        )
    })
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
