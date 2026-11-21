use std::sync::Arc;

use axum::extract::FromRef;
use litellm_auth_store::KeyDatabase;
use litellm_config::{Config, Environment};
use litellm_db::Pools;
use litellm_env::Lookup;
use litellm_gateway_auth::Auth;
use litellm_license::Entitlements;
use litellm_repositories::{Database, KeyRepository, ObjectPermissionRepository};

use crate::{
    Error, Gateway, RedisClients, Resources, db,
    host::{self, NativeRows},
    license::build_license,
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
    /// The key read over the proxy's database, which a virtual key is authenticated from. `None`
    /// without a database.
    pub keys: Option<Arc<KeyDatabase>>,
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
            .map(|(pools, repositories)| (pools, Arc::new(keys(repositories))))
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

/// The key read over the proxy's database, which the gateway makes on every request.
pub(crate) fn keys(database: Database) -> KeyDatabase {
    KeyDatabase::new(
        Arc::new(KeyRepository::new(database.clone())),
        Arc::new(ObjectPermissionRepository::new(database)),
    )
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
