use std::sync::Arc;

use axum::extract::FromRef;
use litellm_config::{Config, Environment};
use litellm_db::Pools;
use litellm_env::Lookup;
use litellm_gateway_auth::{Auth, BearerOrXApiKey, keys::KeyLookup};
use litellm_license::Entitlements;
use litellm_repositories::KeyRepository;

use crate::{
    Error, Gateway, RedisClients, Resources, db, keys::StoredKeys, license::build_license,
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
    /// The virtual keys of the proxy's database. `None` without a database.
    pub keys: Option<Arc<dyn KeyLookup>>,
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
        let inference = crate::build_inference(config, resources).await?;
        let redis = crate::redis::connect(config, resources).await?;
        let (database, keys) = db::connect(config, &resources.secrets)
            .await?
            .map(|(pools, repositories)| {
                let keys: Arc<dyn KeyLookup> =
                    Arc::new(StoredKeys(KeyRepository::new(repositories)));
                (pools, keys)
            })
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

    /// Who the inference routes authenticate: the master key of `config`, and with a database a
    /// virtual key it holds.
    pub(crate) fn auth(&self, config: &Config) -> Auth {
        let auth = match &self.keys {
            Some(keys) => Auth::with_keys(config, keys.clone()),
            None => Auth::from_config(config),
        };
        auth.with_extractor(Arc::new(BearerOrXApiKey))
    }

    pub async fn close(&self) {
        if let Some(database) = &self.database {
            database.close().await;
        }
    }
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
