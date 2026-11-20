use std::sync::Arc;

use axum::extract::FromRef;
use litellm_config::{Config, Environment};
use litellm_db::Pools;
use litellm_env::Lookup;
use litellm_gateway_inference::Gateway;
use litellm_license::Entitlements;

use crate::{Error, RedisClients, Resources, db, license::build_license};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
    pub entitlements: Arc<Entitlements>,
    /// The config's `environment_variables` in front of the process environment.
    pub environment: Arc<dyn Lookup + Send + Sync>,
}

impl AppState {
    pub async fn build(config: &Config, environment: &Environment) -> Result<Self, Error> {
        let resources = Resources::build(config)?;
        let inference = crate::build_inference(config, &resources)?;
        let redis = crate::redis::connect(config, &resources).await?;
        let database = db::connect(config, &resources.secrets).await?;
        let entitlements = Arc::new(build_license(config, environment, &resources).await);
        Ok(Self {
            inference,
            redis,
            database,
            entitlements,
            environment: resources.environment,
        })
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
