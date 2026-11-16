use std::sync::Arc;

use axum::extract::FromRef;
use litellm_config::Config;
use litellm_db::Pools;

use crate::{Error, Gateway, RedisClients, Resources, db};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
}

impl AppState {
    pub async fn build(config: &Config) -> Result<Self, Error> {
        let resources = Resources::build(config)?;
        let inference = crate::build_inference(config, &resources).await?;
        let redis = crate::redis::connect(config, &resources).await?;
        let database = db::connect(config, &resources.secrets).await?;
        Ok(Self {
            inference,
            redis,
            database,
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
