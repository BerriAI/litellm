//! The cost map the native app reads. It boots on the bundled map, as Python's import does, and
//! installs the remote one once a background refresh reads it.

use litellm_config::{Config, Environment};
use litellm_http::ClientVariant;
use litellm_model_catalog::IntegrityLimits;
use litellm_model_catalog_refresh::{
    Access, DEFAULT_REGISTRY_URL, GcsAccess, Remote, RemoteSource, RetryPolicy, S3Access, refresh,
};
use litellm_secret_value::SecretValue;

use crate::{AppState, Error};

/// Where the native app looks for a newer cost map than the bundled one.
pub struct CostMapRefresh {
    url: SecretValue,
    source: RemoteSource,
    limits: IntegrityLimits,
}

impl CostMapRefresh {
    /// `None` when `LITELLM_LOCAL_MODEL_COST_MAP` or `LITELLM_OFFLINE_MODE` keeps the bundled map.
    pub fn new(
        config: &Config,
        environment: &Environment,
        state: &AppState,
    ) -> Result<Option<Self>, Error> {
        let url = config
            .litellm_settings
            .model_cost_map_url
            .clone()
            .unwrap_or_else(|| SecretValue::new(DEFAULT_REGISTRY_URL));
        let Remote::At(url) = Remote::new(
            url,
            environment.litellm_local_model_cost_map.0,
            environment.litellm_offline_mode.0,
        ) else {
            return Ok(None);
        };
        let limits = limits(environment)?;
        let inference = &state.inference;
        let client = inference
            .resources
            .pool
            .client(&inference.http, ClientVariant::Provider)?;
        let access = Access {
            s3: S3Access::Environment {
                region: environment.aws_region_name.clone(),
            },
            gcs: GcsAccess::Environment,
        };
        let source = RemoteSource::new(url.expose(), client, access, RetryPolicy::DEFAULT);
        Ok(Some(Self {
            url,
            source,
            limits,
        }))
    }
}

/// `MODEL_COST_MAP_MIN_MODEL_COUNT` and `MODEL_COST_MAP_MAX_SHRINK_RATIO` over the defaults every
/// host shares.
fn limits(environment: &Environment) -> Result<IntegrityLimits, Error> {
    let count = environment
        .model_cost_map_min_model_count
        .map(|count| count as usize);
    IntegrityLimits::over_default(count, environment.model_cost_map_max_shrink_ratio)
        .map_err(Error::ShrinkRatio)
}

impl CostMapRefresh {
    /// Fetches the remote map once and installs it, or keeps the one the native app reads.
    pub async fn run(self, state: AppState) {
        match refresh(&self.source, self.limits)
            .await
            .or_kept(self.url.expose())
        {
            Ok(cost_map) => {
                let models = cost_map.catalog().model_count();
                state.inference.install(cost_map);
                tracing::info!(models, "installed the remote model cost map");
            }
            Err(kept) => tracing::warn!(
                reason = %kept.reason,
                "the remote model cost map was not installed, keeping the one in use"
            ),
        }
    }
}
