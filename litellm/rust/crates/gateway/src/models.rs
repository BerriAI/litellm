//! The gateway's live model table. It boots on the bundled cost map, then a background fetch of
//! the remote one (`LITELLM_MODEL_COST_MAP_URL`) publishes the table it gives, as Python's
//! import-time fetch installs the remote map over the bundled backup.

use std::{sync::Arc, time::Duration};

use crate::Gateway;
use litellm_core_utils::settings::Lookup;
use litellm_http::Client;
use litellm_model_catalog::{CostMap, IntegrityLimits};
use litellm_model_catalog_refresh::{
    DEFAULT_REGISTRY_URL, GcsAccess, RetryPolicy, S3Access, Start, fetch,
};

use crate::boot::Shaper;

/// `get_model_cost_map`'s fetch: 3 attempts of 5 seconds, backing off from 2 up to 30 seconds.
const POLICY: RetryPolicy = RetryPolicy {
    max_attempts: 3,
    timeout: Duration::from_secs(5),
    min_delay: Duration::from_secs(2),
    max_delay: Duration::from_secs(30),
};

struct Remote {
    url: String,
    client: Client,
    s3: S3Access,
    limits: IntegrityLimits,
}

/// What republishes the model table when the remote cost map changes.
pub struct ModelRefresh {
    shaper: Shaper,
    /// `None` when the bundled map is the only one (`LITELLM_LOCAL_MODEL_COST_MAP`,
    /// `LITELLM_OFFLINE_MODE`).
    remote: Option<Remote>,
}

/// Whether `ModelRefresh::refresh` published a new table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refreshed {
    Published,
    Unchanged,
    Kept,
}

fn says_true(value: Option<String>) -> bool {
    value.is_some_and(|value| value.trim().eq_ignore_ascii_case("true"))
}

impl ModelRefresh {
    pub(crate) fn new(shaper: Shaper, env: &impl Lookup, client: Client) -> Self {
        let bundled_only = says_true(env.get("LITELLM_LOCAL_MODEL_COST_MAP"))
            || says_true(env.get("LITELLM_OFFLINE_MODE"));
        let limits = IntegrityLimits::new(
            env.get("MODEL_COST_MAP_MIN_MODEL_COUNT")
                .and_then(|count| count.trim().parse().ok())
                .unwrap_or(50),
            env.get("MODEL_COST_MAP_MAX_SHRINK_RATIO")
                .and_then(|ratio| ratio.trim().parse().ok())
                .unwrap_or(0.5),
        );
        let remote = limits.filter(|_| !bundled_only).map(|limits| Remote {
            url: env
                .get("LITELLM_MODEL_COST_MAP_URL")
                .unwrap_or_else(|| DEFAULT_REGISTRY_URL.to_owned()),
            client,
            s3: S3Access::Environment {
                region: env.get("AWS_REGION_NAME"),
            },
            limits,
        });
        Self { shaper, remote }
    }

    /// The model table `cost_map` gives.
    pub(crate) fn table(&self, cost_map: CostMap) -> litellm_gateway_inference::ModelTable {
        self.shaper.table(cost_map)
    }

    /// Refresh now, then every `every` when it is set.
    pub async fn run(self, gateway: Arc<Gateway>, every: Option<Duration>) {
        loop {
            self.refresh(&gateway).await;
            let Some(every) = every else {
                return;
            };
            tokio::time::sleep(every).await;
        }
    }

    /// Fetch the remote cost map and publish the table it gives, unless it is the one installed.
    /// A fetch that fails or a map that is refused keeps the table in force.
    pub async fn refresh(&self, gateway: &Gateway) -> Refreshed {
        let Some(remote) = &self.remote else {
            return Refreshed::Kept;
        };
        let fetched = match fetch(
            &remote.client,
            &remote.s3,
            &GcsAccess::Environment,
            &remote.url,
            POLICY,
            Start::Fresh,
        )
        .await
        {
            Ok(fetched) => fetched,
            Err(failed) => {
                litellm_tracing::warn!(
                    "LiteLLM: model cost map fetch failed: {}. Keeping the currently loaded map.",
                    failed.reason
                );
                return Refreshed::Kept;
            }
        };
        let cost_map = match CostMap::read(&fetched.body, fetched.etag, remote.limits) {
            Ok(cost_map) => cost_map,
            Err(error) => {
                litellm_tracing::warn!(
                    "LiteLLM: fetched model cost map was rejected ({}). Keeping the currently loaded map.",
                    litellm_error::Report::new(&error)
                );
                return Refreshed::Kept;
            }
        };
        if cost_map.provenance().revision == gateway.models().cost_map.provenance().revision {
            return Refreshed::Unchanged;
        }
        gateway.publish(self.shaper.table(cost_map));
        Refreshed::Published
    }
}
